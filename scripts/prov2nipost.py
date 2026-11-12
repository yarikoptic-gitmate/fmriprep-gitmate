#!/usr/bin/env python
"""Redo the nipost resampling(s) described in an fMRIPrep provenance.jsonld file.

For every node that ran fMRIPrep's ``ResampleSeries`` interface (a thin wrapper around
``nipost.resample_image``), the arguments of the call are recovered from the PROV
record, the inputs are verified against their recorded checksums, the resampling is
redone with nipost alone (no fMRIPrep, no nipype), and the result is compared with
the checksum that was recorded for the original output.

Example::

    prov2nipost.py out/sub-01/log/<run_uuid>/provenance.jsonld \
        --map /work=/host/path/to/work --map /data=/host/path/to/bids \
        --node bold_std_wf --out-dir redo/

"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import rdflib

NS = {
    'prov': 'http://www.w3.org/ns/prov#',
    'dcterms': 'http://purl.org/dc/terms/',
    'nipype': 'http://nipy.org/nipype/terms/',
    'crypto': 'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/',
}

NODES = """
SELECT ?node ?interface ?started WHERE {
    ?node a nipype:ResampleSeries .
    ?interface dcterms:isPartOf ?node .
    OPTIONAL { ?node prov:startedAtTime ?started }
} ORDER BY ?started ?node
"""

PORTS = """
SELECT ?dir ?port ?value ?location ?sha512 WHERE {
    {
        ?interface prov:qualifiedUsage [ prov:entity ?entity ; nipype:inPort ?port ] .
        BIND("in" AS ?dir)
    } UNION {
        ?entity prov:qualifiedGeneration [ prov:activity ?interface ; nipype:outPort ?port ] .
        BIND("out" AS ?dir)
    }
    OPTIONAL { ?entity prov:value ?value }
    OPTIONAL { ?entity prov:atLocation ?location }
    OPTIONAL { ?entity crypto:sha512 ?sha512 }
}
"""

# The checksum of a file with a given location (files of a list are members of that list)
CHECKSUM = """
SELECT ?sha512 WHERE {
    ?entity prov:atLocation ?location ; crypto:sha512 ?sha512 .
    FILTER(STR(?location) = ?wanted)
}
"""


def sha512(path):
    digest = hashlib.sha512()
    with open(path, 'rb') as fobj:
        for chunk in iter(lambda: fobj.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Paths:
    """Translate the paths that were recorded into paths available here."""

    def __init__(self, maps):
        self.maps = [tuple(m.split('=', 1)) for m in maps]

    def __call__(self, recorded):
        recorded = str(recorded)
        if recorded.startswith('file://'):
            recorded = unquote(urlparse(recorded).path)
        for old, new in self.maps:
            if recorded == old or recorded.startswith(old.rstrip('/') + '/'):
                return Path(new + recorded[len(old.rstrip('/')) :])
        return Path(recorded)


def describe(dataset, interface):
    """Collect inputs and outputs of an interface, by name."""
    inputs, outputs = {}, {}
    for row in dataset.query(PORTS, initNs=NS, initBindings={'interface': interface}):
        port = str(row.port)
        value = row.location if row.location is not None else row.value
        value = value.toPython() if value is not None else None
        target = inputs if str(row.dir) == 'in' else outputs
        target[port] = {
            'value': value,
            'is_file': row.location is not None,
            'sha512': str(row.sha512) if row.sha512 is not None else None,
        }
    return inputs, outputs


def recorded_checksum(dataset, path):
    """Find the checksum recorded for a file that is only listed by its path."""
    for row in dataset.query(
        CHECKSUM, initNs=NS, initBindings={'wanted': rdflib.Literal(f'file://{path}')}
    ):
        return str(row.sha512)
    return None


def as_list(value):
    """Lists are recorded as JSON text; a list of one item, as that item."""
    if value is None:
        return []
    if isinstance(value, str) and value.lstrip().startswith('['):
        return json.loads(value)
    return [value]


def check(path, expected, what, problems):
    if not path.exists():
        problems.append(f'{what}: {path} does not exist')
        return
    if expected is None:
        print(f'    {what}: {path}  [no checksum recorded]')
    elif sha512(path) == expected:
        print(f'    {what}: {path}  [checksum OK]')
    else:
        problems.append(f'{what}: {path} differs from what was recorded')


def redo(dataset, node, interface, paths, out_dir, execute):
    inputs, outputs = describe(dataset, interface)
    print(f'\n== {node}')
    problems = []

    def _file(port):
        if port not in inputs:
            return None
        path = paths(inputs[port]['value'])
        check(path, inputs[port]['sha512'], port, problems)
        return path

    in_file = _file('in_file')
    ref_file = _file('ref_file')
    fieldmap = _file('fieldmap')
    fieldmap_gradient = _file('fieldmap_gradient')

    transforms = []
    for recorded in as_list(inputs.get('transforms', {}).get('value')):
        recorded_path = unquote(urlparse(str(recorded)).path)
        checksum = inputs['transforms']['sha512'] or recorded_checksum(dataset, recorded_path)
        transforms.append(paths(recorded))
        check(transforms[-1], checksum, 'transforms', problems)
    inverse = [bool(v) for v in as_list(inputs.get('inverse', {}).get('value'))] or [False]
    if len(inverse) == 1:
        inverse *= max(len(transforms), 1)

    def _scalar(port, default=None):
        return inputs[port]['value'] if port in inputs else default

    pe_dir = _scalar('pe_dir')
    ro_time = _scalar('ro_time')
    kwargs = {
        'nthreads': int(_scalar('num_threads', 1)),
        'output_dtype': _scalar('output_data_type', 'float32'),
        'order': int(_scalar('order', 3)),
        'mode': _scalar('mode', 'grid-constant'),
        'cval': float(_scalar('cval', 0.0)),
        'prefilter': bool(_scalar('prefilter', True)),
    }
    out_recorded = outputs.get('out_file', {})
    # One directory per node: different nodes give their output the same name
    out_dir = out_dir / str(node).rsplit('/', 1)[-1]
    out_file = out_dir / Path(str(out_recorded.get('value', 'resampled.nii.gz'))).name

    # The equivalent invocation, for the record
    print('    invocation:')
    print(f'      source = nb.load({str(in_file)!r})')
    print(f'      target = nb.load({str(ref_file)!r})')
    print(
        f'      transforms = nipost.load_transforms({[str(t) for t in transforms]!r}, {inverse!r})'
    )
    if fieldmap:
        print(
            f'      fieldmap = nb.load({str(fieldmap)!r})  # pe_dir={pe_dir!r}, ro_time={ro_time!r}'
        )
    print(
        '      nipost.resample_image(source, target, transforms, fieldmap, pe_info, '
        + ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
        + f').to_filename({str(out_file)!r})'
    )

    if problems:
        for problem in problems:
            print(f'    PROBLEM {problem}')
        return False
    if not execute:
        return True

    import nibabel as nb
    from nipost import load_transforms, resample_image
    from nipost.epi import ensure_positive_cosines

    source = nb.load(in_file)
    target = nb.load(ref_file)
    nvols = source.shape[3] if source.ndim > 3 else 1
    pe_info = None
    if pe_dir and ro_time:
        # As done by fmriprep.interfaces.resampling.ResampleSeries
        pe_axis = 'ijk'.index(pe_dir[0])
        pe_flip = pe_dir.endswith('-')
        source, axcodes = ensure_positive_cosines(source)
        axis_flip = axcodes[pe_axis] in 'LPI'
        pe_info = [(pe_axis, -ro_time if (axis_flip ^ pe_flip) else ro_time)] * nvols

    resampled = resample_image(
        source=source,
        target=target,
        transforms=load_transforms(transforms, inverse),
        fieldmap=nb.load(fieldmap) if fieldmap else None,
        pe_info=pe_info,
        fieldmap_gradient=nb.load(fieldmap_gradient) if fieldmap_gradient else None,
        **kwargs,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    resampled.to_filename(out_file)

    # Compare with the original
    if out_recorded.get('sha512') and sha512(out_file) == out_recorded['sha512']:
        print(f'    REPRODUCED {out_file}: checksum identical to the original')
        return True
    original = paths(out_recorded['value']) if out_recorded.get('value') else None
    if original is not None and original.exists():
        import numpy as np

        old = nb.load(original)
        same_data = old.shape == resampled.shape and np.array_equal(
            np.asanyarray(old.dataobj), np.asanyarray(nb.load(out_file).dataobj)
        )
        same_affine = np.allclose(old.affine, resampled.affine)
        if same_data and same_affine:
            print(f'    REPRODUCED {out_file}: voxel data identical (file bytes differ)')
            return True
        diff = np.abs(
            np.asanyarray(old.dataobj, dtype='f8') - np.asanyarray(resampled.dataobj, dtype='f8')
        )
        print(f'    DIFFERENT {out_file}: max abs difference {diff.max():g}')
        return False
    print(f'    UNVERIFIED {out_file}: checksum differs and the original is not available')
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('provenance', type=Path, help='provenance.jsonld written by fMRIPrep')
    parser.add_argument(
        '--map',
        action='append',
        default=[],
        metavar='RECORDED=LOCAL',
        help='translate a recorded path prefix (e.g., a container mount)',
    )
    parser.add_argument('--node', default='', help='only nodes whose identifier contains this')
    parser.add_argument('--out-dir', type=Path, default=Path('redo'))
    parser.add_argument('--dry-run', action='store_true', help='only show the invocations')
    args = parser.parse_args(argv)

    dataset = rdflib.Dataset(default_union=True)
    dataset.parse(args.provenance, format='json-ld')
    paths = Paths(args.map)

    results = [
        redo(dataset, row.node, row.interface, paths, args.out_dir, not args.dry_run)
        for row in dataset.query(NODES, initNs=NS)
        if args.node in str(row.node)
    ]
    print(f'\n{sum(results)}/{len(results)} resampling(s) OK')
    return 0 if results and all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
