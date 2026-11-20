#!/usr/bin/env python
"""Redo, from an fMRIPrep provenance.jsonld file, the resampling(s) it describes.

Two modes are available.

**Resamplings** (default).
For every node that ran fMRIPrep's ``ResampleSeries`` interface (a thin wrapper around
``nipost.resample_image``), the arguments of the call are recovered from the PROV
record, the inputs are verified against their recorded checksums, the resampling is
redone with nipost alone (no fMRIPrep, no nipype), and the result is compared with
the checksum that was recorded for the original output::

    prov2nipost.py out/sub-01/log/<run_uuid>/provenance.jsonld \
        --map /work=/host/path/to/work --map /data=/host/path/to/bids \
        --node bold_std_wf --out-dir redo/

**Targets** (``--target``).
Files that were generated are selected by (the end of) their recorded path, e.g., a
derivative in some output space.
The steps that led to each of them are traced back through the records until
files that are still available are reached: typically the working directory is gone,
and what remains are the raw dataset, the derivatives and TemplateFlow.
The steps are then replayed in order, and every regenerated file is compared with its
recorded checksum.
Resamplings are done with nipost as above; other steps (e.g., slice-timing correction,
reconstruction of the fieldmap, and the data sink that names the derivative and writes
its sidecar JSON) are replayed by running the recorded interface with the recorded
inputs, which requires the environment of fMRIPrep (e.g., its container)::

    prov2nipost.py out/sub-01/log/<run_uuid>/provenance.jsonld \
        --target space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz --out-dir redo/

With ``--dry-run`` nothing is executed: the steps are listed, followed by ``INPUT`` and
``OUTPUT`` lines with the files that are needed and those that would be written, e.g.,
to compose a ``datalad run --input ... --output ...`` command.

"""

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
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


def recorded_path(value):
    """Paths are recorded as such, or as file:// URLs."""
    value = str(value)
    if value.startswith('file://'):
        value = unquote(urlparse(value).path)
    return value


class Paths:
    """Translate the paths that were recorded into paths available here."""

    def __init__(self, maps):
        self.maps = [tuple(m.split('=', 1)) for m in maps]
        self.produced = {}  # recorded path -> where it was regenerated

    def __call__(self, recorded):
        recorded = recorded_path(recorded)
        if recorded in self.produced:
            return Path(self.produced[recorded])
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
        print(f'    {what}: {path}  [not verified]')
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
        # What was regenerated here was compared with its record already
        regenerated = path in paths.produced.values()
        check(path, None if regenerated else inputs[port]['sha512'], port, problems)
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


STEPS = """
SELECT ?node ?interface ?module ?cls ?command WHERE {
    ?interface dcterms:isPartOf ?node ; nipype:module ?module ; nipype:interface ?cls .
    OPTIONAL { ?interface nipype:command ?command }
}
"""

ALL_PORTS = PORTS.replace('SELECT ', 'SELECT ?interface ')

EDGES = 'SELECT ?node ?upstream WHERE { ?node prov:wasInformedBy ?upstream }'

CHECKSUMS = (
    'SELECT ?location ?sha512 WHERE { ?e prov:atLocation ?location ; crypto:sha512 ?sha512 }'
)


def file_items(item):
    """Paths of the files (or directories) among the value(s) of an input or output."""
    if item['is_file']:
        return [recorded_path(item['value'])]
    return [
        recorded_path(value)
        for value in as_list(item['value'])
        if isinstance(value, str) and value.startswith(('/', 'file:///'))
    ]


class Records:
    """What a provenance file tells about steps (executed interfaces) and their files."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.steps = {}
        for row in dataset.query(STEPS, initNs=NS):
            self.steps[row.interface] = {
                'interface': row.interface,
                'node': row.node,
                'name': str(row.node).rsplit('/', 1)[-1],
                'module': str(row.module),
                'cls': str(row.cls),
                'command': str(row.command) if row.command is not None else None,
                'inputs': {},
                'outputs': {},
            }
        for row in dataset.query(ALL_PORTS, initNs=NS):
            value = row.location if row.location is not None else row.value
            if row.interface in self.steps and value is not None:
                key = 'inputs' if str(row.dir) == 'in' else 'outputs'
                self.steps[row.interface][key][str(row.port)] = {
                    'value': value.toPython(),
                    'is_file': row.location is not None,
                    'sha512': str(row.sha512) if row.sha512 is not None else None,
                }
        self.checksums = {
            recorded_path(row.location): str(row.sha512)
            for row in dataset.query(CHECKSUMS, initNs=NS)
        }
        self.upstream = {}
        for row in dataset.query(EDGES, initNs=NS):
            self.upstream.setdefault(row.node, set()).add(row.upstream)
        # Which step generated a file
        self.generators = {}
        for step in self.steps.values():
            for item in step['outputs'].values():
                for path in file_items(item):
                    self.generators.setdefault(path, step)

    def generator(self, path, consumer):
        """Find the step that generated a file a step used.

        Interfaces that copy their inputs into their working directory are recorded
        with the path of the copy: look then for a file of the same name among
        the outputs of the nodes the step was connected to.
        """
        if path in self.generators:
            return path, self.generators[path]
        for step in self.steps.values():
            if step['node'] in self.upstream.get(consumer['node'], ()):
                for item in step['outputs'].values():
                    for candidate in file_items(item):
                        if Path(candidate).name == Path(path).name:
                            return candidate, step
        return path, None


def make_plan(records, targets, paths):
    """Order the steps to replay, and list the files they need."""
    steps, needed, aliases, problems = [], {}, {}, []

    def _resolve(path, consumer):
        if paths(path).exists():
            if paths(path).is_file():
                needed[path] = paths(path)
            return
        source, generator = records.generator(path, consumer)
        if generator is consumer:
            return
        if generator is None:
            # Values that only look like paths, or files that were not there either
            if path in records.checksums:
                problems.append(f'{path} is not available, and nothing is known to generate it')
            return
        if source != path:
            aliases[path] = source
        if paths(source).is_file():
            needed[source] = paths(source)
        else:
            _add(generator)

    def _add(step):
        if step in steps:
            return
        for port, item in step['inputs'].items():
            if port != 'base_directory':
                for path in file_items(item):
                    _resolve(path, step)
        steps.append(step)

    for target in targets:
        _add(records.generators[target])
    return steps, needed, aliases, problems


def destination(step, path, work_dir, out_dir):
    """Where a file a step generated is going to be regenerated."""
    base = step['inputs'].get('base_directory')
    if base is not None:
        return out_dir / Path(path).relative_to(recorded_path(base['value']))
    return work_dir / step['name'] / Path(path).name


def set_input(interface, port, value):
    """Set an input, undoing the simplifications of how values were recorded."""
    from traits.api import TraitError

    candidates = [value, [value]]
    if value in (0, 1):
        candidates.insert(0, bool(value))
    if isinstance(value, list):
        candidates.append(tuple(value))
    for candidate in candidates:
        try:
            setattr(interface.inputs, port, candidate)
            return
        except TraitError as exc:
            error = exc
    raise error


def replay(step, paths, work_dir, out_dir):
    """Run again the interface of a step, with the inputs that were recorded."""
    values = {}
    for port, item in step['inputs'].items():
        value = item['value']
        if port == 'base_directory':
            value = str(out_dir)
        elif item['is_file']:
            value = str(paths(value))
        elif isinstance(value, str):
            if value == 'Unknown':  # how undefined values were recorded
                continue
            if value.lstrip().startswith(('[', '{')):
                value = json.loads(value)
            if isinstance(value, list):
                value = [
                    str(paths(v)) if isinstance(v, str) and v.startswith(('/', 'file:///')) else v
                    for v in value
                ]
            elif isinstance(value, str) and value.startswith(('/', 'file:///')):
                value = str(paths(value))
        values[port] = value

    cls = getattr(importlib.import_module(step['module']), step['cls'])
    if step['cls'] == 'Function':
        names = [name for name in values if name != 'function_str']
        interface = cls(input_names=names, output_names=list(step['outputs']))
    else:
        # Interfaces such as data sinks accept additional inputs (metadata)
        known = cls().inputs.copyable_trait_names()
        interface = cls(**{port: values.pop(port) for port in list(values) if port not in known})
    for port, value in values.items():
        set_input(interface, port, value)

    cwd = work_dir / step['name']
    cwd.mkdir(parents=True, exist_ok=True)
    return interface.run(cwd=str(cwd)).outputs


def redo_targets(records, targets, paths, work_dir, out_dir, execute):
    """Regenerate files, replaying the steps that led to them."""
    steps, needed, aliases, problems = make_plan(records, targets, paths)

    print(f'{len(steps)} step(s) to replay:')
    for index, step in enumerate(steps, 1):
        print(f'  {index}. {step["name"]}  [{step["module"]}.{step["cls"]}]')
        if step['command']:
            print(f'       recorded command: {step["command"]}')
    for path, local in sorted(needed.items()):
        expected = records.checksums.get(path)
        if expected is not None and sha512(local) != expected:
            problems.append(f'{local} differs from what was recorded')
        print(f'INPUT {local}')
    outputs = {}
    for target in targets:
        step = records.generators[target]
        for item in step['outputs'].values():
            for path in file_items(item):
                outputs[path] = destination(step, path, work_dir, out_dir)
                print(f'OUTPUT {outputs[path]}')
    for problem in problems:
        print(f'PROBLEM {problem}')
    if problems:
        return False
    if not execute:
        return True

    reproduced = {}
    for step in steps:
        for alias, source in aliases.items():
            paths.produced.setdefault(alias, paths(source))
        if step['cls'] == 'ResampleSeries':
            # With nipost alone
            redo(records.dataset, step['node'], step['interface'], paths, work_dir, True)
            results = {
                port: [work_dir / step['name'] / Path(path).name for path in file_items(item)]
                for port, item in step['outputs'].items()
            }
        else:
            print(f'\n== {step["node"]}')
            outs = replay(step, paths, work_dir, out_dir)
            results = {port: as_list(getattr(outs, port, None)) for port in step['outputs']}
        for port, item in step['outputs'].items():
            for path, result in zip(file_items(item), results[port], strict=False):
                paths.produced[path] = result = Path(result)
                if path in records.checksums:
                    reproduced[path] = sha512(result) == records.checksums[path]
                    state = 'REPRODUCED' if reproduced[path] else 'DIFFERENT checksum'
                    print(f'    {state} {result}')
    return all(reproduced.get(path, False) for path in outputs)


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
    parser.add_argument(
        '--target',
        action='append',
        default=[],
        metavar='PATH',
        help='regenerate the file(s) whose recorded path ends with this, replaying all '
        'steps necessary, instead of redoing the resamplings alone (may be repeated)',
    )
    parser.add_argument('--out-dir', type=Path, default=Path('redo'))
    parser.add_argument(
        '--work-dir', type=Path, help='where to keep intermediate files of --target'
    )
    parser.add_argument('--dry-run', action='store_true', help='only show what would be done')
    args = parser.parse_args(argv)

    dataset = rdflib.Dataset(default_union=True)
    dataset.parse(args.provenance, format='json-ld')
    paths = Paths(args.map)

    if args.target:
        records = Records(dataset)
        targets = sorted(
            path for path in records.generators if any(path.endswith(t) for t in args.target)
        )
        if not targets:
            print(f'No file was recorded with a path ending with {" or ".join(args.target)}')
            return 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            done = redo_targets(
                records,
                targets,
                paths,
                (args.work_dir or Path(tmp_dir)).absolute(),
                args.out_dir.absolute(),
                not args.dry_run,
            )
        return 0 if done else 1

    results = [
        redo(dataset, row.node, row.interface, paths, args.out_dir, not args.dry_run)
        for row in dataset.query(NODES, initNs=NS)
        if args.node in str(row.node)
    ]
    print(f'\n{sum(results)}/{len(results)} resampling(s) OK')
    return 0 if results and all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
