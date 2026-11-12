# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Write out the provenance of an *fMRIPrep* execution.

*Nipype* is able to record the provenance of every interface it runs
(inputs, outputs, their checksums, command line, environment, timing...)
following the `W3C PROV <https://www.w3.org/TR/prov-overview/>`__ data model.
The utilities in this module collect those records after a workflow has been run,
split them by participant (and session, if processed separately), and serialize
them as `JSON-LD <https://www.w3.org/TR/json-ld11/>`__.

Identifiers
-----------
The run, and all the workflows and nodes executed in it, are given identifiers
that are informative and reproducible, rather than random:

* ``<run-id>`` (see :func:`get_run_id`), for the execution of *fMRIPrep* as a whole;
* ``<run-id>/<workflow fullname>``, for each (nested) workflow;
* ``<run-id>/<node fullname>``, for each node; and
* ``<run-id>/<node fullname>/record[-<index>]``, for the bundle with the detailed record
  of a node (one per iteration, for a ``MapNode``).

These are local names within the :data:`PROV_NAMESPACE`, which follows the convention
proposed by `BEP028 <https://github.com/bids-standard/bids-specification/pull/2099>`__
for identifiers of provenance records described in the current dataset.
"""

import re
from pathlib import Path
from urllib.parse import quote

PROV_FILENAME = 'provenance.jsonld'

PROV_NAMESPACE = 'bids::prov#'
"""Namespace of the identifiers of the run, its workflows and its nodes."""

JSONLD_CONTEXT = {
    'prov': 'http://www.w3.org/ns/prov#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
    'foaf': 'http://xmlns.com/foaf/0.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'nipype': 'http://nipy.org/nipype/terms/',
    'niiri': 'http://iri.nidash.org/',
    'crypto': 'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/',
    'run': PROV_NAMESPACE,
}
"""Prefixes used to keep the JSON-LD serialization compact."""

_RUN_UUID = re.compile(r'(?P<date>\d{8})-(?P<time>\d{6})_(?P<uid>[0-9a-fA-F]{4})')


def get_run_id(run_uuid, name='fmriprep'):
    """Derive a short, informative identifier of a run from its ``run_uuid``.

    The identifier is composed of the name of the tool, the time the run was
    started and the first four characters of its UUID, which makes collisions unlikely
    while keeping it readable:

    >>> get_run_id('20200306-105302_d365772b-fd60-4741-a722-372c2f558b50')
    'fmriprep-20200306T105302-d365'

    Custom identifiers are kept, removing characters that are unsafe in an IRI:

    >>> get_run_id('my run #1')
    'fmriprep-my-run-1'

    """
    if match := _RUN_UUID.match(str(run_uuid)):
        return f'{name}-{match["date"]}T{match["time"]}-{match["uid"].lower()}'
    return f'{name}-{re.sub(r"[^0-9A-Za-z_.]+", "-", str(run_uuid)).strip("-")}'


def get_run_start(run_uuid):
    """Extract the (local) time a run was started from its ``run_uuid``, if possible.

    >>> get_run_start('20200306-105302_d365772b-fd60-4741-a722-372c2f558b50').isoformat()[:19]
    '2020-03-06T10:53:02'
    >>> get_run_start('my run #1') is None
    True

    """
    from datetime import datetime

    if match := _RUN_UUID.match(str(run_uuid)):
        try:
            started = datetime.strptime(f'{match["date"]}{match["time"]}', '%Y%m%d%H%M%S')  # noqa: DTZ007
        except ValueError:
            return None
        return started.astimezone()  # The run_uuid is stamped with the local time
    return None


def group_nodes(execgraph, default_dir):
    """Split the nodes of an execution graph by the log directory they report to.

    *fMRIPrep* sets the ``crashdump_dir`` of all the nodes of each
    single-subject workflow to ``<output_dir>/sub-<label>[/ses-<label>]/log/<run_uuid>``.
    Nodes that do not belong to any of those workflows (for example, the node
    setting up the *FreeSurfer* subjects directory) report to ``default_dir``.
    Such nodes are shared prerequisites, and are therefore included in all groups.

    Parameters
    ----------
    execgraph : :obj:`networkx.DiGraph`
        The execution graph, as returned by :meth:`nipype.pipeline.engine.Workflow.run`.
    default_dir : :obj:`os.PathLike`
        The log directory of nodes that are not specific to one participant.

    Returns
    -------
    groups : :obj:`dict`
        A mapping of log directories to the list of nodes that report to them.
        ``default_dir`` is only a key if no node reports to a different directory.

    """
    default_dir = Path(default_dir)
    shared = []
    groups = {}
    for node in execgraph.nodes():
        log_dir = Path((node.config or {}).get('execution', {}).get('crashdump_dir', default_dir))
        if log_dir == default_dir:
            shared.append(node)
        else:
            groups.setdefault(log_dir, []).append(node)

    if not groups:
        return {default_dir: shared}
    return {log_dir: shared + nodes for log_dir, nodes in groups.items()}


def _add_records(doc, node, node_id):
    """Add the detailed record(s) of a node as bundle(s), and return their identifiers."""
    import prov.model as pm
    from nipype.interfaces.base import isdefined
    from nipype.interfaces.base.support import InterfaceResult
    from nipype.utils.provenance import ProvStore

    result = node.result
    if not isinstance(result.runtime, list):
        # Reuse the record stored when the interface was run, if available
        records = {'record': result.provenance or ProvStore().add_results(result)}
    else:  # MapNode: one record per iteration
        records = {}
        for idx, runtime in enumerate(result.runtime):
            subresult = InterfaceResult(result.interface[idx], runtime, outputs={})
            if result.inputs and idx < len(result.inputs):
                subresult.inputs = result.inputs[idx]
            if result.outputs:
                for key, values in result.outputs.items():
                    if isdefined(values) and idx < len(values):
                        subresult.outputs[key] = values[idx]
            records[f'record-{idx}'] = ProvStore().add_results(subresult)

    bundle_ids = []
    for name, record in records.items():
        bundle = pm.ProvBundle(
            record.get_records(), identifier=doc.valid_qualified_name(f'run:{node_id}/{name}')
        )
        doc.add_bundle(bundle)
        bundle_ids.append(bundle.identifier)
    return bundle_ids


def build_provenance(execgraph, run_uuid, *, name='fmriprep', version=None, command=None):
    """Describe the execution of a workflow as a W3C PROV document.

    The document has one ``prov:Activity`` for the run, one for each (nested)
    workflow and one for each node, where:

    * workflows and nodes are related to the workflow (or run) they belong to
      with ``dcterms:isPartOf``;
    * nodes are related to the nodes they take inputs from with ``prov:wasInformedBy``;
    * the detailed record of each node (the interface that was run with its inputs,
      outputs, environment, command line...) is kept in a ``prov:Bundle`` (that is,
      a named graph) which ``prov:wasGeneratedBy`` the node.

    Parameters
    ----------
    execgraph : :obj:`networkx.DiGraph`
        The execution graph (or a subgraph of it), as returned by
        :meth:`nipype.pipeline.engine.Workflow.run`.
    run_uuid : :obj:`str`
        The unique identifier of the run (``config.execution.run_uuid``).
    name : :obj:`str`
        Name of the tool that was run.
    version : :obj:`str`
        Version of the tool that was run.
    command : :obj:`str`
        Command line that started the run.

    Returns
    -------
    doc : :obj:`prov.model.ProvDocument`
        The provenance document.

    """
    from datetime import datetime

    import prov.model as pm
    from nipype.utils.provenance import ProvStore, dcterms, nipype_ns

    doc = ProvStore().g  # Comes with the namespaces Nipype's records use
    doc.add_namespace('run', PROV_NAMESPACE)

    def _qname(local):
        return doc.valid_qualified_name(f'run:{local}')

    # The run itself
    run_id = get_run_id(run_uuid, name=name)
    run_attrs = {pm.PROV['label']: run_id, dcterms['identifier']: str(run_uuid)}
    if command:
        run_attrs[nipype_ns['command']] = command
    run = doc.activity(
        _qname(run_id), get_run_start(run_uuid), datetime.now().astimezone(), run_attrs
    )
    agent_attrs = {pm.PROV['type']: pm.PROV['SoftwareAgent'], pm.PROV['label']: name}
    if version:
        agent_attrs[nipype_ns['version']] = str(version)
    agent_id = '-'.join(quote(str(part), safe='') for part in (name, version) if part)
    doc.wasAssociatedWith(run, doc.agent(_qname(agent_id), agent_attrs))

    # Nested workflows
    workflows = {}

    def _workflow(hierarchy):
        if not hierarchy:
            return run.identifier
        if hierarchy not in workflows:
            attrs = {
                pm.PROV['type']: nipype_ns['Workflow'],
                pm.PROV['label']: hierarchy[-1],
                dcterms['isPartOf']: _workflow(hierarchy[:-1]),
            }
            wf_id = _qname(f'{run_id}/{quote(".".join(hierarchy), safe="")}')
            workflows[hierarchy] = doc.activity(wf_id, other_attributes=attrs).identifier
        return workflows[hierarchy]

    # Nodes
    activities = {}
    for node in execgraph.nodes():
        # The itername tells apart the expansions of iterables, unlike the fullname
        node_id = f'{run_id}/{quote(node.itername, safe="")}'
        *hierarchy, _ = node.fullname.split('.')
        is_mapnode = isinstance(node.result.runtime, list)
        attrs = [
            (pm.PROV['type'], nipype_ns[node.interface.__class__.__name__]),
            (pm.PROV['type'], nipype_ns['MapNode' if is_mapnode else 'Node']),
            (pm.PROV['label'], node.name),
            (nipype_ns['hashval'], node.hash_exists()[1]),
            (dcterms['isPartOf'], _workflow(tuple(hierarchy))),
        ]
        activities[node] = doc.activity(_qname(node_id), other_attributes=attrs)
        for bundle_id in _add_records(doc, node, node_id):
            bundle = doc.entity(bundle_id, {pm.PROV['type']: pm.PROV_BUNDLE})
            doc.wasGeneratedBy(bundle, activities[node])

    # Dependencies
    for source, target in execgraph.edges():
        doc.wasInformedBy(activities[target], activities[source])

    return doc


def write_provenance(
    execgraph,
    default_dir,
    run_uuid,
    *,
    filename=PROV_FILENAME,
    name='fmriprep',
    version=None,
    command=None,
):
    """Write the provenance of an executed workflow, one JSON-LD file per log directory.

    Each file is the PROV-O serialization of the document generated by
    :func:`build_provenance` for the nodes that report to that log directory
    (see :func:`group_nodes`).
    Because identifiers are derived from the run and the names of workflows and nodes,
    the files written for the different participants of a run may be merged consistently.

    Parameters
    ----------
    execgraph : :obj:`networkx.DiGraph`
        The execution graph, as returned by :meth:`nipype.pipeline.engine.Workflow.run`.
    default_dir : :obj:`os.PathLike`
        The log directory of nodes that are not specific to one participant.
    run_uuid : :obj:`str`
        The unique identifier of the run (``config.execution.run_uuid``).
    filename : :obj:`str`
        Name of the file to be written within each log directory.
    name : :obj:`str`
        Name of the tool that was run.
    version : :obj:`str`
        Version of the tool that was run.
    command : :obj:`str`
        Command line that started the run.

    Returns
    -------
    out_files : :obj:`list` of :obj:`pathlib.Path`
        The files that were written.

    """
    out_files = []
    for log_dir, nodes in group_nodes(execgraph, default_dir).items():
        doc = build_provenance(
            execgraph.subgraph(nodes), run_uuid, name=name, version=version, command=command
        )

        log_dir.mkdir(exist_ok=True, parents=True)
        out_file = log_dir / filename
        with out_file.open('w') as fobj:
            doc.serialize(
                fobj,
                format='rdf',
                rdf_format='json-ld',
                context=JSONLD_CONTEXT,
                auto_compact=True,
                indent=1,
            )
        out_files.append(out_file)
    return out_files
