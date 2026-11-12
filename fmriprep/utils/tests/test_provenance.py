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
"""Tests of provenance records."""

import pytest
from nipype import config as ncfg
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from fmriprep.utils.provenance import (
    PROV_NAMESPACE,
    get_run_id,
    group_nodes,
    write_provenance,
)

PROV = 'http://www.w3.org/ns/prov#'
NIPYPE = 'http://nipy.org/nipype/terms/'
RDFS_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'
CRYPTO = 'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/'
DCTERMS = 'http://purl.org/dc/terms/'
RUN_UUID = '20200306-105302_d365772b-fd60-4741-a722-372c2f558b50'
RUN_ID = 'fmriprep-20200306T105302-d365'


def _add(a, b):
    from pathlib import Path

    out_file = Path('sum.txt').absolute()
    out_file.write_text(f'{a + b}')
    return a + b, str(out_file)


def _values(dataset, predicate):
    """Collect the objects of a predicate across all the graphs of a dataset."""
    from rdflib import URIRef

    return {str(quad[2]) for quad in dataset.quads((None, URIRef(predicate), None))}


def _init_wf(base_dir, log_dir, subjects):
    wf = pe.Workflow(name='fmriprep_wf', base_dir=str(base_dir))
    shared = pe.Node(niu.Function(function=_add, output_names=['out', 'out_file']), name='shared')
    shared.inputs.a = 0
    shared.inputs.b = 1
    for subject in subjects:
        sub_wf = pe.Workflow(name=f'sub_{subject}_wf')
        first = pe.Node(
            niu.Function(function=_add, output_names=['out', 'out_file']), name='first'
        )
        first.inputs.b = 2
        second = pe.MapNode(
            niu.Function(function=_add, output_names=['out', 'out_file']),
            name='second',
            iterfield=['a'],
        )
        second.inputs.a = [3, 4]
        sub_wf.connect(first, 'out', second, 'b')
        # Mimic fmriprep.workflows.base.init_fmriprep_wf
        sub_wf.config['execution']['crashdump_dir'] = str(log_dir / f'sub-{subject}' / 'log')
        for node in sub_wf._get_all_nodes():
            node.config = {'execution': dict(sub_wf.config['execution'])}
        wf.connect(shared, 'out', sub_wf, 'first.a')
    return wf


@pytest.fixture
def execgraph(tmp_path, monkeypatch):
    # Nipype drops provenance.{provn,ttl} files into the current directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(ncfg._sections['execution'], 'write_provenance', 'true')
    monkeypatch.setitem(
        ncfg._sections['execution'], 'crashdump_dir', str(tmp_path / 'out' / 'logs')
    )
    return _init_wf(tmp_path / 'work', tmp_path / 'out', ['01', '02']).run(plugin='Linear')


def test_group_nodes(tmp_path, execgraph):
    log_dir = tmp_path / 'out' / 'logs'
    groups = group_nodes(execgraph, log_dir)
    assert sorted(groups) == [tmp_path / 'out' / f'sub-{s}' / 'log' for s in ('01', '02')]
    for path, nodes in groups.items():
        subject = path.parent.name.removeprefix('sub-')
        assert sorted(node.fullname for node in nodes) == [
            'fmriprep_wf.shared',
            f'fmriprep_wf.sub_{subject}_wf.first',
            f'fmriprep_wf.sub_{subject}_wf.second',
        ]

    # Without participant-specific nodes, everything goes to the default directory
    assert list(group_nodes(execgraph.subgraph(groups[path][:1]), log_dir)) == [log_dir]


@pytest.mark.parametrize(
    ('run_uuid', 'expected'),
    [
        (RUN_UUID, RUN_ID),
        ('20200306-105302_D365772B', RUN_ID),
        ('custom', 'fmriprep-custom'),
        ('my run #1', 'fmriprep-my-run-1'),
    ],
)
def test_get_run_id(run_uuid, expected):
    assert get_run_id(run_uuid) == expected
    assert get_run_id(run_uuid, name='tool') == expected.replace('fmriprep', 'tool', 1)


def test_write_provenance(tmp_path, execgraph):
    rdflib = pytest.importorskip('rdflib')

    out_files = write_provenance(
        execgraph,
        tmp_path / 'out' / 'logs',
        RUN_UUID,
        version='26.0.0',
        command='fmriprep bids out participant',
    )
    assert sorted(out_files) == [
        tmp_path / 'out' / f'sub-{s}' / 'log' / 'provenance.jsonld' for s in ('01', '02')
    ]

    def _id(local=''):
        return rdflib.URIRef(f'{PROV_NAMESPACE}{RUN_ID}{local}')

    rdf_type = rdflib.RDF.type
    label = rdflib.URIRef(RDFS_LABEL)
    part_of = rdflib.URIRef(f'{DCTERMS}isPartOf')
    informed_by = rdflib.URIRef(f'{PROV}wasInformedBy')
    activity = rdflib.URIRef(f'{PROV}Activity')
    shared_graphs = []
    for out_file in out_files:
        subject = out_file.parent.parent.name.removeprefix('sub-')
        other = {'01': '02', '02': '01'}[subject]

        graph = rdflib.Dataset()
        graph.parse(out_file, format='json-ld')
        default = graph.default_graph

        # The run, the workflows and the nodes are activities with informative identifiers
        sub_wf = f'/fmriprep_wf.sub_{subject}_wf'
        assert set(default.subjects(rdf_type, activity)) == {
            _id(),
            _id('/fmriprep_wf'),
            _id('/fmriprep_wf.shared'),
            _id(sub_wf),
            _id(f'{sub_wf}.first'),
            _id(f'{sub_wf}.second'),
        }

        # The run
        assert str(default.value(_id(), label)) == RUN_ID
        assert str(default.value(_id(), rdflib.URIRef(f'{DCTERMS}identifier'))) == RUN_UUID
        assert str(default.value(_id(), rdflib.URIRef(f'{NIPYPE}command'))).startswith('fmriprep ')
        assert str(default.value(_id(), rdflib.URIRef(f'{PROV}startedAtTime'))).startswith(
            '2020-03-06T10:53:02'
        )
        agent = default.value(_id(), rdflib.URIRef(f'{PROV}wasAssociatedWith'))
        assert agent == rdflib.URIRef(f'{PROV_NAMESPACE}fmriprep-26.0.0')
        assert str(default.value(agent, rdflib.URIRef(f'{NIPYPE}version'))) == '26.0.0'

        # The hierarchy of workflows
        workflow = rdflib.URIRef(f'{NIPYPE}Workflow')
        assert set(default.subjects(rdf_type, workflow)) == {_id('/fmriprep_wf'), _id(sub_wf)}
        assert set(default.subject_objects(part_of)) == {
            (_id('/fmriprep_wf'), _id()),
            (_id('/fmriprep_wf.shared'), _id('/fmriprep_wf')),
            (_id(sub_wf), _id('/fmriprep_wf')),
            (_id(f'{sub_wf}.first'), _id(sub_wf)),
            (_id(f'{sub_wf}.second'), _id(sub_wf)),
        }
        assert str(default.value(_id(f'{sub_wf}.second'), label)) == 'second'
        assert (_id(f'{sub_wf}.second'), rdf_type, rdflib.URIRef(f'{NIPYPE}MapNode')) in default
        assert (_id(f'{sub_wf}.first'), rdf_type, rdflib.URIRef(f'{NIPYPE}Function')) in default

        # Dependencies between nodes
        assert set(default.subject_objects(informed_by)) == {
            (_id(f'{sub_wf}.first'), _id('/fmriprep_wf.shared')),
            (_id(f'{sub_wf}.second'), _id(f'{sub_wf}.first')),
        }

        # One bundle per node execution: shared + first + two iterations of second
        bundles = {
            _id('/fmriprep_wf.shared/record'): _id('/fmriprep_wf.shared'),
            _id(f'{sub_wf}.first/record'): _id(f'{sub_wf}.first'),
            _id(f'{sub_wf}.second/record-0'): _id(f'{sub_wf}.second'),
            _id(f'{sub_wf}.second/record-1'): _id(f'{sub_wf}.second'),
        }
        assert set(default.subjects(rdf_type, rdflib.URIRef(f'{PROV}Bundle'))) == set(bundles)
        assert set(bundles) == {g.identifier for g in graph.graphs()} - {default.identifier}
        for bundle, node in bundles.items():
            assert (bundle, rdflib.URIRef(f'{PROV}wasGeneratedBy'), node) in default

        # The details of what was run are available, and do not leak across participants
        workdirs = _values(graph, f'{NIPYPE}workingDirectory')
        assert len(workdirs) == 4
        assert sum(f'sub_{subject}_wf' in workdir for workdir in workdirs) == 3
        assert not any(f'sub_{other}_wf' in workdir for workdir in workdirs)
        outputs = _values(graph, f'{PROV}atLocation')
        assert len(outputs) == 4
        assert all(output.endswith('sum.txt') for output in outputs)
        # Checksums of 1, 3, 3 + 3, and 3 + 4
        assert len(_values(graph, f'{CRYPTO}sha512')) == 4

        shared_graphs.append(
            {(s, p, o) for s, p, o in default if 'sub_' not in f'{s}{o}' and 'Time' not in str(p)}
        )

    # What is not specific to a participant is described identically in all files
    assert shared_graphs[0] == shared_graphs[1]
    assert (_id('/fmriprep_wf.shared'), rdf_type, activity) in shared_graphs[0]
