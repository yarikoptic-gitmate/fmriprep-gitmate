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

from fmriprep.utils.provenance import group_nodes, write_provenance

PROV = 'http://www.w3.org/ns/prov#'
NIPYPE = 'http://nipy.org/nipype/terms/'
RDFS_LABEL = 'http://www.w3.org/2000/01/rdf-schema#label'
CRYPTO = 'http://id.loc.gov/vocabulary/preservation/cryptographicHashFunctions/'


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


def test_write_provenance(tmp_path, execgraph):
    rdflib = pytest.importorskip('rdflib')

    out_files = write_provenance(execgraph, tmp_path / 'out' / 'logs')
    assert sorted(out_files) == [
        tmp_path / 'out' / f'sub-{s}' / 'log' / 'provenance.jsonld' for s in ('01', '02')
    ]

    prov_type = rdflib.RDF.type
    activity = rdflib.URIRef(f'{PROV}Activity')
    for out_file in out_files:
        subject = out_file.parent.parent.name.removeprefix('sub-')

        graph = rdflib.Dataset()
        graph.parse(out_file, format='json-ld')

        # One activity per node in the default graph, with dependencies
        default = graph.default_graph
        labels = {
            str(default.value(node, rdflib.URIRef(RDFS_LABEL))): node
            for node in default.subjects(prov_type, activity)
        }
        assert sorted(labels) == ['Function_first', 'Function_second', 'Function_shared']
        starters = {
            (label, starter)
            for label, node in labels.items()
            for start in default.objects(node, rdflib.URIRef(f'{PROV}qualifiedStart'))
            for starter in default.objects(start, rdflib.URIRef(f'{PROV}hadActivity'))
        }
        assert starters == {
            ('Function_first', labels['Function_shared']),
            ('Function_second', labels['Function_first']),
        }

        # One bundle per node execution: shared + first + two iterations of second
        bundles = set(default.subjects(prov_type, rdflib.URIRef(f'{PROV}Bundle')))
        assert len(bundles) == 4
        assert bundles == {g.identifier for g in graph.graphs()} - {default.identifier}

        # The details of what was run are available, and do not leak across participants
        workdirs = _values(graph, f'{NIPYPE}workingDirectory')
        assert len(workdirs) == 4
        assert sum(f'sub_{subject}_wf' in workdir for workdir in workdirs) == 3
        assert not any('sub_' in workdir.replace(f'sub_{subject}_wf', '') for workdir in workdirs)
        outputs = _values(graph, f'{PROV}atLocation')
        assert len(outputs) == 4
        assert all(output.endswith('sum.txt') for output in outputs)
        # Checksums of 1, 3, 3 + 3, and 3 + 4
        assert len(_values(graph, f'{CRYPTO}sha512')) == 4
