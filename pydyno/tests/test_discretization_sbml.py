import pydyno.discretization.sbml_discretize as ds
import tellurium as te
from pydyno.util_tellurium import SbmlModel, SbmlSimulation
import pytest
import  numpy as np
import pydyno.discretization.base as base


@pytest.fixture(scope="class")
def sbml_dom_path():
    import os
    test_path = os.path.dirname(__file__)
    model_path = os.path.join(os.path.dirname(test_path), 'examples', 'sbml_example', 'double_enzymatic_sbml.xml')
    r = te.loadSBMLModel(model_path)
    model = SbmlModel(r)
    sim = SbmlSimulation()
    r.selections = ['time', '__s0', '__s1', '__s2', '__s3', '__s4', '__s5', 'r0', 'r1', 'r2', 'r3']
    r.simulate(0, 100, 100)
    sim.add_simulation(r)
    dom = ds.SbmlDomPath(model, sim)
    return dom


class TestPathSbmlSingle:

    def test_run_tropical(self, sbml_dom_path):
        signatures, _ = sbml_dom_path.get_path_signatures(type_analysis='consumption',
                                                         dom_om=1, target='s0', depth=2)
        assert np.array_equal(signatures.sequences.values,
                              np.array([[1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_run_tropical_multi_one(self, sbml_dom_path):
        sbml_dom_path.get_path_signatures(type_analysis='consumption',
                                     dom_om=1, target='s0', depth=2, num_processors=2)

    def test_dominant_connected_reactions(self, sbml_dom_path):
        graph = sbml_dom_path.create_bipartite_graph()
        rxn_df = sbml_dom_path.all_reaction_flux[0]
        dom_rxns = base._dominant_connected_reactions(graph, 's0', 1.0101010101010102,
                                                      rxn_df, 1, 'out_edges', 1)
        assert dom_rxns == ['r0']

    def test_flip_network_edges(self, sbml_dom_path):
        graph = sbml_dom_path.create_bipartite_graph()
        graph2 = graph.copy()
        base._flip_network_edges(graph, ['r1'], [])
        diff1 = graph.edges() - graph2.edges()
        diff2 = graph2.edges() - graph.edges
        assert diff1 == {('r1', 's2'), ('s4', 'r1'), ('r1', 's0')}
        assert diff2 == {('r1', 's4'), ('s2', 'r1'), ('s0', 'r1')}

    def test_species_connected_to_node(self, sbml_dom_path):
        graph = sbml_dom_path.create_bipartite_graph()
        sps = base._species_connected_to_node(graph, 'r1', 'out_edges', 1)
        assert sps == ['s4']
