import numpy as np
from pydyno.discretize import Discretize
import pydyno.discretization.pysb_discretize as dp
import pydyno.discretization.sbml_discretize as ds
import pydyno.discretization.base as base
from pysb.examples.tyson_oscillator import model
from pydyno.examples.double_enzymatic.mm_two_paths_model import model as model2
from pysb.simulator import ScipyOdeSimulator
import pytest
import tellurium as te
from pydyno.util_tellurium import SbmlModel, SbmlSimulation


@pytest.fixture(scope="class")
def discretize():
    time = np.linspace(0, 100, 100)
    sim = ScipyOdeSimulator(model, tspan=time).run()
    tro = Discretize(model, sim, 1)
    return tro


class TestDiscretize:
    def test_run_tropical(self, discretize):
        signatures = discretize.get_signatures()
        assert np.array_equal(signatures.loc['__s2_c'].values[0],
                              np.array([21, 1, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2, 21,
                                        21, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21, 21, 21, 21, 21, 21, 21,
                                        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2, 21,
                                        21, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21, 21, 21, 21, 21, 21, 21,
                                        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2,
                                        2, 21, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21]))

    def test_run_tropical_multi_one(self, discretize):
        discretize.get_signatures(cpu_cores=2)

    def test_equations_to_tropicalize(self, discretize):
        imp_nodes = discretize.get_important_nodes(get_passengers_by='imp_nodes')
        assert imp_nodes == [2, 4]

    def test_equations_to_tropicalize_invalid_method(self, discretize):
        with pytest.raises(ValueError):
            discretize.get_important_nodes(get_passengers_by='random')


@pytest.fixture(scope="class")
def pysb_dom_path():
    time = np.linspace(0, 100, 100)
    sim = ScipyOdeSimulator(model2, tspan=time).run()
    dom = dp.PysbDomPath(model=model2, simulations=sim)
    return dom


class TestPathPysbSingle:

    def test_run_tropical(self, pysb_dom_path):
        signatures, paths = pysb_dom_path.get_path_signatures(type_analysis='consumption',
                                                         dom_om=1, target='s0', depth=2)
        assert np.array_equal(signatures.sequences.values,
                              np.array([[1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_run_tropical_multi_one(self, pysb_dom_path):
        pysb_dom_path.get_path_signatures(type_analysis='consumption',
                                     dom_om=1, target='s0', depth=2, num_processors=2)

    def test_dominant_connected_reactions(self, pysb_dom_path):
        graph = pysb_dom_path.create_bipartite_graph()
        rxn_df = pysb_dom_path.get_reaction_flux_df(0)
        dom_rxns = base._dominant_connected_reactions(graph, 's0', 1.0101010101010102,
                                                      rxn_df, 1, 'out_edges', 1)
        assert dom_rxns == ['r0']

    def test_flip_network_edges(self, pysb_dom_path):
        graph = pysb_dom_path.create_bipartite_graph()
        graph2 = graph.copy()
        base._flip_network_edges(graph, ['r1'], [])
        diff1 = graph.edges() - graph2.edges()
        diff2 = graph2.edges() - graph.edges
        assert diff1 == {('r1', 's2'), ('s4', 'r1'), ('r1', 's0')}
        assert diff2 == {('r1', 's4'), ('s2', 'r1'), ('s0', 'r1')}

    def test_species_connected_to_node(self, pysb_dom_path):
        graph = pysb_dom_path.create_bipartite_graph()
        sps = base._species_connected_to_node(graph, 'r1', 'out_edges', 1)
        assert sps == ['s4']


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
    sim.tspan = r.getSimulationData()['time']
    dom = ds.SbmlDomPath(model, sim)
    return dom


class TestPathSbmlSingle:

    def test_run_tropical(self, sbml_dom_path):
        signatures, paths = sbml_dom_path.get_path_signatures(type_analysis='consumption',
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
        rxn_df = sbml_dom_path.get_reaction_flux_df(0)
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