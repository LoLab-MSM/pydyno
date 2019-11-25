import numpy as np
from pydyno.discretize import Discretize
import pydyno.discretize_path as dp
from pysb.examples.tyson_oscillator import model
from pydyno.examples.double_enzymatic.mm_two_paths_model import model as model2
from pysb.simulator import ScipyOdeSimulator
from pysb.testing import *


class TestDynSignBase(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 100, 100)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time).run()
        self.tro = Discretize(self.model, self.sim, 1)

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None


class TestDinSygnSingle(TestDynSignBase):

    def test_run_tropical(self):
        signatures = self.tro.get_signatures()
        assert np.array_equal(signatures.loc['__s2_c'].values[0],
                              np.array([21, 1, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2, 21,
                                        21, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21, 21, 21, 21, 21, 21, 21,
                                        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2, 21,
                                        21, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21, 21, 21, 21, 21, 21, 21,
                                        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 2, 2, 2,
                                        2, 21, 21, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 21]))

    def test_run_tropical_multi_one(self):
        self.tro.get_signatures(cpu_cores=2)

    def test_equations_to_tropicalize(self):
        imp_nodes = self.tro.get_important_nodes(get_passengers_by='imp_nodes')
        assert imp_nodes == [2, 4]

    @raises(ValueError)
    def test_equations_to_tropicalize_invalid_method(self):
        self.tro.get_important_nodes(get_passengers_by='random')


class TestPathSignBase(object):
    def setUp(self):
        self.model = model2
        self.time = np.linspace(0, 100, 100)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time).run()
        self.dom = dp.DomPath(model=self.model, simulations=self.sim)
        self.graph = self.dom.create_bipartite_graph()
        self.rxn_df = self.dom.get_reaction_flux_df(self.dom.trajectories, self.dom.parameters[0])

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None


class TestPathSygnSingle(TestPathSignBase):

    def test_run_tropical(self):
        signatures, paths = self.dom.get_path_signatures(type_analysis='consumption',
                                                         dom_om=1, target='s0', depth=2)
        assert np.array_equal(signatures.sequences.values,
                              np.array([[1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    def test_run_tropical_multi_one(self):
        self.dom.get_path_signatures(type_analysis='consumption',
                                     dom_om=1, target='s0', depth=2, num_processors=2)

    def test_dominant_connected_reactions(self):
        dom_rxns = dp._dominant_connected_reactions(self.graph, 's0', 1.0101010101010102, self.rxn_df, 1, 'out_edges', 1)
        assert dom_rxns == ['r0']

    def test_flip_network_edges(self):
        graph = self.graph.copy()
        dp._flip_network_edges(graph, ['r1'], [])
        diff1 = graph.edges() - self.graph.edges()
        diff2 = self.graph.edges() - graph.edges
        assert diff1 == {('r1', 's2'), ('s4', 'r1'), ('r1', 's0')}
        assert diff2 == {('r1', 's4'), ('s2', 'r1'), ('s0', 'r1')}

    def test_species_connected_to_node(self):
        sps = dp._species_connected_to_node(self.graph, 'r1', 'out_edges', 1)
        assert sps == ['s4']

