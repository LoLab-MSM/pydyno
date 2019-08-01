import numpy as np
from nose.tools import *
from pydyno.discretize import Discretize
from pydyno.discretize_path import DomPath
from pysb.examples.tyson_oscillator import model
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
            self.model = model
            self.time = np.linspace(0, 100, 100)
            self.sim = ScipyOdeSimulator(self.model, tspan=self.time).run()
            self.dom = DomPath(model=self.model, simulations=self.sim, type_analysis='consumption',
                               dom_om=1, target='s2', depth=2)

        def tearDown(self):
            self.model = None
            self.time = None
            self.sim = None

    class TestPathSygnSingle(TestPathSignBase):

        def test_run_tropical(self):
            signatures, paths = self.dom.get_path_signatures()
            assert np.array_equal(signatures.sequences.values,
                                  np.array([[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                                         [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]))

        def test_run_tropical_multi_one(self):
            self.dom.get_path_signatures(num_processors=2)


