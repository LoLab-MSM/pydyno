import numpy as np
from nose.tools import *
from tropical import discretize
from pysb.examples.tyson_oscillator import model
from pysb.simulator import ScipyOdeSimulator
from pysb.simulator.base import SimulatorException
from pysb.testing import *


class TestDynSignBase(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 100, 100)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time).run()
        self.tro = discretize.Discretize(self.model, self.sim, 1)

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

    # @raises(AssertionError)
    # def test_set_combinations_no_eqs_for_trop(self):
    #     self.tro.set_combinations_sm()

