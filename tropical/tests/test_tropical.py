import numpy as np
from nose.tools import *
from tropical import dynamic_signatures
from pysb.examples.tyson_oscillator import model
from pysb.simulator import ScipyOdeSimulator
from pysb.simulator.base import SimulatorException
from pysb.testing import *

class TestDynSignBase(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 100, 100)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time).run()
        self.tro = dynamic_signatures.Tropical(self.model)

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None

class TestDinSygnSingle(TestDynSignBase):
    def test_setup_tropical(self):
        self.tro.setup_tropical(tspan=self.time)
        assert self.tro._is_setup == True

    def test_run_tropical(self):
        signatures = dynamic_signatures.run_tropical(self.model, self.sim)
        assert np.array_equal(signatures[3][1],
                              np.array(['NoDoms', 'M11', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'M10',
                                        'M10', 'M10', 'NoDoms', 'NoDoms', 'NoDoms', 'M11', 'M11', 'M11',
                                        'M11', 'M11', 'M11', 'M11', 'M11', 'M11', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'M10',
                                        'M10', 'M10', 'NoDoms', 'NoDoms', 'NoDoms', 'M11', 'M11', 'M11',
                                        'M11', 'M11', 'M11', 'M11', 'M11', 'M11', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms',
                                        'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'NoDoms', 'M10',
                                        'M10', 'M10', 'M10', 'NoDoms', 'NoDoms', 'M11', 'M11', 'M11', 'M11',
                                        'M11', 'M11', 'M11', 'M11', 'M11', 'M11', 'NoDoms'],
                                        dtype='|S6'))

    @raises(Exception)
    def test_signature_no_setup(self):
        self.tro.signature(y=self.sim.all, param_values=self.sim.param_values)

    def test_run_tropical_multi_one(self):
        dynamic_signatures.run_tropical_multi(self.model, self.sim)

    def test_equations_to_tropicalize(self):
        self.tro.equations_to_tropicalize(get_passengers_by='imp_nodes')
        assert self.tro.eqs_for_tropicalization.keys() == [3, 5]

    @raises(ValueError)
    def test_equations_to_tropicalize_invalid_method(self):
        self.tro.equations_to_tropicalize(get_passengers_by='random')

    @raises(AssertionError)
    def test_set_combinations_no_eqs_for_trop(self):
        self.tro.set_combinations_sm()

