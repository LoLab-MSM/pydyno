import numpy as np
from nose.tools import *
from pysb.examples.tyson_oscillator import model
from pysb.simulator import ScipyOdeSimulator
from pysb.simulator.base import SimulatorException
from tropical.max_plus_multiprocessing_numpy import Tropical, run_tropical


class TestFluxVisualization(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 200, 200)
        self.sim = ScipyOdeSimulator(self.model, tspan=self.time).run()
        self.y = self.sim.all

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None

    def test_tropicalization(self):
        tropical = Tropical(self.model)
        tropical.tropicalize(tspan=self.time)

    def test_verbose(self):
        tropical_verbose = Tropical(self.model)
        tropical_verbose.tropicalize(tspan=self.time, verbose=True)

    def test_type_sign_production(self):
        tropical_production = Tropical(self.model)
        tropical_production.tropicalize(tspan=self.time, type_sign='production')

    def test_type_sign_coonsumption(self):
        tropical_consumption = Tropical(self.model)
        tropical_consumption.tropicalize(tspan=self.time, type_sign='consumption')

    @raises(ValueError)
    def test_type_sign_wrong(self):
        tropical_wrong_type = Tropical(self.model)
        tropical_wrong_type.tropicalize(tspan=self.time, type_sign='bla')

    def test_find_passengers_imp(self):
        tropical_find_passengers_imp = Tropical(self.model)
        tropical_find_passengers_imp.tropicalize(tspan=self.time, find_passengers_by='imp_nodes')

    def test_find_passengers_qssa(self):
        tropical_find_passengers_qssa = Tropical(self.model)
        tropical_find_passengers_qssa.tropicalize(tspan=self.time, find_passengers_by='qssa')

    @raises(Exception)
    def test_find_passenger_wrong_type(self):
        tropical_wrong_type = Tropical(self.model)
        tropical_wrong_type.tropicalize(tspan=self.time, find_passengers_by='bla')

    # def test_visualization(self):
    #     tropical_visualization = Tropical(self.model)
    #     tropical_visualization.tropicalize(tspan=self.time)
    #     tropical_visualization.visualization(driver_species=[3, 5])

    def test_get_passenger(self):
        tropical_passenger = Tropical(self.model)
        tropical_passenger.tropicalize(tspan=self.time)
        tropical_passenger.get_passenger()

    def test_run_tropical(self):
        run_tropical(self.model, self.time)

    def test_plot_imposed_trace(self):
        tropical_trace = Tropical(self.model)
        tropical_trace.tropicalize(tspan=self.time, plot_imposed_trace=True)

    @raises(SimulatorException)
    def test_time_not_defined(self):
        tropical_no_time = Tropical(self.model)
        tropical_no_time.tropicalize()

    @raises(Exception)
    def test_parameters_different_length(self):
        tropical_pars = Tropical(self.model)
        tropical_pars.tropicalize(tspan=self.time, param_values=[1, 2])





