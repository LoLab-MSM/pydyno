from tropicalize import Tropical
from pysb.testing import *
from pysb.examples.tyson_oscillator import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from pysb.simulator.base import SimulatorException
from tropicalize import run_tropical


class TestFluxVisualization(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 200, 200)
        self.sim = ScipyOdeSimulator.execute(self.model, tspan=self.time)

    def tearDown(self):
        self.model = None
        self.time = None
        self.sim = None

    def test_tropicalization(self):
        tropical = Tropical(self.model)
        tropical.tropicalize(tspan=self.time)

    def test_verbpse(self):
        tropical_verbose = Tropical(self.model)
        tropical_verbose.tropicalize(tspan=self.time, verbose=True)

    def test_visualization(self):
        tropical_visualization = Tropical(self.model)
        tropical_visualization.tropicalize(tspan=self.time)
        tropical_visualization.visualization(driver_species=[3, 5])

    def test_get_trop_data(self):
        tropical_data = Tropical(self.model)
        tropical_data.tropicalize(tspan=self.time)
        tropical_data.get_trop_data()

    def test_get_passenger(self):
        tropical_passenger = Tropical(self.model)
        tropical_passenger.tropicalize(tspan=self.time)
        tropical_passenger.get_passenger()

    def test_get_tropical_eq(self):
        tropical_eq = Tropical(self.model)
        tropical_eq.tropicalize(tspan=self.time)
        tropical_eq.get_tropical_eqs()

    def test_get_driver_signatures(self):
        tropical_signatures = Tropical(self.model)
        tropical_signatures.tropicalize(tspan=self.time)
        tropical_signatures.get_driver_signatures()

    def test_run_tropical(self):
        run_tropical(self.model, self.time)

    def test_plot_imposed_trace(self):
        tropical_trace = Tropical(self.model)
        tropical_trace.tropicalize(tspan=self.time, plot_imposed_trace=True)

    def test_list_parameters(self):
        tropical_paramaters = Tropical(self.model)
        tropical_paramaters.tropicalize(tspan=self.time, param_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    @raises(SimulatorException)
    def test_time_not_defined(self):
        tropical_no_time = Tropical(self.model)
        tropical_no_time.tropicalize()

    @raises(Exception)
    def test_parameters_different_length(self):
        tropical_pars = Tropical(self.model)
        tropical_pars.tropicalize(tspan=self.time, param_values=[1, 2])






