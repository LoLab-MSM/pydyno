import numpy as np
from earm2_flat import model
from tropical.dynamic_signatures_range import run_tropical
from pysb.simulator.scipyode import ScipyOdeSimulator

parameters = np.load('calibrated_6572pars.npy')

t = np.linspace(0, 20000,  100)
sim = ScipyOdeSimulator(model, tspan=t, param_values=parameters[0]).run()
run_tropical(model, simulations=sim, diff_par=1)
