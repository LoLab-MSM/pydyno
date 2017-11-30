import os
import tropical.helper_functions as hf
import numpy as np
from earm.lopez_embedded import model
from tropical.dynamic_signatures_range import run_tropical
from pysb.simulator.scipyode import ScipyOdeSimulator

directory = os.path.dirname(__file__)
parameters_path = os.path.join(directory, "parameters_5000")
all_parameters = hf.listdir_fullpath(parameters_path)
parameters = hf.read_pars(all_parameters[237])

t = np.linspace(0, 20000,  100)
sim = ScipyOdeSimulator(model, tspan=t, param_values=parameters).run()
run_tropical(model, simulations=sim, diff_par=1)
