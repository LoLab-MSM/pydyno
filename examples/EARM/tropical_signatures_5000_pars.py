import os
import numpy as np
import tropical.helper_functions as hf
from earm.lopez_embedded import model
from tropical.dynamic_signatures import run_tropical
from pysb.simulator import SimulationResult

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'parameters_5000')
pars = hf.listdir_fullpath(pars_path)
parameters = hf.read_all_pars(pars)
t = np.linspace(0, 20000, 100)

# tro = Tropical(model)
# tro.get_simulations(parameters[[0,1]], tspan=t, simulator='scipy')
# traj = SimulationResult.load('/Users/dionisio/Desktop/bla')
a=run_tropical(model, trajectories='/Users/dionisio/Desktop/bla', tspan=t, simulator='scipy', cpu_cores=4)
# a=run_tropical(model, trajectories=parameters[[0,1]], tspan=t, simulator='scipy', cpu_cores=4)