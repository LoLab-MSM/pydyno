import os
import numpy as np
import tropical.helper_functions as hf
import pickle
from tropical.dynamic_signatures_range import run_tropical_multi
from pysb.simulator.base import SimulationResult
from earm.lopez_embedded import model
import matplotlib
matplotlib.use('TkAgg')


directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'parameters_5000')
pars = hf.listdir_fullpath(pars_path)
parameters = hf.read_all_pars(pars)
t = np.linspace(0, 20000, 100)

#getting simulations
sim = SimulationResult.load('simulations_earm6572.h5')
a=run_tropical_multi(model, simulations=sim, cpu_cores=4)

with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)