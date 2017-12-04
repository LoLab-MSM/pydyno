import os
import numpy as np
import tropical.helper_functions as hf
import pickle
from tropical.dynamic_signatures_range import run_tropical_multi
from pysb.simulator.base import SimulationResult
from earm.lopez_embedded import model

t = np.linspace(0, 20000, 100)

#getting simulations
sim = SimulationResult.load('simulations_earm6572.h5')
a=run_tropical_multi(model, simulations=sim, cpu_cores=4)

with open('earm_signatures.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)