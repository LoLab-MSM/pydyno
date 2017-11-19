import os
import numpy as np
import tropical.helper_functions as hf
# from earm.lopez_embedded import model
from tropical.dynamic_signatures import run_tropical_multi
from pysb.simulator.cupsoda import CupSodaSimulator
import pickle

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'parameters_5000')
pars = hf.listdir_fullpath(pars_path)
parameters = hf.read_all_pars(pars)
t = np.linspace(0, 20000, 100)

# #getting simulations
# sim = CupSodaSimulator(model=model, tspan=t, param_values=parameters).run()
# sim.save()
# a=run_tropical_multi(model, simulations=sim, cpu_cores=4)
#
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)