import os
import tropical.helper_functions as hf
import numpy as np
from earm.lopez_embedded import model
# from tropical.max_plus_multiprocessing_numpy import run_tropical
from tropical.max_plus_global_new import run_tropical

directory = os.path.dirname(__file__)
parameters_path = os.path.join(directory, "parameters_5000")
all_parameters = hf.listdir_fullpath(parameters_path)
parameters = hf.read_pars(all_parameters[237])
# parameters[58] = 0
# parameters[63] = 0
# parameters[56] = 0
t = np.linspace(0, 20000,  100)

# run_tropical(model, t, parameters, diff_par=1, type_sign='consumption', sp_visualize=[37])
a=run_tropical(model, t, parameters, diff_par=0.1, type_sign='consumption', global_signature=False,
             pre_equilibrate=False, sp_visualize=[37])

