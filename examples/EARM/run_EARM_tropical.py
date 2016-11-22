from earm.lopez_embedded import model
from max_plus_multiprocessing import run_tropical
import numpy as np
import os
import helper_functions as hf

directory = os.path.dirname(__file__)
parameters_path = os.path.join(directory, "parameters_5000")
all_parameters = hf.listdir_fullpath(parameters_path)
parameters = hf.read_pars(all_parameters[0])
t = np.linspace(0, 20000,  100)

run_tropical(model, t, parameters, diff_par=1, type_sign='consumption', sp_visualize=[6])
