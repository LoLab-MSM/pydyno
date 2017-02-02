from earm.lopez_embedded import model
from tropical.max_plus_multiprocessing_numpy import run_tropical_multiprocessing
import numpy as np
import os
import tropical.helper_functions as hf

directory = os.path.dirname(__file__)
parameters_path = os.path.join(directory, "parameters_5000")
all_parameters = hf.listdir_fullpath(parameters_path)
parameters = hf.read_all_pars(all_parameters)

t = np.linspace(0, 20000, 100)
a = run_tropical_multiprocessing(model, t, parameters, type_sign='consumption',
                                 find_passengers_by='imp_nodes', verbose=False)