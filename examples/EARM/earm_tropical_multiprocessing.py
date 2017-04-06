import os
import numpy as np
import tropical.helper_functions as hf
from earm.lopez_embedded import model
from tropical.max_plus_global import run_tropical_multiprocessing

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'parameters_5000')
pars = hf.listdir_fullpath(pars_path)
t = np.linspace(0, 20000, 100)

run_tropical_multiprocessing(model, t, pars[:10], global_signature=True, dir_path=directory)
