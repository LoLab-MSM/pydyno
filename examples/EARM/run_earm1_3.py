import numpy as np

from old_functions.max_plus_multiprocessing_numpy import run_tropical
from pysb.examples.earm_1_3 import model

t = np.linspace(0, 20000,  100)

run_tropical(model, t, diff_par=1, type_sign='consumption', sp_visualize=[6])
