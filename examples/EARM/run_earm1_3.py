import numpy as np

from pysb.examples.earm_1_3 import model
from tropical.max_plus_multiprocessing import run_tropical

t = np.linspace(0, 20000,  100)

run_tropical(model, t, diff_par=1, type_sign='consumption', sp_visualize=[6])
