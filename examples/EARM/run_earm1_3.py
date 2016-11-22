from pysb.examples.earm_1_3 import model
# from max_plus_consumption_production import run_tropical
from max_plus_multiprocessing import run_tropical
import numpy as np


t = np.linspace(0, 20000,  100)

run_tropical(model, t, diff_par=1, type_sign='consumption', sp_visualize=[6])
