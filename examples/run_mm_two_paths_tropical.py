import numpy as np

from mm_two_paths_model import model
from tropical.max_plus_multiprocessing import run_tropical

tspan = np.linspace(0, 50, 501)

run_tropical(model, tspan, diff_par=0.5, type_sign='consumption', sp_visualize=[0, 5])

