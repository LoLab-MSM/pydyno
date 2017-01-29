import numpy as np

from examples.double_enzymatic.mm_two_paths_model import model
from tropical.max_plus_multiprocessing_numpy import run_tropical
tspan = np.linspace(0, 50, 101)

run_tropical(model, tspan, diff_par=0.5, type_sign='production', sp_visualize=[5, 0])

