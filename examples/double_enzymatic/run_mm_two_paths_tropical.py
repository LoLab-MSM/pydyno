import numpy as np

from examples.double_enzymatic.mm_two_paths_model import model
from old_functions.max_plus_multiprocessing_numpy import run_tropical
# from DynSign.max_plus_pos_neg import run_tropical

tspan = np.linspace(0, 50, 101)

run_tropical(model, tspan, diff_par=0.5, type_sign='consumption', find_passengers_by='imp_nodes', sp_visualize=[0])

