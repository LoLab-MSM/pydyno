import numpy as np

from old_functions.max_plus_multiprocessing_numpy import run_tropical
from pysb.examples.tyson_oscillator import model

# from DynSign.max_plus_pos_neg import run_tropical

tspan = np.linspace(0, 100, 100)
run_tropical(model, tspan, diff_par=.5, sp_visualize=None, find_passengers_by='imp_nodes', plot_imposed_trace=False,
             type_sign='consumption', verbose=False)