import numpy as np
from pysb.examples.tyson_oscillator import model
from tropical.max_plus_global import run_tropical

tspan = np.linspace(0, 100, 100)
run_tropical(model, tspan, diff_par=.5, sp_visualize=None, find_passengers_by='imp_nodes', plot_imposed_trace=True,
             global_signature=True, type_sign='production', verbose=False)
