import numpy as np

from pysb.examples.tyson_oscillator import model
from tropical.max_plus_multiprocessing_numpy import run_tropical

tspan = np.linspace(0, 200, 200)
run_tropical(model, tspan, diff_par=.5, sp_visualize=[3, 5], find_passengers_by='imp_nodes', type_sign='production',
             verbose=False)
