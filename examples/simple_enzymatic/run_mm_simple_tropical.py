import numpy as np

from examples.simple_enzymatic.mm_simple_model import model
# from DynSign.max_plus_multiprocessing_numpy import run_tropical
from tropical.max_plus_pos_neg import run_tropical

tspan = np.linspace(0, 50, 501)

run_tropical(model, tspan, diff_par=0.5, type_sign='production', find_passengers_by='imp_nodes',
             plot_imposed_trace=True)
