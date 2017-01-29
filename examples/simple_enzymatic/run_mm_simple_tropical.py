import numpy as np

from examples.simple_enzymatic.mm_simple_model import model
from tropical.max_plus_multiprocessing_numpy import run_tropical

tspan = np.linspace(0, 50, 501)

run_tropical(model, tspan, diff_par=0.5, type_sign='production', find_passengers_by='qssa',
             plot_imposed_trace=True)
