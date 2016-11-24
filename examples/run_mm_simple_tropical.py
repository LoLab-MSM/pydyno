import numpy as np

from mm_simple_model import model
from tropical.max_plus_multiprocessing import run_tropical

tspan = np.linspace(0, 50, 501)

run_tropical(model, tspan, sp_visualize=None)
