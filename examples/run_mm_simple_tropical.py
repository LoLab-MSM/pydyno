from mm_simple_model import model
from max_plus_multiprocessing import run_tropical
import numpy as np

tspan = np.linspace(0, 50, 501)

run_tropical(model, tspan, sp_visualize=None)
