from pysb.examples.tyson_oscillator import model
from max_plus_multiprocessing import run_tropical
import numpy as np

tspan = np.linspace(0, 200, 200)
run_tropical(model, tspan, sp_visualize=[3,5])
