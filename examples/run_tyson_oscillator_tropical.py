import numpy as np

from pysb.examples.tyson_oscillator import model
from tropical.max_plus_multiprocessing import run_tropical

tspan = np.linspace(0, 200, 200)
run_tropical(model, tspan, sp_visualize=[3,5])
