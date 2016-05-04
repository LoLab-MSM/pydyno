from mm_simple_model import model
from tropicalize import run_tropical
import numpy as np

tspan = np.linspace(0, 5, 51)

run_tropical(model, tspan, sp_visualize=[3, 5])
