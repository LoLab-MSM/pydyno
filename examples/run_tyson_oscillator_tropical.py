from pysb.examples.tyson_oscillator import model 
# from max_monomials_signature import run_tropical
from tropicalize import run_tropical
import numpy as np
import matplotlib.pyplot as plt


tspan = np.linspace(0, 100, 100)
# run_tropical(model, tspan, sp_visualize=[3,5], stoch=True)
print run_tropical(model, tspan, sp_visualize='cdc2U_cyclinP')

