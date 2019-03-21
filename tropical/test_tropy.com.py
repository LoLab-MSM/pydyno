# Import libraries
import numpy as np
from tropical.discretize import Discretize
from tropical.examples.double_enzymatic.mm_two_paths_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator

# Run the model simulation to obtain the dynmics of the molecular species
tspan = np.linspace(0, 50, 101)
sim = ScipyOdeSimulator(model, tspan=tspan).run()

tro = Discretize(model=model, simulations=sim, diff_par=1)
tro.get_signatures()