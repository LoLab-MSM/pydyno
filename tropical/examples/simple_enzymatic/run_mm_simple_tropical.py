import numpy as np

from pysb.simulator.scipyode import ScipyOdeSimulator
from tropical.dynamic_signatures import run_tropical
from tropical.examples import model

tspan = np.linspace(0, 50, 501)
sim = ScipyOdeSimulator(model, tspan=tspan).run()
run_tropical(model, simulations=sim, diff_par=0.5)
