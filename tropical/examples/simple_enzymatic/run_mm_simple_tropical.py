import numpy as np
from tropical.dynamic_signatures_range import run_tropical
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.examples.tyson_oscillator import model

tspan = np.linspace(0, 50, 501)
sim = ScipyOdeSimulator(model, tspan=tspan).run()
run_tropical(model, simulations=sim, diff_par=0.5)
