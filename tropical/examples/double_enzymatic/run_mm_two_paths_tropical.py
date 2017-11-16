import numpy as np
from tropical.dynamic_signatures import run_tropical
from tropical.examples import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 50, 101)
sim = ScipyOdeSimulator(model, tspan=tspan).run()
run_tropical(model, simulations=sim, passengers_by='imp_nodes', diff_par=0.5)

