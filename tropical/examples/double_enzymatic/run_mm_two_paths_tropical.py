import numpy as np
from tropical.dynamic_signatures_range import run_tropical
from tropical.examples.double_enzymatic.mm_two_paths_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 10, 51)
sim = ScipyOdeSimulator(model, tspan=tspan).run()
a = run_tropical(model, simulations=sim, passengers_by='imp_nodes', diff_par=0.5, sp_to_vis=[0])
