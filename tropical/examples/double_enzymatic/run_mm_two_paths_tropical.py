import numpy as np
from tropical.dynamic_signatures_range import run_tropical_multi
from tropical.examples.double_enzymatic.mm_two_paths_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator

parameters = np.load('calibrated_pars.npy')
parameters = parameters[:901]

tspan = np.linspace(0, 50, 101)
sim = ScipyOdeSimulator(model, tspan=tspan).run(param_values=parameters)
a = run_tropical_multi(model, simulations=sim, passengers_by='imp_nodes', diff_par=0.5)

# from tropical.dynamic_signatures_range import Tropical
# tro=Tropical(model)
# tro.equations_to_tropicalize()
# tro.set_combinations_sm()
# tro.all_comb
