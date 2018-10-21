import numpy as np
from tropical.discretize import Discretize
from tropical.examples.double_enzymatic.mm_two_paths_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator

pars = np.load('new_pars.npy')
tspan = np.linspace(0, 6, 51)
sim = ScipyOdeSimulator(model, tspan=tspan, param_values=pars[1800]).run()
disc = Discretize(model, sim, diff_par=0.5)
signatures = disc.get_signatures(cpu_cores=4)

