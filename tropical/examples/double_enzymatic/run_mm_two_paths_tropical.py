import numpy as np
from tropical.discretize import Discretize
from tropical.examples.double_enzymatic.mm_two_paths_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 10, 51)
sim = ScipyOdeSimulator(model, tspan=tspan).run()
disc = Discretize(model, sim, diff_par=0.5)
signatures = disc.get_signatures()

