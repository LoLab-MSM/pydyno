import numpy as np
from tropical.discretize import Discretize
from pysb.examples.tyson_oscillator import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 100, 100)
model.enable_synth_deg()
sims = ScipyOdeSimulator(model, tspan=tspan).run()

disc = Discretize(model, sims, diff_par=1)
signatures = disc.get_signatures()


