import numpy as np
from tropical.discretize import Discretize
from pysb.examples.tyson_oscillator import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 100, 100)
model.enable_synth_deg()
pars1 = [par.value for par in model.parameters]
pars2 = [pars1, pars1]
sims = ScipyOdeSimulator(model, tspan=tspan).run(param_values=pars2)

disc = Discretize(model, sims, diff_par=1)
signatures = disc.get_signatures()

