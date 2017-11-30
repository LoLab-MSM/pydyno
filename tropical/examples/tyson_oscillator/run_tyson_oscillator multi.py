import numpy as np
from tropical.dynamic_signatures_range import run_tropical_multi
from pysb.examples.tyson_oscillator import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 100, 100)
model.enable_synth_deg()
pars1 = [par.value for par in model.parameters]
pars2 = [pars1, pars1]
sims = ScipyOdeSimulator(model, tspan=tspan).run(param_values=pars2)

a = run_tropical_multi(model, simulations=sims)
