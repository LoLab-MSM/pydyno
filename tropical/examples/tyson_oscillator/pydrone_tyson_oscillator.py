from run_tyson_oscillator import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from tropical.pydrone import Pydrone

tspan = np.linspace(0, 100, 100)
pars1 = np.array([par.value for par in model.parameters])
pars2 = [pars1, pars1, pars1/2, pars1/2]
sims = ScipyOdeSimulator(model, tspan=tspan).run(param_values=pars2)

pyd = Pydrone(model, sims, diff_par=1)
