from pydyno.discretize import Discretize
from pydyno.examples.tyson_oscillator.run_tyson_oscillator import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator

tspan = np.linspace(0, 100, 100)
pars1 = np.array([par.value for par in model.parameters])
pars2 = [pars1, pars1, pars1/2, pars1/2]
sims = ScipyOdeSimulator(model, tspan=tspan).run(param_values=pars2)

dis = Discretize(model, sims, 1)
signatures = dis.get_signatures(cpu_cores=1)
print(signatures)