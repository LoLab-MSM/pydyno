from tropical.discretize_path import DomPath
from tropical.examples.jnk3.jnk3_no_ask1 import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from tropical.visualize_discretization import visualization_path

param_values = np.array([p.value for p in model.parameters])
idx_pars_calibrate = [1, 5, 9, 11, 15, 17, 19, 23, 25, 27, 31, 35, 36, 37, 38, 39, 41, 43] #pydream3
rates_of_interest_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]
pars = np.load('most_likely_par_100000.npy')
param_values[rates_of_interest_mask] = 10 ** pars
# param_values[44] = param_values[44] * 0.5
tspan = np.linspace(0, 60, 100)

sim = ScipyOdeSimulator(model, tspan, param_values=[param_values, param_values/2]).run()
# dompath = run_dompath_multi(sim, ref=100, target='s27', depth=3, cpu_cores=2)
d = DomPath(model, sim, type_analysis='consumption', target='s27', depth=7, dom_om=0.5)
signatures, paths = d.get_path_signatures(cpu_cores=2)
visualization_path(model, paths[1], type_analysis='consumption', filename='test.pdf')