from pysb.examples.tyson_oscillator import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from tropical.pydrone import Pydrone

tspan = np.linspace(0, 100, 100)
pars1 = np.array([par.value for par in model.parameters])
pars2 = [pars1, pars1, pars1/2, pars1/2]
sims = ScipyOdeSimulator(model, tspan=tspan).run(param_values=pars2)

pyd = Pydrone(model, sims, diff_par=1)
pyd.discretize(2)
print(pyd.sequences.loc['__s2_p'])
pyd.cluster_signatures_spectral('__s2_c', nclusters=3, cluster_range=True)
pyd.analysis_cluster.plot_cluster_dynamics([2], norm=True)