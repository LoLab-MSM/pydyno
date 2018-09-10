from tropical.discretize import Discretize
from tropical.examples.tyson_oscillator.run_tyson_oscillator import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from tropical.clustering import ClusterSequences

tspan = np.linspace(0, 100, 100)
pars1 = np.array([par.value for par in model.parameters])
pars2 = [pars1, pars1, pars1/2, pars1/2]
sims = ScipyOdeSimulator(model, tspan=tspan).run(param_values=pars2)

dis = Discretize(model, sims, 1)
a=dis.get_signatures(1)

b = ClusterSequences(a.loc['__s2_c'])
b.diss_matrix()
c = b.silhouette_score_agglomerative_range([2, 3])