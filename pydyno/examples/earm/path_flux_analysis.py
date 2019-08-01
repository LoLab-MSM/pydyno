from pydyno.discretize_path import DomPath
from pydyno.examples.earm.earm2_flat import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator

pars = np.load('calibrated_6572pars.npy')
tspan = np.linspace(0, 20000, 1000)

# viz = ModelVisualization(model)
# data = viz.static_view(get_passengers=False)
sim = ScipyOdeSimulator(model, tspan).run(param_values=pars, num_processors=4)
# dompath = run_dompath_multi(sim, ref=100, target='s27', depth=3, cpu_cores=2)
dis = DomPath(model, sim, type_analysis='consumption', target='s65', depth=7, dom_om=0.5)
signatures, paths = dis.get_path_signatures(num_processors=2)
