from tropical.dominant_path_analysis import run_dompath_multi, run_dompath_single
from tropical.examples.earm.earm2_flat import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator

pars = np.load('calibrated_6572pars.npy')
tspan = np.linspace(0, 20000, 1000)

# viz = ModelVisualization(model)
# data = viz.static_view(get_passengers=False)
sim = ScipyOdeSimulator(model, tspan).run(param_values=pars[0])
# dompath = run_dompath_multi(sim, ref=100, target='s27', depth=3, cpu_cores=2)
d = run_dompath_single(model, sim, target='s65', depth=12, dom_om=0.5)
