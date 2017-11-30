import numpy as np
from tropical.dynamic_signatures_range import run_tropical
from pysb.examples.tyson_oscillator import model
from pysb.simulator.scipyode import ScipyOdeSimulator

tspan = np.linspace(0, 100, 100)
model.enable_synth_deg()
sims = ScipyOdeSimulator(model, tspan=tspan).run()

a = run_tropical(model, simulations=sims)

