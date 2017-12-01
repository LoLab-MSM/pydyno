from pysb.simulator.base import SimulationResult
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from earm.lopez_embedded import model

obs_names = ['mBid', 'cPARP', 'aSmac']
sims = SimulationResult.load('simulations_earm6572.h5')
n_sims = sims.nsims
tspan = sims.tout[0]
plt.figure()
for i in range(n_sims):
    plt.plot(tspan, sims.observables[i]['aSmac'] / model.parameters['Smac_0'].value, color='blue')
plt.show()