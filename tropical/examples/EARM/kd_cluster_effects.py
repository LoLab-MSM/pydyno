from tropical.util import species_effect_tdeath
from pysb.simulator import ScipyOdeSimulator
from earm2_flat import model
import numpy as np
import pickle
import tropical.util as hf
tspan = np.linspace(0,20000, 100)

solver = ScipyOdeSimulator(model, tspan=tspan, integrator='vode')
calibrated_pars = np.load('calibrated_6572pars.npy')

with open('results_spectral.pickle', 'rb') as handle:
    clusters = pickle.load(handle)
clusters_sp37 = clusters[13]['labels']

num_of_clusters = set(clusters_sp37)
clus_pars = {}
for j in num_of_clusters:
    item_index = np.where(clusters_sp37 == j)
    clus_pars[j] = calibrated_pars[item_index]

perturbations = {56: 0.2, 57: 0.2, 58: 0.2}
effects = species_effect_tdeath(solver, cluster_pars=clus_pars, perturbations=perturbations,
                                ftn=hf.sig_apop, num_processors=1)