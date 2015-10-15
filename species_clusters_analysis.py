#Cluster analysis of species

import numpy as np
import pickle
import os
import random
import csv
from pysb.tools.max_monomials_signature import run_tropical
from earm.lopez_embedded import model

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

par_sets = listdir_fullpath('/home/oscar/tropical_project/parameters_2000')

species_clusters = pickle.load(open('/home/oscar/tropical_project/species_clusters_info.p', 'rb'))

tspan = np.linspace(0, 20000, 10000)

par_from_cluster = random.choice(species_clusters['20']['b'])
par_path = par_sets[par_from_cluster]
print par_path
f = open(par_path) 
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))
run_tropical(model, tspan, parameters=parames, sp_visualize=[20], stoch=False)
