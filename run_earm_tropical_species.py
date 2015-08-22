import sys
sys.path.insert(0, '/home/oscar/tropical_project/pysb')
sys.path.insert(0, '/home/oscar/tropical_project/earm-jpino')

import matplotlib
matplotlib.use('Agg')

from earm.lopez_embedded import model
from pysb.tools.max_monomials_signature import run_tropical
from multiprocessing import Pool
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import csv
import os, os.path
import collections
import pickle

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

f = open('/home/oscar/tropical_project/parameters_2000/pars_embedded_0.txt')
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))

tspan = np.linspace(0, 20000, 10000)

signatures, drivers = run_tropical(model, tspan, parameters=parames, sp_visualize=None, stoch = False)

species = species=collections.OrderedDict() 
for spe in drivers:
    species[spe] = []

def compare_parameters(par):

    f = open(par)
    data = csv.reader(f)
    params = []
    for d in data:params.append(float(d[1]))
    signs = run_tropical(model, tspan, parameters=params, sp_visualize = None, stoch = False)[0]
    return signs

p = Pool(32)
species_signs = p.map(compare_parameters,listdir_fullpath('/home/oscar/tropical_project/parameters_2000'))


for sp in drivers:
    for sign in species_signs:
        print sp
        species[sp].append(sign[sp])

#for sign in species_signs:
#    for sp in species_ready:
#        species[sp].append(sign[sp])

pickle.dump(species, open("/home/oscar/tropical_project/species_parameters_2000_drivers.p", "wb"))
print 'ready'

