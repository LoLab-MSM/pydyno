#DONT PUSH IT TO REPO

import sys
sys.path.insert(0, '/home/oscar/tropical_project/pysb')
sys.path.insert(0, '/home/oscar/tropical_project/earm-jpino')

import matplotlib
matplotlib.use('Agg')

from earm.lopez_embedded import model 
from pysb.tools.max_monomials import run_tropical
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

f = open('/home/oscar/tropical_project/parameters_5000/pars_embedded_0.txt') 
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))

tspan = np.linspace(0, 20000, 10000)

m_data, m_names = run_tropical(model, tspan, parameters=parames, sp_visualize=None, stoch=False)

drivers=[14,19,37]

if drivers is not None:
    species_ready = []
    for i in drivers:
        if i in m_data.keys(): species_ready.append(i)
        else: print 'specie' + ' ' + str(i) + ' ' + 'is not a driver'
elif driver_species is None:
    raise Exception('list of driver species must be defined')

if species_ready == []:
    raise Exception('None of the input species is a driver')    

species=collections.OrderedDict()     
for spe in species_ready:
    tmp = collections.OrderedDict()
    for mon in m_names[spe]:
        tmp[str(mon)] = [] 
    species[spe] = tmp
    

def compare_parameters(par):

    f = open(par) 
    data = csv.reader(f)
    params = []
    for d in data:params.append(float(d[1])) 
    monomials_data, monomial_names = run_tropical(model, tspan, parameters=params, sp_visualize=None, stoch=False)
                     
    return monomials_data   

p = Pool()
monomials_pars = p.map(compare_parameters,listdir_fullpath('/home/oscar/tropical_project/parameters_5000'))         

for monomials_data in monomials_pars:            
    for sp in species_ready:
        for idx, m in enumerate(species[sp].keys()):
#            tmp1 = monomials_data[sp][idx].astype(int)
#            tmp2 = tmp1.astype(bool)
#            tmp3 = tmp2.astype(int)
            species[sp][m].append(monomials_data[sp][idx])

pickle.dump(species, open("/home/oscar/tropical_project/species_monomers_parameters_euclidean_2500.p","wb"))
            

