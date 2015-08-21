import sys
sys.path.insert(0, '/home/oscar/tropical_project/pysb')
sys.path.insert(0, '/home/oscar/tropical_project/earm-jpino')

import matplotlib
matplotlib.use('Agg')

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from earm.lopez_embedded import model
from pysb.tools.max_monomials import run_tropical
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def scatter(x, name):

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig('/home/oscar/tropical_project/tropical_clustering2_binario/'+ name, dpi=150)
    plt.close(f)
    return

def spec_clus(x,name):
    np.random.seed(0)
    colors = np.array([col for col in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    spectral = cluster.SpectralClustering(n_clusters = 2, eigen_solver = 'arpack', affinity = 'nearest_neighbors')
    spectral.fit(x)
    y_pred = spectral.labels_.astype(np.int)
    sc = ax.scatter(x[:,0], x[:,1], color=colors[y_pred].tolist(), lw=0, s=40)
    plt.savefig('/home/oscar/tropical_project/tropical_clustering2_spectral/'+ name, dpi=150)
    plt.close(f)
    return 


def dbscan_clus(x,name):
    np.random.seed(0)
    colors = np.array([col for col in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    x = StandardScaler().fit_transform(x)
    dbscan_clus = cluster.DBSCAN(eps=0.2)
    dbscan_clus.fit(x)       
    if hasattr(dbscan_clus, 'labels_'):
        y_pred = dbscan_clus.labels_.astype(np.int)
    else:
        y_pred = dbscan_clus.predict(X)
    sc = ax.scatter(x[:,0], x[:,1], color=colors[y_pred].tolist(), lw=0, s=40)
    
    pars_colors = {}
    for c in 'bgrcmyk':
        pars_colors[c] = [idx for idx,co in enumerate(colors[y_pred].tolist()) if co == c]
    
    plt.savefig('/home/carlos/Documents/tropical_project/tropical_clustering2000_euclidean_dbscan/'+ name, dpi=150)
    plt.close(f)
    return pars_colors

species = pickle.load(open('/home/carlos/Documents/tropical_project/species_monomers_parameters_euclidean_2000.p', 'rb'))

drivers=[14,19,37]

if drivers is not None:
    species_ready = []
    for i in drivers:
        if i in species.keys(): species_ready.append(i)
        else: print 'specie' + ' ' + str(i) + ' ' + 'is not a driver'
elif driver_species is None:
    raise Exception('list of driver species must be defined')

if species_ready == []:
    raise Exception('None of the input species is a driver')


monomials_clusters = {}
for spp in species_ready:
    for mm in species[spp].keys():
        print spp, mm
        specie_normalized = StandardScaler().fit_transform(species[spp][mm])
        x = TSNE(metric='euclidean',random_state=20150101).fit_transform(specie_normalized)
        monomials_clusters[str(spp)+str(mm)] = dbscan_clus(x, str(spp)+str(mm))

par_sets = listdir_fullpath('/home/carlos/Documents/tropical_project/embedded_pso_pars/parameters_2000')        

for clus in monomials_clustersp["14__s14*__s17*bind_BadM_Bcl2_kf"].keys():
    par_from_cluster = random.choice(monomials_clusters["14__s14*__s17*bind_BadM_Bcl2_kf"][clus])
    par_path = par_sets[par_from_cluster]
    f = open(par_path) 
    data = csv.reader(f)
    parames = []
    for i in data:parames.append(float(i[1]))
    run_tropical(model, tspan, parameters=parames, sp_visualize=[14,19,37], stoch=False)
        


