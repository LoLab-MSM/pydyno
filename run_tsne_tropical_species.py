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

    plt.savefig('/home/oscar/tropical_project/tropical_clustering2000_species_dbscan_drivers_hamming/'+ name, dpi=150)
    plt.close(f)
    return pars_colors

species = pickle.load(open('/home/oscar/tropical_project/species_parameters_2000_drivers.p', 'rb'))

drivers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]


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
    print spp
    x = TSNE(metric='hamming',random_state=20150101).fit_transform(species[spp])
    dbscan_clus(x, str(spp))
    monomials_clusters[str(spp)] = dbscan_clus(x, str(spp))

pickle.dump(monomials_clusters, open('/home/oscar/tropical_project/species_clusters_info_hamming.p', 'wb'))

