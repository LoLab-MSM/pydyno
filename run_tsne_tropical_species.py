from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as metrics 
import sys
sys.setrecursionlimit(10000)

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

    plt.savefig(name, dpi=150)
    plt.close(f)
    return pars_colors

for sp in listdir_fullpath('/home/carlos/Documents/tropical_project/species_pars_info'):
    species = np.load(sp)
    if len(np.unique(species)) == 1: print sp + ' ' + 'only has one driver monomial'
    else:
        dist_matrix = metrics.pdist(species, 'euclidean')
        fig = plt.figure(1)
        
        Y = sch.linkage(dist_matrix, method='average')
        print Y
        Z = sch.dendrogram(Y)
        fig.show()

#     ax = fig.add_subplot(111, projection='3d')
#     
#     x = TSNE(n_components=3,metric='minkowski',random_state=20150101).fit_transform(species)
#     ax.scatter(x[:,0],x[:,1],x[:,2])
#     dbscan_clus(x, sp.split('.')[0])    


# monomials_clusters[str(spp)] = dbscan_clus(x, str(spp))
# pickle.dump(monomials_clusters, open('/home/oscar/tropical_project/species_clusters_info_hamming.p', 'wb'))

