# Clustering of parameter vectors

import matplotlib
import itertools

matplotlib.use('Agg')

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from sklearn import cluster


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def scatter(x, name):
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig('/home/oscar/Documents/tropical_project/' + name, dpi=150)
    plt.close(f)
    return


def dbscan_clus(x, name):
    np.random.seed(0)
    colors = np.array([col for col in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    X = x  # StandardScaler().fit_transform(x)
    dbscan_clus = cluster.DBSCAN(eps=1.40, min_samples=45)
    dbscan_clus.fit(X)
    if hasattr(dbscan_clus, 'labels_'):
        y_pred = dbscan_clus.labels_.astype(np.int)
    else:
        y_pred = dbscan_clus.predict(X)
    sc = ax.scatter(x[:, 0], x[:, 1], color=colors[y_pred].tolist(), lw=0, s=40)

    pars_colors = {}
    for c in 'bgrcmyk':
        pars_colors[c] = [idx for idx, co in enumerate(colors[y_pred].tolist()) if co == c]

    plt.savefig('/home/oscar/Documents/tropical_project/' + name, dpi=150)
    plt.close(f)
    return pars_colors


def spec_clus(x, name):
    np.random.seed(0)
    colors = np.array([col for col in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    spectral = cluster.SpectralClustering(n_clusters=5, eigen_solver='arpack', affinity='nearest_neighbors')
    spectral.fit(x)
    y_pred = spectral.labels_.astype(np.int)
    sc = ax.scatter(x[:, 0], x[:, 1], color=colors[y_pred].tolist(), lw=0, s=40)
    plt.savefig('/home/oscar/Documents/tropical_project/' + name, dpi=150)
    plt.close(f)
    return


def kmeans_clus(x, name):
    np.random.seed(0)
    colors = np.array([col for col in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    kmeans = cluster.KMeans(n_clusters=5)
    kmeans.fit(x)
    y_pred = kmeans.labels_.astype(np.int)
    sc = ax.scatter(x[:, 0], x[:, 1], color=colors[y_pred].tolist(), lw=0, s=40)
    plt.savefig('/home/oscar/Documents/tropical_project/' + name, dpi=150)
    plt.close(f)
    return x


parameters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5000')

parameters_data = np.zeros((len(parameters_path), 126))

for r, par in enumerate(parameters_path):
    f = open(par)
    data = csv.reader(f)
    params = []
    for i in data: params.append(float(i[1]))
    parameters_data[r] = params

cols_to_delete = []

for i in range(126):
    max_par = max(parameters_data[:, i])
    min_par = min(parameters_data[:, i])
    if max_par == min_par:
        cols_to_delete.append(i)
    else:
        for c in range(len(parameters_data[:, i])):
            # parameters_data[c, i] = (parameters_data[c, i] - np.mean(parameters_data[:,i])) / (np.std(parameters_data[:,i]))
            parameters_data[c, i] = (parameters_data[c, i] - min_par) / (max_par - min_par)

np.delete(parameters_data, cols_to_delete, 1)

x = TSNE(metric='euclidean', random_state=20150101).fit_transform(parameters_data)
# x = StandardScaler().fit_transform(x)
scatter(x, 'parameters')
parameters_clustered = dbscan_clus(x, 'parameters_clustered')


def num_to_path(number):
    return parameters_path[number]


parameters_clustered_path = {}
for clus in parameters_clustered:
    parameters_clustered_path['cluster_' + clus] = map(num_to_path, parameters_clustered[clus])

# GETTING THE PARAMETERS FROM THE CLUSTERING OF TROPICAL SIGNATURES

all_parameters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5000')

clusters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_in_cluster5000')

cluster_pars_path = {}
for sc in clusters_path:
    ff = open(sc)
    data_paths = csv.reader(ff)
    params_path = []
    for dd in data_paths: params_path.append(dd[0])
    cluster_pars_path[sc.split('0/')[1]] = params_path


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

pars_clust_path_ready = removekey(parameters_clustered_path, 'cluster_k')

all_intersections = list(itertools.product(*[cluster_pars_path.keys(), pars_clust_path_ready.keys()]))


for i in all_intersections:
    c21 = set(cluster_pars_path[i[0]]).intersection(parameters_clustered_path[i[1]])
    print len(c21), 'dynamic_cluster', len(cluster_pars_path[i[0]]), 'static_cluster', len(pars_clust_path_ready[i[1]]), 'cluster_names', i
