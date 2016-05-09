# Clustering of parameter vectors

import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import cluster


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


colors = np.hstack([col for col in 'bgrcmykbgrcmykbgrcmykbgrcmyk'] * 20)


def tsne_cluster(parameters, **kwargs):
    standardized_pars = preprocessing.MinMaxScaler().fit_transform(parameters)
    tsne_out = TSNE(metric='euclidean', random_state=20150101, **kwargs).fit_transform(standardized_pars)
    return tsne_out


def scatter(parameter_sets, out_path):
    tsne_pars = tsne_cluster(parameter_sets)
    # We create a scatter plot.
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(tsne_pars[:, 0], tsne_pars[:, 1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return


def dbscan(parameter_sets, out_path, min_samples=45, eps=1.40, **kwargs):
    np.random.seed(0)
    tsne_pars = tsne_cluster(parameter_sets)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    dbscan_clus = cluster.DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    y_pred = dbscan_clus.fit_predict(tsne_pars)
    ax.scatter(tsne_pars[:, 0], tsne_pars[:, 1], color=colors[y_pred].tolist(), lw=0, s=40)

    pars_colors = {}
    for c in 'bgrcmyk':
        pars_colors[c] = [idx for idx, co in enumerate(colors[y_pred].tolist()) if co == c]

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return pars_colors


def spectral(parameter_sets, out_path, n_clusters=5, eigen_solver='arpack', affinity='nearest_neighbors', **kwargs):
    np.random.seed(0)
    tsne_pars = tsne_cluster(parameter_sets)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    spectral_clust = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, affinity=affinity,
                                                **kwargs)
    spectral_clust.fit(tsne_pars)
    y_pred = spectral_clust.labels_.astype(np.int)
    ax.scatter(tsne_pars[:, 0], tsne_pars[:, 1], color=colors[y_pred].tolist(), lw=0, s=40)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return


def kmeans(parameter_sets, out_path, n_clusters=5, **kwargs):
    np.random.seed(0)
    tsne_pars = tsne_cluster(parameter_sets)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    kmeans_clust = cluster.KMeans(n_clusters=n_clusters, **kwargs)
    kmeans_clust.fit(tsne_pars)
    y_pred = kmeans_clust.labels_.astype(np.int)
    ax.scatter(tsne_pars[:, 0], tsne_pars[:, 1], color=colors[y_pred].tolist(), lw=0, s=40)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return
