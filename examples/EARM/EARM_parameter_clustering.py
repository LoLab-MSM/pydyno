import os
import csv
import numpy as np
from parameter_clustering import dbscan


def listdir_fullpath(d):
    return [os.path.join(d, fi) for fi in os.listdir(d)]

directory = os.path.dirname(__file__)
parameters_path = listdir_fullpath(os.path.join(directory, "parameters_5000"))

parameters_data = np.zeros((len(parameters_path), 126))

for r, par in enumerate(parameters_path):
    f = open(par)
    data = csv.reader(f)
    params = [float(i[1]) for i in data]
    parameters_data[r] = params

parameters_clustered = dbscan(parameters_data, 'parameters_clustered')


# def num_to_path(number):
#     return parameters_path[number]
#
#
# parameters_clustered_path = {}
# for clus in parameters_clustered:
#     parameters_clustered_path['cluster_' + clus] = map(num_to_path, parameters_clustered[clus])
#
# # GETTING THE PARAMETERS FROM THE CLUSTERING OF TROPICAL SIGNATURES
#
# all_parameters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5000')
#
# clusters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_in_cluster5000')
#
# cluster_pars_path = {}
# for cluster in clusters_path:
#     ff = open(cluster)
#     data_paths = csv.reader(ff)
#     params_path = [dd[0] for dd in data_paths]
#     cluster_pars_path[cluster.split('0/')[1]] = params_path
#
#
# def removekey(d, key):
#     r = dict(d)
#     del r[key]
#     return r
#
#
# pars_clust_path_ready = removekey(parameters_clustered_path, 'cluster_k')
#
# all_intersections = list(itertools.product(*[cluster_pars_path.keys(), pars_clust_path_ready.keys()]))
#
# for i in all_intersections:
#     c21 = set(cluster_pars_path[i[0]]).intersection(parameters_clustered_path[i[1]])
#     print len(c21), 'dynamic_cluster', len(cluster_pars_path[i[0]]), 'static_cluster', len(
#         pars_clust_path_ready[i[1]]), 'cluster_names', i
