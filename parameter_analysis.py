#Clustering of parameter vectors

import matplotlib
matplotlib.use('Agg')

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import os
import numpy

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
    plt.savefig('/home/oscar/tropical_project/parameter_clustering/'+ name, dpi=150)
    plt.close(f)
    return


parameters_path = listdir_fullpath('/home/oscar/tropical_project/parameters_2')

parameters_data = numpy.zeros((len(parameters_path), 126))

for r, par in enumerate(parameters_path):
	f = open(par)
	data = csv.reader(f)
	params = []
	for i in data:params.append(float(i[1]))
	parameters_data[r] =  params

cols_to_delete = []

for i in range(126):
	max_par = max(parameters_data[:,i])
	min_par = min(parameters_data[:,i])
	if max_par == min_par: cols_to_delete.append(i)
	else:
		for c in range(len(parameters_data[:,i])):
			parameters_data[c,i] = (parameters_data[c,i] - min_par)/(max_par-min_par)

numpy.delete(parameters_data, cols_to_delete,1)

x = TSNE(metric='euclidean',random_state=20150101).fit_transform(parameters_data)
x = StandardScaler().fit_transform(x)
scatter(x, 'parameters1')

