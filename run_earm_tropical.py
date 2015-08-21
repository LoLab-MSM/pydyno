#DONT PUSH IT TO REPO

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



f = open('/home/carlos/Documents/tropical_project/embedded_pso_pars/parameters_2000/pars_embedded_400.txt') 
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))

tspan = np.linspace(0, 20000, 10000)

print run_tropical(model, tspan, parameters=parames, sp_visualize=None, stoch=False)[2]


