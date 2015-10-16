from earm.lopez_embedded import model
from tropicalize import run_tropical
from multiprocessing import Pool
import csv
import os
import numpy as np

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

tspan = np.linspace(0,20000,100)

def compare_drivers(par):

    f = open(par)
    data = csv.reader(f)
    params = []
    for d in data:params.append(float(d[1]))
    drivers = run_tropical(model, tspan, parameters=params, sp_visualize = None)
    return drivers

p = Pool(6)
all_drivers = p.map(compare_drivers,listdir_fullpath('/home/carlos/Documents/tropical_project/parameters_2000'))
