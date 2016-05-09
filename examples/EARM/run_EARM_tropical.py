from earm.lopez_embedded import model
from tropicalize import run_tropical
import numpy as np
import csv

# tipe 1 cluster: 5400
# type 2 cluster: 4052

f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_1795.txt')
data = csv.reader(f)
parames = [float(i[1]) for i in data]
t = np.linspace(0, 20000,  100)

run_tropical(model, t, parames, sp_visualize=[19, 20, 54, 47])

