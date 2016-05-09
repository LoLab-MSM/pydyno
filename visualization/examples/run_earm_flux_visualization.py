from __future__ import print_function
from earm.lopez_embedded import model
from visualization.species_visualization import run_flux_visualization
import numpy as np
import csv

f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_5400.txt')
data = csv.reader(f)
parames = [float(i[1]) for i in data]

tspan = np.linspace(0, 10000, 100)

run_flux_visualization(model, tspan, parameters=parames)
print('finished')
