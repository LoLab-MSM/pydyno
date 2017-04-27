import tropical.helper_functions as hf

from earm.lopez_embedded import model
from tropical.miscellaneous_analysis import parameter_distribution

# Script to get the comparison of parameter distribution between different parameter clusters in EARM

clus = hf.listdir_fullpath('/home/oscar/Documents/tropical_earm/clustered_parameters_bid')
new_path = '/home/oscar/Documents/tropical_earm/parameters_5000'

for par in model.parameters:
    parameter_distribution(clus, par.name, new_path)