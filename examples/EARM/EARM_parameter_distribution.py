from miscellaneous_analysis import parameter_distribution
from earm.lopez_embedded import model
import helper_functions as hf

# parameter_distribution('/home/oscar/Documents/tropical_project/parameters_5000', 'bind_L_R_to_LR_kr')
# parameter_distribution('/home/oscar/tropical_project_new/parameters_clusters/data_frame47_Type2', 'bind_L_R_to_LR_kr')

clus = hf.listdir_fullpath('/home/oscar/tropical_project_new/parameters_clusters')

for c in clus:
    for par in model.parameters:
        parameter_distribution(c, par.name)