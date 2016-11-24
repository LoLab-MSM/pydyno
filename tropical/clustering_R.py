import os
# import rpy2's package module
import rpy2.robjects.packages as rpackages
# R vector of strings
from rpy2.robjects.vectors import StrVector
# import R's utility package
utils = rpackages.importr('utils')
base = rpackages.importr('base')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the listls')

# R package names
packnames = ('WeightedCluster', 'TraMineR', 'seqdist2')

# Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

directory = os.path.dirname(__file__)
r_cluster_function_path = os.path.join(directory, 'clustering_sequences.R')
cluster_function = base.dget(r_cluster_function_path)

df_list = '/home/oscar/Documents/tropical_earm/earm_dataframes_consumption/data_frame37.csv'
sm_df = '/home/oscar/Documents/tropical_earm/subs_matrix_consumption/sm_37.csv'

cluster_function(df_list, sm='CONSTANT', nclusters=4, clusterMethod='PAM', clustered_pars_path='/home/oscar/Documents/tropical_earm/clustered_parameters_bid_consumption/')
