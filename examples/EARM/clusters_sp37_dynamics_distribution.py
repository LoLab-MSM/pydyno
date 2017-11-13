from earm.lopez_embedded import model
import numpy as np
from tropical.analysis_of_clusters import AnalysisCluster
# from DynSign.helper_functions import sig_apop
import tropical.helper_functions as hf
import os
import csv

directory = os.path.dirname(__file__)
parameters_path = os.path.join(directory, "parameters_5000")
all_parameters = hf.listdir_fullpath(parameters_path)
parameters = hf.read_all_pars(all_parameters)

clusters = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/signatures_all_kpars/' \
           'hdbscan_clusters.npy'

t = np.linspace(0, 20000, 100)
pars_clusters = np.load(clusters)

## ALL KPARS
# parameters_cluster = parameters
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/signatures_all_kpars/simulations.npy'
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 KD bcl2 cluster 1

# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_kd08_bcl2/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 1.8 oe bcl2 cluster 1

# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_oe08_bcl2/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 kd bcl2 1.5 oe bax cluster 1

# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_kd08_bcl2_bax/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)


## ALL KPARS 0.8 kd bcl2 1.5 oe bak cluster 1

# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_kd08_bcl2_bak/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 kd mcl1 cluster 2

# item_index = np.where(pars_clusters == 2)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus2_kd08_mcl1/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 kd bcl2 cluster 2

# item_index = np.where(pars_clusters == 2)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus2_kd08_bcl2/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 kd bcl2, bclxl cluster 2

# item_index = np.where(pars_clusters == 2)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus2_kd08_bcl2_bclxl/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 kd bcl2, mcl1 cluster 2

# item_index = np.where(pars_clusters == 2)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus2_kd08_mcl1_bcl2/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 KD mcl1 cluster 1
#
# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_kd08_mcl1/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 1.8 oe mcl1 cluster 1

# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_oe08_mcl1/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

## ALL KPARS 0.8 KD bclxl cluster 1

# item_index = np.where(pars_clusters == 1)
# parameters_cluster = parameters[item_index]
#
# all_simulations_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
#                        'all_pars_clus1_kd08_bclxl/simulations.npy'
#
# simulations_result = np.load(all_simulations_path)

# a = AnalysisCluster(model, t, parameters_cluster, clusters=None, sim_results=simulations_result)
a = AnalysisCluster(model, t, parameters_cluster, clusters=None, sim_results=simulations_result)
# a.plot_sp_ic_distributions([55, 56, 57, 58], save_path='/Users/dionisio/Documents/')
# a.plot_clusters_ic_distributions([63, 64, 56], save_path='/Users/dionisio/Documents/')
# a.violin_plot_sps([82, 83, 84, 85, 86, 87], save_path='/Users/dionisio/Documents/')
a.plot_dynamics_cluster_types([39], save_path='/Users/dionisio/Documents/', species_to_fit=[39], fit_ftn=hf.sig_apop,
                              norm=True, **{'p0': [100, 100, 100]})

# a.plot_dynamics_cluster_types([39], save_path='/Users/dionisio/Documents/', norm=True)

# a.plot_dynamics_cluster_types([13, 47, 15],  save_path='/Users/dionisio/Documents/', norm=True)

# a.scatter_plot_pars(ic_par_idxs=[56, 64], cluster=3, save_path='/Users/dionisio/Documents/')