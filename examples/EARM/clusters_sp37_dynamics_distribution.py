from earm.lopez_embedded import model
import numpy as np
from tropical.analysis_of_clusters import AnalysisCluster
from tropical.helper_functions import sig_apop
import os

directory = os.path.dirname(__file__)
all_simulations_path = os.path.join(directory, 'all_IC_simulations/all_simulations.npy')
pars_path = os.path.join(directory, 'all_IC_simulations/IC_10000_pars_consumption_kpar0.npy')
clusters_path = os.path.join(directory, 'bid_clusteredPAM_pars_consumption')
all_parameters = np.load(pars_path)
t = np.linspace(0, 20000, 100)
types = [1, 2, 3, 4]

clusters = [clusters_path+'/data_frame37_Type1.csv',
            clusters_path+'/data_frame37_Type2.csv',
            clusters_path+'/data_frame37_Type3.csv',
            clusters_path+'/data_frame37_Type4.csv'
            ]
simulations_result = np.load(all_simulations_path)
a = AnalysisCluster(model, t, all_parameters, clusters, sim_results=simulations_result)
a.plot_sp_IC_distributions([55, 56, 57], save_path=directory)
# a.plot_dynamics_cluster_types([39], save_path='/Users/dionisio/Documents/', species_to_fit=[39], fit_ftn=sig_apop, ic_idx=[23],
#                               **{'p0': [100, 100, 100]})
# a.plot_dynamics_cluster_types([33],  save_path='/Users/dionisio/Documents/', ic_idx=[19])
