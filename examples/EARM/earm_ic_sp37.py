from earm.lopez_embedded import model
import numpy as np
from tropical.analysis_of_clusters import AnalysisCluster
from tropical.helper_functions import sig_apop
import os

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'IC_10000_pars_consumption_kpar0.npy')

all_parameters = np.load(pars_path)
t = np.linspace(0, 20000, 100)
types = [1, 2, 3, 4]

clusters = ['/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type1.csv',
            '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type2',
            '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type3',
            '/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type4'
            ]
simulations_result = np.load('/Users/dionisio/Documents/PhD/tropical_earm_IC/KPAR_0/all_simulations.npy')
a = AnalysisCluster(model, t, all_parameters, clusters, sim_results=simulations_result)
# a.plot_sp_IC_distributions([55, 56, 57], '/home/oscar/Desktop')
a.plot_dynamics_cluster_types([39], save_path='/Users/dionisio/Documents/', species_to_fit=[39], fit_ftn=sig_apop, ic_idx=[23],
                              **{'p0': [100, 100, 100]})
