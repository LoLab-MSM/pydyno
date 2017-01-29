from earm.lopez_embedded import model
import numpy as np
from tropical.analysis_of_clusters import AnalysisCluster
from tropical.helper_functions import sig_apop

all_parameters = np.load('/home/oscar/Documents/tropical_earm_IC/IC_10000_parameters_consumption.npy')
t = np.linspace(0, 20000, 100)
types = [1, 2, 3, 4]

clusters = ['/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type1',
            '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type2',
            '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type3',
            '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type4'
            ]
simulations_result = np.load('/home/oscar/Desktop/all_simulations.npy')
a = AnalysisCluster(model, t, all_parameters, clusters, sim_results=simulations_result)
# a.plot_sp_IC_distributions([55, 56, 57], '/home/oscar/Desktop')
a.plot_dynamics_cluster_types([39], save_path='/home/oscar/Desktop', species_to_fit=[39], fit_ftn=sig_apop, ic_idx=[23],
                              **{'p0': [100, 100, 100]})
