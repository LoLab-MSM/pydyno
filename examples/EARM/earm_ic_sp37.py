from earm.lopez_embedded import model
import numpy as np
from tropical.analysis_of_clusters import AnalysisCluster

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

a = AnalysisCluster(model, t, all_parameters, clusters)
a.plot_sp_IC_distributions([55, 56, 57], '/home/oscar/Desktop')
# a.plot_dynamics_cluster_types([37], save_path='/home/oscar/Desktop', ic_idx=[55])
