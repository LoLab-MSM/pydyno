from earm.lopez_embedded import model
import numpy as np
import csv
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt

all_parameters = np.load('/home/oscar/Documents/tropical_earm_IC/IC_10000_parameters_consumption.npy')
t = np.linspace(0, 20000, 100)
types = [1, 2, 3, 4]


def plot_bid_dynamics_cluster_types():
    sim = ScipyOdeSimulator(model=model, tspan=t)
    for idx in types:
        path_cluster_type = '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption' \
                            '/data_frame37_Type{0}'.format(idx)
        f = open(path_cluster_type)
        data = csv.reader(f)
        pars_idx = [int(d[0]) for d in data]
        plt.figure(1)
        for i, j in enumerate(pars_idx):
            parameters = all_parameters[j]
            bid_0 = all_parameters[j][55]
            x = sim.run(param_values=parameters).all
            plt.plot(t, x['__s37']/bid_0, color='b')
            # plt.plot(t, x['__s44'], color='r')
            # plt.plot(t, x['__s45'], color='green')
            plt.ylim(0, 1)
            plt.xlabel('Time(s)')
            plt.ylabel('Population')
            plt.title('Bid in mitochondria')
        print(idx)
        plt.savefig('/home/oscar/Desktop/Bid_type{0}'.format(idx))
        plt.clf()


def plot_bid_IC():
    for idx in types:
        print(idx)
        path_cluster_type = '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
                            'data_frame37_Type{0}'.format(idx)
        f = open(path_cluster_type)
        data = csv.reader(f)
        pars_idx = [int(d[0]) for d in data]
        cluster_pars = all_parameters[pars_idx]
        bid_values = cluster_pars[:, 55]
        bcl2_values = cluster_pars[:, 58]
        bclxl_values = cluster_pars[:, 56]
        mcl1_values = cluster_pars[:, 57]
        weights_bid = np.ones_like(bid_values) / len(bid_values)
        weights_bcl2 = np.ones_like(bcl2_values) / len(bcl2_values)
        weights_bclxl = np.ones_like(bclxl_values) / len(bclxl_values)
        weights_mcl1 = np.ones_like(mcl1_values) / len(mcl1_values)
        plt.figure(1)
        plt.hist(bid_values, bins=20, weights=weights_bid, alpha=0.4, color='gray', label='Bid_0')
        plt.hist(bcl2_values, bins=20, weights=weights_bcl2, alpha=0.4, color='b', label='Bcl2_0')
        plt.hist(bclxl_values, bins=20, weights=weights_bclxl, alpha=0.4, color='r', label='BclxL_0')
        plt.hist(mcl1_values, bins=20, weights=weights_mcl1, alpha=0.4, color='green', label='Mcl1')
        plt.legend(loc=0)
        plt.title('Type{0}'.format(idx))
        plt.xlim(5000, 100000)
        plt.ylim(0, 0.2)
        plt.xlabel('concentration')
        plt.ylabel('percentage')
        plt.savefig('/home/oscar/Documents/tropical_earm_IC/type{0}'.format(idx))
        plt.clf()
    return

# plot_bid_dynamics_cluster_types()
# plot_bid_IC()

clusters = ['/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type1',
            '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type2',
            '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type3',
            '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/' \
            'data_frame37_Type4'
            ]

from tropical.analysis_of_clusters import AnalysisCluster
a = AnalysisCluster(model, t, all_parameters, clusters)
# a.plot_sp_IC_distributions([55, 56, 57], '/home/oscar/Desktop')
a.plot_dynamics_cluster_types([37], save_path='/home/oscar/Desktop', ic_idx=[55])
