from __future__ import division
import mlpy
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.cluster as cluster
import numpy as np
from sklearn import metrics
from tropical.pam_clustering import kMedoids
import colorsys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import OrderedDict
import random
import hdbscan
from scipy import stats
from tropical.distinct_colors import distinct_colors
import matplotlib.patches as mpatches
from tropical import helper_functions as hf

signatures_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
                  'signatures_all_kpars/data_frames/data_frame37.csv'
X = pd.read_csv(signatures_path, header=0, index_col=0).as_matrix()[:, 0:50]
unique_sequences, indices, seq_weigths = np.unique(X, return_counts=True, return_inverse=True, axis=0)


def lcs_distance2(seq1, seq2):
    return mlpy.lcs_std(seq1, seq1)[0] + mlpy.lcs_std(seq2, seq2)[0] - 2*mlpy.lcs_std(seq1, seq2)[0]


def lcs_distance(seq1, seq2):
    seq_len = len(seq1)
    return 2 * seq_len - 2 * mlpy.lcs_std(seq1, seq2)[0]

diss = pairwise_distances(unique_sequences, metric=lcs_distance, n_jobs=1)

## HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, metric='precomputed').fit(diss)


## PAM
pam1 = cluster.KMedoids(distance_metric=lcs_distance, n_clusters=10).fit(unique_sequences)
pam2 = kMedoids(diss, 10)[1]

pam1_labels = pam1.labels_
pam2_labels = np.zeros(shape=len(unique_sequences), dtype=np.int)

for clus, seqs_idx in pam2.items():
    pam2_labels[seqs_idx] = int(clus)

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X=diss, labels=pam1.labels_, metric='precomputed'))


# plot function
unique_states = np.unique(unique_sequences)
N = len(unique_states)

# function to get n different colors
# colores = [colorsys.hsv_to_rgb(x*1.0/N, 0.5, 0.7) for x in range(N)]
# random.shuffle(colores)

colores = distinct_colors(N)

states_color_map = OrderedDict((state, colores[x]) for x, state in enumerate(unique_states))
cmap = ListedColormap(states_color_map.values())
bounds = states_color_map.keys()
norm = BoundaryNorm(bounds, cmap.N)

# TEMPORAL DICTIONARY OF REACTION RATES
# from DynSign.max_plus_global_new import Tropical
# from earm.lopez_embedded import model
# from pysb import Parameter
#
# tro = Tropical(model)
# tro.setup_tropical(tspan=[1,2,3], type_sign='consumption', find_passengers_by='imp_nodes', max_comb=None, verbose=False)
# comb_dict =tro.get_comb_dict()
# flat_comb = hf.merge_dicts(*comb_dict[37].values())

# parsed_comb = {}
# for item in flat_comb:
#     for rr in flat_comb[item]:
#         bla = [n.name for n in rr.atoms() if type(n) is Parameter and n.name.endswith(('kf', 'kc'))]

time = np.linspace(0, 20000, 100)/3600


def plot_sequences(all_sequences, cluster_labels, title='', type_plot='all'):
    """

    :param all_sequences:
    :param cluster_labels:
    :param type_plot: it can be all, mode
    :return:
    """
    # sil_samples = metrics.silhouette_samples(X=diss, labels=cluster_labels, metric='precomputed')
    if type_plot == 'mode':
        # plt.figure(1)
        f, axs = plt.subplots(4, 3, sharex=True, sharey=True)
        axs = axs.reshape(12)
        axs[-1].axis('off')
        for i in set(cluster_labels):
            clus_seqs = all_sequences[cluster_labels == i]
            seqs_in_cluster = clus_seqs.shape[0]
            modal_states, mode_counts = stats.mode(clus_seqs, axis=0)
            mc_norm = np.divide(mode_counts[0], seqs_in_cluster, dtype=np.float)
            ind = time[:50]
            width_bar = ind[1] - ind[0]
            colors = [states_color_map[c] for c in modal_states[0]]
            legend_patches = [mpatches.Patch(color=states_color_map[c], label=c) for c in set(modal_states[0])]
            axs[i+1].bar(ind, mc_norm, color=colors, width=width_bar)
            axs[i+1].legend(handles=legend_patches)
            axs[i+1].set_ylabel('State frequency (n={0})'.format(len(clus_seqs)))
            axs[i+1].set_title('Cluster {0}'.format(i))
        plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
        plt.suptitle(title)
        f.text(0.5, 0.04, 'Time (h)', ha='center')
        plt.savefig('/Users/dionisio/Desktop/cluster_hdbscan', bbox_inches='tight')
        # plt.title('Cluster {0}'.format(i))
        # plt.xlabel('Time')
        # plt.ylabel('Sequences')
        # plt.xlim(ind.min(), ind.max())
        # plt.ylim(0, 1)
        # plt.show()

    else:
        for i in set(cluster_labels):
            clus_seqs = all_sequences[cluster_labels == i]
            clus_sil_samples = sil_samples[cluster_labels == i]
            clus_sil_sort = np.argsort(clus_sil_samples)
            clus_seqs = clus_seqs[clus_sil_sort]
            xx = np.array(range(clus_seqs.shape[1] + 1))

            plt.figure()
            for idx, seq in enumerate(clus_seqs):
                y = np.array([idx]*(clus_seqs.shape[1] + 1))
                points = np.array([xx, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                lc.set_array(seq)
                lc.set_linewidth(10)
                plt.gca().add_collection(lc)

            # ax = plt.axes()
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.title('Cluster {0}'.format(i))
            plt.xlabel('Time')
            plt.ylabel('Sequences')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(0, len(clus_seqs))
            plt.show()

plot_sequences(unique_sequences[indices], hdb.labels_[indices], title='Clusters in mBid', type_plot='mode')
# plot_sequences(unique_sequences, pam1_labels)
# plot_sequences(unique_sequences, pam2_labels)


