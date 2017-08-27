from __future__ import division
import os
import pandas as pd
import numpy as np
import mlpy
from sklearn.metrics.pairwise import pairwise_distances, distance_metrics
import hdbscan
from pam_clustering import kMedoids
import matplotlib.pyplot as plt
from distinct_colors import distinct_colors
from collections import OrderedDict
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
from scipy import stats
import matplotlib.patches as mpatches
from sklearn import metrics
from matplotlib.collections import LineCollection


class ClusterSequences(object):
    """
    Class to cluster tropical signatures
    """
    def __init__(self, data, unique_sequences=True, truncate_seq=None):
        if os.path.isfile(data):
            data_seqs = pd.read_csv(data, header=0, index_col=0)
            # convert column names into float numbers
            data_seqs.columns = [float(i) for i in data_seqs.columns.tolist()]
        elif type(data) is np.ndarray:
            data_seqs = data
            data_seqs = pd.DataFrame(data=data_seqs)
        else:
            raise Exception('data not valid')
        if isinstance(truncate_seq, int):
            data_seqs = data_seqs[data_seqs.columns.tolist()[:truncate_seq]]
        if unique_sequences:
            data_seqs = data_seqs.groupby(data_seqs.columns.tolist()).size().rename('count').reset_index()
            data_seqs.set_index([range(len(data_seqs)), 'count'], inplace=True)
            self.sequences = data_seqs
            self.unique = True
        else:
            self.sequences = data_seqs
            self.unique = False

        # States in sequences
        self.unique_states = pd.unique(data_seqs[data_seqs.columns.tolist()].values.ravel())

        self.diss = None
        self.labels = None

    @staticmethod
    def lcs_dist_diff_length(seq1, seq2):
        d_1_2 = mlpy.lcs_std(seq1, seq1)[0] + mlpy.lcs_std(seq2, seq2)[0] - 2*mlpy.lcs_std(seq1, seq2)[0]
        return d_1_2

    @staticmethod
    def lcs_dist_same_length(seq1, seq2):
        seq_len = len(seq1)
        d_1_2 = 2 * seq_len - 2 * mlpy.lcs_std(seq1, seq2)[0]
        return d_1_2

    def diss_matrix(self, metric='LCS', n_jobs=1):
        # TODO check if ndarray have sequences of different lengths
        if metric in distance_metrics().keys():
            diss = pairwise_distances(self.sequences.values, metric=metric, n_jobs=n_jobs)
        elif metric == 'LCS':
            diss = pairwise_distances(self.sequences.values, metric=self.lcs_dist_same_length, n_jobs=n_jobs)
        elif callable(metric):
            diss = pairwise_distances(self.sequences.values, metric=metric, n_jobs=n_jobs)
        else:
            raise Exception('metric not accepted')
        self.diss = diss

    def hdbscan(self, min_cluster_size=50, min_samples=5):

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed').fit(self.diss)
        self.labels = hdb.labels_
        return hdb.labels_

    # def plot_sequences(self, metric='LCS', min_cluster_size=50, min_samples=5, n_jobs=1):
    #     if self.diss is None:
    #         self.diss = self.diss_matrix(metric=metric, n_jobs=n_jobs)
    #
    #     self.hdbscan(min_cluster_size=min_cluster_size, min_samples=min_samples)



class PlotSequences(object):
    """
    Class to plot sequences
    """
    def __init__(self, sequence_obj):
        """

        :param sequence_obj ClusterSequence object
        """
        self.sequences = sequence_obj.sequences
        self.cluster_labels = sequence_obj.labels
        self.unique_states = sequence_obj.unique_states
        self.colors = distinct_colors(len(self.unique_states))
        self.cmap, self.norm, self.states_color_dict = self.cmap_norm()
        self.diss = sequence_obj.diss

    def cmap_norm(self):
        states_color_map = OrderedDict((state, self.colors[x]) for x, state, in enumerate(self.unique_states))
        cmap = ListedColormap(states_color_map.values())
        bounds = states_color_map.keys()
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm, states_color_map

    def modal_plot(self, title=''):
        clusters = set(self.cluster_labels)
        n_rows = int(math.ceil(len(clusters)/3))
        f, axs = plt.subplots(n_rows, 3, sharex=True, sharey=True, figsize=(8, 6))
        f.subplots_adjust(hspace=.5)
        axs = axs.reshape(n_rows * 3)

        plots_off = (n_rows * 3) - len(clusters)
        for i in range(1, plots_off+1):
            axs[-i].axis('off')

        for clus in clusters:
            clus_seqs = self.sequences.iloc[self.cluster_labels == clus]
            n_seqs = clus_seqs.shape[0]
            modal_states, mode_counts = stats.mode(clus_seqs, axis=0)
            mc_norm = np.divide(mode_counts[0], n_seqs, dtype=np.float)
            width_bar = self.sequences.columns[1] - self.sequences.columns[0]
            colors = [self.states_color_dict[c] for c in modal_states[0]]
            legend_patches = [mpatches.Patch(color=self.states_color_dict[c], label=c) for c in set(modal_states[0])]
            axs[clus + 1].bar(self.sequences.columns.tolist(), mc_norm, color=colors, width=width_bar)
            axs[clus + 1].legend(handles=legend_patches, fontsize='x-small')
            axs[clus + 1].set_ylabel('State frequency (n={0})'.format(len(clus_seqs)), fontsize='x-small')
            axs[clus + 1].set_title('Cluster {0}'.format(clus))
        plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
        plt.suptitle(title)
        f.text(0.5, 0.04, 'Time (h)', ha='center')
        plt.savefig('cluster_hdbscan_modal', bbox_inches='tight')
        return

    def all_trajectories_plot(self, title='', sort_seq='silhouette'):
        clusters = set(self.cluster_labels)
        n_rows = int(math.ceil(len(clusters)/3))
        f, axs = plt.subplots(n_rows, 3, sharex=True, sharey=True)
        axs = axs.reshape(n_rows * 3)

        plots_off = (n_rows * 3) - len(clusters)
        for i in range(plots_off):
            axs[-i].axis('off')

        # TODO search for other types of sorting
        if sort_seq == 'silhouette':
            sil_samples = metrics.silhouette_samples(X=self.diss, labels=self.cluster_labels, metric='precomputed')

        for i in clusters:
            clus_seqs = self.sequences.iloc[self.cluster_labels == i]
            clus_sil_samples = sil_samples[self.cluster_labels == i]
            clus_sil_sort = np.argsort(clus_sil_samples)
            clus_seqs = clus_seqs[clus_sil_sort]
            xx = np.array(range(clus_seqs.shape[1] + 1))

            for idx, seq in enumerate(clus_seqs):
                y = np.array([idx]*(clus_seqs.shape[1] + 1))
                points = np.array([xx, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=self.cmap, norm=self.norm)
                lc.set_array(seq)
                lc.set_linewidth(10)
                axs[i + 1].add_collection(lc)
                axs[i + 1].set_ylabel('Trajectories (n={0})'.format(len(clus_seqs)))
                axs[i + 1].set_title('Cluster {0}'.format(i))
        plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
        plt.suptitle(title)
        f.text(0.5, 0.04, 'Time (h)', ha='center')
        plt.savefig('cluster_hdbscan_all_tr', bbox_inches='tight')








