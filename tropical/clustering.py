from __future__ import division
import matplotlib
matplotlib.use('TkAgg')
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, distance_metrics
import sklearn.cluster as cluster
import hdbscan
import matplotlib.pyplot as plt
from distinct_colors import distinct_colors
from collections import OrderedDict
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
from scipy import stats
import matplotlib.patches as mpatches
from sklearn import metrics
from matplotlib.collections import LineCollection
import editdistance
import tropical.lcs as lcs
import collections
from kmedoids import kMedoids
# def lcs_length(a, b):
#     table = [[0] * (len(b) + 1) for _ in xrange(len(a) + 1)]
#     for i, ca in enumerate(a, 1):
#         for j, cb in enumerate(b, 1):
#             table[i][j] = (
#                 table[i - 1][j - 1] + 1 if ca == cb else
#                 max(table[i][j - 1], table[i - 1][j]))
#     return table[-1][-1]


class ClusterSequences(object):
    """
    Class to cluster DynSign signatures

    Parameters
    ----------
    data: str file or np.ndarray
        file or ndarray where rows are DynSign signatures and columns are dominant states at specific
        time points
    unique_sequences: bool, optional
        Drop repeated sequences
    truncate_seq: int
        Index of where to truncate a sequence
    """
    def __init__(self, data, unique_sequences=True, truncate_seq=None):

        if isinstance(data, str):
            if os.path.isfile(data):
                data_seqs = pd.read_csv(data, header=0, index_col=0)
                # convert column names into float numbers
                data_seqs.columns = [float(i) for i in data_seqs.columns.tolist()]
        elif isinstance(data, np.ndarray):
            data_seqs = data
            data_seqs = pd.DataFrame(data=data_seqs)
        else:
            raise TypeError('data type not valid')

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
        unique_states = pd.unique(data_seqs[data_seqs.columns.tolist()].values.ravel())
        unique_states.sort()
        self.unique_states = unique_states

        self.diss = None
        self.labels = None
        self.cluster_method = ''

    @staticmethod
    def lcs_dist_diff_length(seq1, seq2):
        d_1_2 = lcs.lcs_std(seq1, seq1)[0] + lcs.lcs_std(seq2, seq2)[0] - 2*lcs.lcs_std(seq1, seq2)[0]
        return d_1_2

    @staticmethod
    def lcs_dist_same_length(seq1, seq2):
        seq_len = len(seq1)
        d_1_2 = 2 * seq_len - 2 * lcs.lcs_std(seq1, seq2)[0]
        return d_1_2

    @staticmethod
    def levenshtein(seq1, seq2):
        d_1_2 = editdistance.eval(seq1, seq2).__float__()
        return d_1_2

    def diss_matrix(self, metric='LCS', n_jobs=1):
        # TODO check if ndarray have sequences of different lengths
        if metric in hdbscan.dist_metrics.METRIC_MAPPING.keys():
            diss = pairwise_distances(self.sequences.values, metric=metric, n_jobs=n_jobs)
        elif metric == 'LCS':
            diss = pairwise_distances(self.sequences.values, metric=self.lcs_dist_same_length, n_jobs=n_jobs)
        elif metric == 'levenshtein':
            diss = pairwise_distances(self.sequences.values, metric=self.levenshtein, n_jobs=n_jobs)
        elif callable(metric):
            diss = pairwise_distances(self.sequences.values, metric=metric, n_jobs=n_jobs)
        else:
            raise ValueError('metric not accepted')
        self.diss = diss

    def hdbscan(self, min_cluster_size=50, min_samples=5, alpha=1.0, cluster_selection_method='eom'):

        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, alpha=alpha,
                              cluster_selection_method=cluster_selection_method, metric='precomputed').fit(self.diss)
        self.labels = hdb.labels_
        self.cluster_method = 'hdbscan'
        return

    def Kmedoids(self, n_clusters):
        kmedoids = kMedoids(self.diss, n_clusters)
        labels = np.empty(len(self.sequences), dtype=np.int32)
        for lb, seq_idx in kmedoids[1].items():
            labels[seq_idx] = lb
        self.cluster_method = 'kmedoids'
        self.labels = labels
        return

    def Kmeans(self, n_clusters, **kwargs):
        kmeans = cluster.KMeans(n_clusters=n_clusters, **kwargs).fit(self.diss)
        self.labels = kmeans.labels_
        self.cluster_method = 'kmeans'
        return

    def silhouette_score(self):
        if self.labels is None:
            raise Exception('you must cluster the signatures first')
        score = metrics.silhouette_score(self.diss, self.labels, metric='precomputed')
        return score

    def calinski_harabaz_score(self):
        if self.labels is None:
            raise Exception('you must cluster the signatures first')
        score = metrics.calinski_harabaz_score(self.sequences, self.labels)
        return score

    def elbow_plot(self, cluster_range):
        if self.cluster_method not in ['kmeans']:
            raise ValueError('Analysis not valid for {0}'.format(self.cluster_method))
        if isinstance(cluster_range, int):
            cluster_range = range(1, cluster_range)
        elif isinstance(cluster_range, collections.Iterable):
            pass
        else:
            raise TypeError('Type not valid')
        cluster_errors = []
        for num_clusters in cluster_range:
            clusters = cluster.KMeans(num_clusters).fit(self.diss)
            cluster_errors.append(clusters.inertia_)
        clusters_df = pd.DataFrame({'num_clusters':cluster_range, 'cluster_errors': cluster_errors})
        plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker='o')
        plt.savefig('elbow_analysis.png', format='png')


class PlotSequences(object):
    """
    Class to plot dynamic signatures sequences
    """
    def __init__(self, sequence_obj):
        """

        Parameters
        ----------
        sequence_obj
        """
        if sequence_obj.diss is None or sequence_obj.labels is None:
            raise Exception('Clustering has not been done in the ClusterSequence class')

        self.unique = sequence_obj.unique
        self.sequences = sequence_obj.sequences
        self.diss = sequence_obj.diss
        self.cluster_labels = sequence_obj.labels
        self.unique_states = sequence_obj.unique_states
        self.colors = distinct_colors(len(self.unique_states))
        self.cmap, self.norm, self.states_color_dict = self.cmap_norm()

    def cmap_norm(self):
        states_color_map = OrderedDict((state, self.colors[x]) for x, state, in enumerate(self.unique_states))
        cmap = ListedColormap(states_color_map.values())
        bounds = states_color_map.keys()
        bounds.append(bounds[-1]+1)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm, states_color_map

    def modal_plot(self, title='', legend_plot=False):
        clusters = set(self.cluster_labels)
        if -1 in clusters:
            clusters = list(clusters)[:-1] # this is to not plot the signatures that can't be clustered :(
        else:
            clusters = list(clusters)
        n_rows = int(math.ceil(len(clusters)/3))
        f, axs = plt.subplots(n_rows, 3, sharex=True, sharey=True, figsize=(8, 6))
        f.subplots_adjust(hspace=.5)
        axs = axs.reshape(n_rows * 3)

        # if legend_plot:
        #     fig_legend = plt.figure(100, figsize=(2, 1.25))
        #     legend_patches = [mpatches.Patch(color=c, label=l) for l, c in self.states_color_dict.items()]
        #     fig_legend.legend(legend_patches, self.states_color_dict.keys(), loc='center', frameon=False, ncol=4)
        #     plt.savefig('legends.png', format='png', bbox_inches='tight', dpi=1000)

        plots_off = (n_rows * 3) - len(clusters)
        for i in range(1, plots_off+1):
            axs[-i].axis('off')

        for clus in clusters: # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.sequences.iloc[self.cluster_labels == clus]
            n_seqs = clus_seqs.shape[0]
            if self.unique:
                total_seqs = 0
                for seq in clus_seqs.index.values:
                    total_seqs += seq[1]
            else:
                total_seqs = n_seqs

            modal_states, mode_counts = stats.mode(clus_seqs, axis=0)
            mc_norm = np.divide(mode_counts[0], n_seqs, dtype=np.float)
            width_bar = self.sequences.columns[1] - self.sequences.columns[0]
            colors = [self.states_color_dict[c] for c in modal_states[0]]
            legend_patches = [mpatches.Patch(color=self.states_color_dict[c], label=c) for c in set(modal_states[0])]
            axs[clus].bar(self.sequences.columns.tolist(), mc_norm, color=colors, width=width_bar)
            axs[clus].legend(handles=legend_patches, fontsize='x-small')
            axs[clus].set_ylabel('frequency (n={0})'.format(total_seqs), fontsize='x-small')
            axs[clus].set_title('Cluster {0}'.format(clus))
        plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
        plt.suptitle(title)
        f.text(0.5, 0.04, 'Time (h)', ha='center')
        plt.savefig('cluster_hdbscan_modal', bbox_inches='tight', dpi=1000)
        return

    def all_trajectories_plot(self, title='', sort_seq='silhouette'):
        clusters = set(self.cluster_labels)
        if -1 in clusters:
            clusters = list(clusters)[:-1]  # this is to not plot the signatures that can't be clustered :(
        else:
            clusters = list(clusters)
        n_rows = int(math.ceil(len(clusters)/3))
        f, axs = plt.subplots(n_rows, 3, sharex=True, figsize=(8, 6))
        f.subplots_adjust(hspace=.6, wspace=.4)
        axs = axs.reshape(n_rows * 3)

        plots_off = (n_rows * 3) - len(clusters)
        for i in range(1, plots_off+1):
            axs[-i].axis('off')

        # TODO search for other types of sorting
        if sort_seq == 'silhouette':
            sil_samples = metrics.silhouette_samples(X=self.diss, labels=self.cluster_labels, metric='precomputed')

        for clus in clusters: # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.sequences.iloc[self.cluster_labels == clus]
            n_seqs = clus_seqs.shape[0]
            if self.unique:
                total_seqs = 0
                for seq in clus_seqs.index.values:
                    total_seqs += seq[1]
            else:
                total_seqs = n_seqs

            clus_sil_samples = sil_samples[self.cluster_labels == clus]
            clus_sil_sort = np.argsort(clus_sil_samples)
            clus_seqs = clus_seqs.iloc[clus_sil_sort]
            xx = self.sequences.columns
            count_seqs = 0

            for index, seq in clus_seqs.iterrows():
                y = np.array([count_seqs]*(clus_seqs.shape[1]))
                points = np.array([xx, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=self.cmap, norm=self.norm)
                lc.set_array(seq.values)
                lc.set_linewidth(10)
                axs[clus].add_collection(lc)
                axs[clus].set_ylabel('Trajectories (n={0})'.format(total_seqs), fontsize='xx-small')
                axs[clus].set_ylim(0, len(clus_seqs))
                axs[clus].set_xlim(xx.min(), xx.max())
                axs[clus].set_title('Cluster {0}'.format(clus))
                count_seqs += 1
        plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
        plt.suptitle(title)
        f.text(0.5, 0.04, 'Time (h)', ha='center')
        plt.savefig('cluster_hdbscan_all_tr', bbox_inches='tight', dpi=500)
