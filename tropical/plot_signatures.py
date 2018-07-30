from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from tropical.util import rate_2_interactions, get_labels_entropy
import math
from scipy import stats # I could remove this dependence, mode implementation only depends on numpy
from future.utils import listvalues
from collections import OrderedDict
from tropical.distinct_colors import distinct_colors
import numpy as np
from sklearn import metrics


class PlotSequences(object):
    """
    Class to plot dynamic signatures sequences
    """

    def __init__(self, sequence_obj, no_clustering=False):
        """

        Parameters
        ----------
        sequence_obj
        """
        # TODO: find a way to make repeated sequences a big lane proportional to the number of repetitions
        self.unique = sequence_obj.unique
        self.sequences = sequence_obj.sequences
        self.unique_states = sequence_obj.unique_states
        self.diss = sequence_obj.diss
        colors = distinct_colors(len(self.unique_states))
        self.states_colors = OrderedDict((state, colors[x]) for x, state, in enumerate(self.unique_states))
        self.cmap, self.norm = self.cmap_norm()

        if no_clustering:
            self.cluster_labels = np.zeros(len(self.sequences), dtype=np.int)
        else:
            if sequence_obj.diss is None or sequence_obj.labels is None:
                raise Exception('Clustering has not been done in the ClusterSequence class')
            self.cluster_labels = sequence_obj.labels

    def cmap_norm(self):
        cmap = ListedColormap(listvalues(self.states_colors))
        bounds = list(self.states_colors)
        bounds.append(bounds[-1] + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm

    def legend_plot(self, model, reactions_comb, combs=None):
        """
        Creates a plot with the legend of the Colors and the proteins involved in interactions
        Parameters
        ----------
        model : PySB model
        reactions_comb : dict,
            A dictionary whose keys are the level of the combination and the values are
            dictionaries with labels as keys and reaction rates as values
        combs : vector-like
            Labels (integers) of the dominant combination rates

        Returns
        -------

        """
        fig_legend = plt.figure(100, figsize=(2, 1.25))
        comb_flat = {}
        #Flatten reactions_comb dict to search by label key
        for com in reactions_comb.values():
            comb_flat.update(com)

        legend_patches = []
        labels = []
        if combs:
            for l in combs:
                if l != -1:
                    label = " ".join(rate_2_interactions(model, str(rate)) for rate in comb_flat[l])
                    # Delete repeated proteins
                    words = label.split()
                    label = ", ".join(sorted(set(words), key=words.index))
                    label = '{}: {}'.format(l, label)

                    labels.append(label)
                    legend_patches.append(mpatches.Patch(color=self.states_colors[l], label=label))
        else:
            for l, c in self.states_colors.items():
                if l != -1:
                    label = " ".join(rate_2_interactions(model, str(rate)) for rate in comb_flat[l])
                    # Delete repeated proteins
                    words = label.split()
                    label=", ".join(sorted(set(words), key=words.index))
                    label = '{}: {}'.format(l, label)

                    labels.append(label)
                    legend_patches.append(mpatches.Patch(color=c, label=label))
        fig_legend.legend(legend_patches, labels, loc='center', frameon=False, ncol=4)
        plt.savefig('legends.png', format='png', bbox_inches='tight', dpi=1000)
        return

    def plot_sequences(self, type_fig='modal', title='', sort_seq=None):  # , legend_plot=False):
        clusters = set(self.cluster_labels)
        if -1 in clusters:
            clusters = list(clusters)[:-1]  # this is to not plot the signatures that can't be clustered :( from hdbscan
        else:
            clusters = list(clusters)
        n_rows = int(math.ceil(len(clusters) / 3))
        if len(clusters) == 1:
            f, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(8, 6))
            axs = [axs]
        elif len(clusters) == 2:
            f, axs = plt.subplots(n_rows, 2, sharex=True, figsize=(8, 6))
        else:
            f, axs = plt.subplots(n_rows, 3, sharex=True, figsize=(8, 6))
            f.subplots_adjust(hspace=.6, wspace=.4)
            axs = axs.reshape(n_rows * 3)

            plots_off = (n_rows * 3) - len(clusters)
            for i in range(1, plots_off + 1):
                axs[-i].axis('off')

        if type_fig == 'modal':
            self.__modal(clusters, axs)
            plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
            plt.suptitle(title)
            f.text(0.5, 0.04, 'Time (h)', ha='center')
            plt.savefig('cluster_modal_' + title + '.pdf', bbox_inches='tight', format='pdf')

        elif type_fig == 'trajectories':
            self.__trajectories(clusters, axs, sort_seq)
            plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
            plt.suptitle(title)
            f.text(0.5, 0.04, 'Time (h)', ha='center')
            plt.savefig('cluster_all_tr_' + title + '.pdf', bbox_inches='tight', format='pdf')

        elif type_fig == 'entropy':
            self.__entropy(clusters, axs)
            plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
            plt.suptitle(title)
            f.text(0.5, 0.04, 'Time (h)', ha='center')
            plt.savefig('entropy_' + title + '.pdf', bbox_inches='tight', format='pdf')

        else:
            raise NotImplementedError('Type of visualization not implements')

        return

    def __modal(self, clusters, axs):
        for clus in clusters:  # if we start from 1 it won't plot the sets not clustered
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
            colors = [self.states_colors[c] for c in modal_states[0]]
            legend_patches = [mpatches.Patch(color=self.states_colors[c], label=c) for c in set(modal_states[0])]
            axs[clus].bar(self.sequences.columns.tolist(), mc_norm, color=colors, width=width_bar)
            axs[clus].legend(handles=legend_patches, fontsize='x-small')
            axs[clus].set_ylabel('frequency (n={0})'.format(total_seqs), fontsize='small')
            axs[clus].set_title('Cluster {0}'.format(clus))
        return

    def __trajectories(self, clusters, axs, sort_seq=None):
        # TODO search for other types of sorting
        if sort_seq == 'silhouette':
            sort_values = metrics.silhouette_samples(X=self.diss, labels=self.cluster_labels, metric='precomputed')
        else:
            sort_values = np.random.rand(len(self.cluster_labels))

        for clus in clusters:  # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.sequences.iloc[self.cluster_labels == clus]
            n_seqs = clus_seqs.shape[0]
            if self.unique:
                total_seqs = 0
                for seq in clus_seqs.index.values:
                    total_seqs += seq[1]
            else:
                total_seqs = n_seqs

            clus_sort_samples = sort_values[self.cluster_labels == clus]
            clus_sil_sort = np.argsort(clus_sort_samples)
            clus_seqs = clus_seqs.iloc[clus_sil_sort]
            xx = self.sequences.columns
            count_seqs = 0

            for seq in clus_seqs.itertuples(index=False):
                y = np.array([count_seqs] * (clus_seqs.shape[1]))
                points = np.array([xx, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=self.cmap, norm=self.norm)
                lc.set_array(np.array(seq))
                lc.set_linewidth(10)
                axs[clus].add_collection(lc)
                axs[clus].set_ylabel('Trajectories (n={0})'.format(total_seqs), fontsize='small')
                axs[clus].set_ylim(0, len(clus_seqs))
                axs[clus].set_xlim(xx.min(), xx.max())
                axs[clus].set_title('Cluster {0}'.format(clus))
                count_seqs += 1
        return

    def __entropy(self, clusters, axs):
        for clus in clusters:  # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.sequences.iloc[self.cluster_labels == clus]
            n_seqs = clus_seqs.shape[0]
            if self.unique:
                total_seqs = 0
                for seq in clus_seqs.index.values:
                    total_seqs += seq[1]
            else:
                total_seqs = n_seqs

            time_points = clus_seqs.shape[1]
            entropies = [0] * time_points

            for col_idx, col_t in enumerate(clus_seqs):
                entropy = get_labels_entropy(clus_seqs[col_t].values)
                entropies[col_idx] = entropy

            axs[clus].plot(range(time_points), entropies)
            axs[clus].set_ylabel('Entropy', fontsize='small')
            axs[clus].set_title('Cluster {0}'.format(clus))
        return