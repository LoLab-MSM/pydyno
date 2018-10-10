from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from tropical.util import get_labels_entropy
import math
from scipy import stats  # I could remove this dependence, mode implementation only depends on numpy
from future.utils import listvalues
from collections import OrderedDict
from tropical.distinct_colors import distinct_colors
import numpy as np
from sklearn import metrics

n_row_fontsize = {1: 'medium', 2: 'medium', 3: 'small', 4: 'small', 5: 'x-small', 6: 'x-small', 7: 'xx-small',
                  8: 'xx-small', 9: 'xx-small'}


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
        if len(self.unique_states) <= 128:
            colors = distinct_colors(len(self.unique_states))
        else:
            import seaborn as sns
            colors = sns.color_palette('hls', len(self.unique_states))
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

    def plot_sequences(self, type_fig='modal', title='', filename='', sort_seq=None):  # , legend_plot=False):
        """
        Function to plot three different figures of the sequences.
        The modal figure takes the mode state at each time and plots
        the percentage of that state compated to all the other states.

        The trajectories figure plots each of the sequences.

        The entropy figure plots the entropy calculated from the
        percentages of each of the states at each time relative
        to the total.

        Parameters
        ----------
        type_fig: str
            Type of figure to plot. Valid values are: `modal`, `trajectories`, `entropy`
        title: str
            Title of the figure
        filename: str
            Name of file
        sort_seq: str
            Method to sort sequences for a plot. Valid values are: `silhouette`.
             It is only available when the type of plot is `trajectories`

        Returns
        -------

        """
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
            self.__modal(clusters, axs, n_rows)
            plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
            plt.suptitle(title)
            f.text(0.5, 0.04, 'Time (h)', ha='center')
            plt.savefig(filename + 'cluster_modal' + '.pdf', bbox_inches='tight', format='pdf')

        elif type_fig == 'trajectories':
            self.__trajectories(clusters, axs, n_rows, sort_seq)
            plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
            plt.suptitle(title)
            f.text(0.5, 0.04, 'Time (h)', ha='center')
            plt.savefig(filename + 'cluster_all_tr' + '.pdf', bbox_inches='tight', format='pdf')

        elif type_fig == 'entropy':
            self.__entropy(clusters, f, axs, n_rows)
            plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
            plt.suptitle(title)
            # f.text(0.5, 0.04, 'Time (h)', ha='center')
            plt.savefig(filename + 'entropy' + '.pdf', bbox_inches='tight', format='pdf')

        else:
            raise NotImplementedError('Type of visualization not implemented')

        return

    def __modal(self, clusters, axs, nrows):
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
            axs[clus].set_ylabel('Freq (n={0})'.format(total_seqs), fontsize=n_row_fontsize[nrows])  # Frequency
            axs[clus].set_title('Cluster {0}'.format(clus), fontsize=n_row_fontsize[nrows])
        return

    def __trajectories(self, clusters, axs, nrows, sort_seq=None):
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
                axs[clus].set_ylabel('Seq (n={0})'.format(total_seqs), fontsize=n_row_fontsize[nrows])  # Sequences
                axs[clus].set_ylim(0, len(clus_seqs))
                axs[clus].set_xlim(xx.min(), xx.max())
                axs[clus].set_title('Cluster {0}'.format(clus), fontsize=n_row_fontsize[nrows])
                count_seqs += 1
        return

    def __entropy(self, clusters, fig, axs, nrows):
        max_entropy = 0
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

            if max(entropies) > max_entropy:
                max_entropy = max(entropies)

            axs[clus].plot(range(time_points), entropies)
            # axs[clus].set_ylabel('Entropy', fontsize='small')
            axs[clus].set_title('Cluster {0}'.format(clus), fontsize=n_row_fontsize[nrows])

        for clus in clusters:
            axs[clus].set_ylim(0, max_entropy)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        return