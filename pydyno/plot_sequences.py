import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats
from sklearn import metrics
from pydyno.util import get_labels_entropy
import os

# TODO there must be a better way to define the fontsize
N_ROW_FONTSIZE = {1: 'medium', 2: 'medium', 3: 'small', 4: 'small', 5: 'x-small', 6: 'x-small', 7: 'xx-small',
                  8: 'xx-small', 9: 'xx-small'}


class PlotSequences:
    """
    Visualize discretized sequences
    """
    def __init__(self, seq_analysis):
        self._seq_analysis = seq_analysis

    @property
    def seq_analysis(self):
        return self._seq_analysis

    @seq_analysis.setter
    def seq_analysis(self, new_seq):
        self._seq_analysis = new_seq

    def plot_sequences(self, type_fig='modal', plot_all=False, title='', dir_path='', sort_seq=None):
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
        plot_all: bool
            If true the function plots all the sequences, otherwise it plots the clustered
            sequences in different subplots
        title: str
            Title of the figure
        dir_path: str
            Path to directory where the plots are going to be saved. File names are
            assigned automatically
        sort_seq: str
            Method to sort sequences for a plot. Valid values are: `silhouette`.
             It is only available when the type of plot is `trajectories`

        Returns
        -------

        """
        if plot_all:
            cluster_labels = np.zeros(len(self.seq_analysis.sequences), dtype=np.int)
        else:
            # Check that the sequences has been clustered
            if self.seq_analysis.labels is None:
                raise Exception('Cluster the sequences first')
            cluster_labels = self.seq_analysis.labels

        clusters = set(cluster_labels)
        if -1 in clusters:
            clusters = list(clusters)[:-1]  # this is to not plot the signatures that can't be clustered :( from hdbscan
        else:
            clusters = list(clusters)
        n_rows = int(math.ceil(len(clusters) / 3.))
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
            self.__modal(cluster_labels, clusters, axs, n_rows)

        elif type_fig == 'trajectories':
            self.__trajectories(cluster_labels, clusters, axs, n_rows, sort_seq)

        elif type_fig == 'entropy':
            self.__entropy(cluster_labels, clusters, f, axs, n_rows)

        else:
            raise NotImplementedError('Type of visualization not implemented')

        plt.setp([a.get_xticklabels() for a in f.axes[:-3]], visible=False)
        plt.suptitle(title)
        f.text(0.5, 0.04, 'Time', ha='center')
        final_path = os.path.join(dir_path, type_fig + '.png')
        plt.savefig(final_path, bbox_inches='tight', format='png')
        # plt.close('all')

        return

    def __modal(self, cluster_labels, clusters, axs, nrows):
        color_label_map = {}
        for clus in clusters:  # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.seq_analysis.sequences.iloc[cluster_labels == clus]
            n_seqs = clus_seqs.shape[0]
            total_seqs = 0
            for seq in clus_seqs.index.values:
                total_seqs += seq[1]

            modal_states, mode_counts = stats.mode(clus_seqs, axis=0)
            mc_norm = np.divide(mode_counts[0], n_seqs, dtype=np.float)
            width_bar = self.seq_analysis.sequences.columns[1] - self.seq_analysis.sequences.columns[0]
            colors = [self.seq_analysis.states_colors[c] for c in modal_states[0]]
            color_label_map.update({self.seq_analysis.states_colors[c]:c for c in set(modal_states[0])})
            axs[clus].set_ylim(0, 1)
            axs[clus].bar(self.seq_analysis.sequences.columns.tolist(), mc_norm, color=colors, width=width_bar)
            axs[clus].set_ylabel('subnetwork freq (n={0})'.format(total_seqs), fontsize=N_ROW_FONTSIZE[nrows])  # Frequency
            axs[clus].set_title('Cluster {0}'.format(clus), fontsize=N_ROW_FONTSIZE[nrows])

        return

    def __trajectories(self, cluster_labels, clusters, axs, nrows, sort_seq=None):
        # TODO search for other types of sorting
        if sort_seq == 'silhouette':
            sort_values = metrics.silhouette_samples(X=self.seq_analysis.diss, labels=cluster_labels, metric='precomputed')
        else:
            sort_values = np.random.rand(len(cluster_labels))

        for clus_idx, clus in enumerate(clusters):  # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.seq_analysis.sequences.iloc[cluster_labels == clus]
            total_seqs = 0
            for seq in clus_seqs.index.values:
                total_seqs += seq[1]

            clus_sort_samples = sort_values[cluster_labels == clus]
            clus_sil_sort = np.argsort(clus_sort_samples)
            clus_seqs = clus_seqs.iloc[clus_sil_sort]
            xx = self.seq_analysis.sequences.columns
            count_seqs = 0

            for seq in clus_seqs.itertuples(index=False):
                y = np.array([count_seqs] * (clus_seqs.shape[1]))
                points = np.array([xx, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=self.seq_analysis.cmap, norm=self.seq_analysis.norm)
                lc.set_array(np.array(seq))
                lc.set_linewidth(10)
                axs[clus_idx].add_collection(lc)
                axs[clus_idx].set_ylabel('Seq (n={0})'.format(total_seqs), fontsize=N_ROW_FONTSIZE[nrows])  # Sequences
                axs[clus_idx].set_ylim(0, len(clus_seqs))
                axs[clus_idx].set_xlim(xx.min(), xx.max())
                axs[clus_idx].set_title('Cluster {0}'.format(clus), fontsize=N_ROW_FONTSIZE[nrows])
                count_seqs += 1
        return

    def __entropy(self, cluster_labels, clusters, fig, axs, nrows):
        max_entropy = 0
        for clus in clusters:  # if we start from 1 it won't plot the sets not clustered
            clus_seqs = self.seq_analysis.sequences.iloc[cluster_labels == clus]
            total_seqs = 0
            for seq in clus_seqs.index.values:
                total_seqs += seq[1]

            time_points = clus_seqs.shape[1]
            entropies = [0] * time_points

            for col_idx, col_t in enumerate(clus_seqs):
                entropy = get_labels_entropy(clus_seqs[col_t].values)
                entropies[col_idx] = entropy

            if max(entropies) > max_entropy:
                max_entropy = max(entropies)

            axs[clus].plot(range(time_points), entropies)
            # axs[clus].set_ylabel('Entropy', fontsize='small')
            axs[clus].set_title('Cluster {0}'.format(clus), fontsize=N_ROW_FONTSIZE[nrows])

        for clus in clusters:
            axs[clus].set_ylim(0, max_entropy)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        return
