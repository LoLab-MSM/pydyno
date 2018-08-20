from tropical.clustering import ClusterSequences
from tropical.discretize import Discretize
from plot_signatures import PlotSequences
from cluster_analysis import AnalysisCluster


class Pydrone(object):
    """
    Class to perform sequence analysis on systems biology models
    """
    def __init__(self, model, simulations, diff_par):
        self._model = model
        self.simulations = simulations
        self.sequences = None
        self.sp_analyzed = None
        self.sp_cluster_sequences = None
        self._diff_par = diff_par

    @property
    def model(self):
        return self._model

    @property
    def diff_par(self):
        return self._diff_par

    @diff_par.setter
    def diff_par(self, new_diff_par):
        self._diff_par = new_diff_par

    def discretize(self, cpu_cores=1):
        seqs = Discretize(self.model, self.simulations, self.diff_par)
        self.sequences = seqs.get_signatures(cpu_cores)
        return self

    def cluster_signatures_agglomerative(self, species, nclusters, metric='LCS',
                                         cluster_range=False, linkage='average',
                                         unique_sequences=False, truncate_seq=None,
                                         cpu_cores=1, **kwargs):
        cs = ClusterSequences(self.sequences.loc[species], unique_sequences, truncate_seq)
        cs.diss_matrix(metric=metric, n_jobs=cpu_cores)
        if cluster_range:
            sil_df = cs.silhouette_score_agglomerative_range(nclusters, linkage, n_jobs=cpu_cores, **kwargs)
            nclusters = sil_df.loc[sil_df['cluster_silhouette'].idxmax()].num_clusters
        cs.agglomerative_clustering(nclusters, linkage, **kwargs)
        self.sp_cluster_sequences = cs
        self.sp_analyzed = species
        return self

    def cluster_signatures_spectral(self, species, nclusters, metric='LCS',
                                    cluster_range=False, unique_sequences=False,
                                    truncate_seq=None, cpu_cores=1, **kwargs):
        cs = ClusterSequences(self.sequences.loc[species], unique_sequences, truncate_seq)
        cs.diss_matrix(metric=metric, n_jobs=cpu_cores)
        if cluster_range:
            sil_df = cs.silhouette_score_spectral_range(nclusters, n_jobs=cpu_cores, **kwargs)
            nclusters = sil_df.loc[sil_df['cluster_silhouette'].idxmax()].num_clusters
        cs.spectral_clustering(nclusters, n_jobs=cpu_cores, **kwargs)
        self.sp_cluster_sequences = cs
        return self

    @property
    def plot_signatures(self):
        if self.sp_cluster_sequences is None:
            raise ValueError('No clustering has been done')
        ps = PlotSequences(self.sp_cluster_sequences)
        return ps

    @property
    def analysis_cluster(self):
        if self.sp_cluster_sequences is None:
            raise ValueError('No clustering has been done')
        labels = self.sp_cluster_sequences.labels
        ac = AnalysisCluster(self.model, self.simulations, labels)
        return ac
