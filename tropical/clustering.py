import pandas as pd
import numpy as np
import sklearn.cluster as cluster
from sklearn import metrics
from tropical.kmedoids import kMedoids
from tropical.sequence_analysis import Sequences

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None


class ClusterSequences(object):
    """
    Class to cluster DynSign signatures

    Parameters
    ----------
    seqdata: Sequences
        Sequences obtained using the sequence_analysis module
    """

    def __init__(self, seqdata):
        # Checking seqdata
        if isinstance(seqdata, Sequences):
            self.signatures = seqdata
        else:
            raise TypeError('seqdata must be an instance of the Sequences class')

        self.labels = None
        self.cluster_method = ''

    def __repr__(self):
        return ('{} (Sequences:{}, Unique States:{})'.format(self.__class__.__name__,
                                                             self.signatures, self.cluster_method))

    def hdbscan_clustering(self, min_cluster_size=50, min_samples=5,
                           alpha=1.0, cluster_selection_method='eom', **kwargs):
        """

        Parameters
        ----------
        min_cluster_size
        min_samples
        alpha
        cluster_selection_method
        kwargs

        Returns
        -------

        """
        if hdbscan is None:
            raise Exception('Please install the hdbscan package for this feature')
        if self.signatures.diss is None:
            raise Exception('Get the dissimilarity matrix first')
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                              alpha=alpha, cluster_selection_method=cluster_selection_method,
                              metric='precomputed', **kwargs).fit(self.signatures.diss)
        self.labels = hdb.labels_

        self.cluster_method = 'hdbscan'
        return

    def Kmedoids(self, n_clusters):
        """

        Parameters
        ----------
        n_clusters : int
            Number of clusters

        Returns
        -------

        """
        if self.signatures.diss is None:
            raise Exception('Get the dissimilarity matrix first')
        kmedoids = kMedoids(self.signatures.diss, n_clusters)
        labels = np.empty(len(self.signatures.sequences), dtype=np.int32)
        for lb, seq_idx in kmedoids[1].items():
            labels[seq_idx] = lb
        self.cluster_method = 'kmedoids'
        self.labels = labels
        return

    def agglomerative_clustering(self, n_clusters, linkage='average', **kwargs):
        ac = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                             linkage=linkage, **kwargs).fit(self.signatures.diss)
        self.labels = ac.labels_
        self.cluster_method = 'agglomerative'
        return

    def spectral_clustering(self, n_clusters, random_state=None, n_jobs=1, **kwargs):
        gamma = 1. / len(self.signatures.diss[0])
        kernel = np.exp(-self.signatures.diss * gamma)
        sc = cluster.SpectralClustering(n_clusters=n_clusters, random_state=random_state,
                                        affinity='precomputed', n_jobs=n_jobs, **kwargs).fit(kernel)
        self.labels = sc.labels_
        self.cluster_method = 'spectral'

    def silhouette_score(self):
        """

        Returns : Silhouette score to measure quality of the clustering
        -------

        """
        if self.labels is None:
            raise Exception('you must cluster the signatures first')
        if self.cluster_method == 'hdbscan':
            # Keep only clustered sequences
            clustered = np.where(self.labels != -1)[0]
            updated_labels = self.labels[clustered]
            updated_diss = self.signatures.diss[clustered][:, clustered]
            score = metrics.silhouette_score(updated_diss, updated_labels, metric='precomputed')
            return score
        else:
            score = metrics.silhouette_score(self.signatures.diss, self.labels, metric='precomputed')
            return score

    def silhouette_score_spectral_range(self, cluster_range, n_jobs=1, random_state=None, **kwargs):
        if isinstance(cluster_range, int):
            cluster_range = list(range(2, cluster_range + 1))  # +1 to cluster up to cluster_range
        elif hasattr(cluster_range, "__len__") and not isinstance(cluster_range, str):
            pass
        else:
            raise TypeError('Type not valid')

        gamma = 1. / len(self.signatures.diss[0])
        kernel = np.exp(-self.signatures.diss * gamma)
        cluster_silhouette = []
        for num_clusters in cluster_range:
            clusters = cluster.SpectralClustering(num_clusters, n_jobs=n_jobs, affinity='precomputed',
                                                  random_state=random_state, **kwargs).fit(kernel)
            score = metrics.silhouette_score(self.signatures.diss, clusters.labels_, metric='precomputed')
            cluster_silhouette.append(score)
        clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_silhouette': cluster_silhouette})
        return clusters_df

    def __agg_cluster_score(self, num_clusters, linkage='average', **kwargs):
        clusters = cluster.AgglomerativeClustering(num_clusters, linkage=linkage,
                                                   affinity='precomputed', **kwargs).fit(self.signatures.diss)
        score = metrics.silhouette_score(self.signatures.diss, clusters.labels_, metric='precomputed')
        return score

    def silhouette_score_agglomerative_range(self, cluster_range, linkage='average', n_jobs=1, **kwargs):
        """

        Parameters
        ----------
        cluster_range : list-like or int
            Range of the number of clusterings to obtain the silhouette score
        linkage : str
            Type of agglomerative linkage
        kwargs : key arguments to pass to the aggomerative clustering function

        Returns
        -------

        """

        if isinstance(cluster_range, int):
            cluster_range = list(range(2, cluster_range + 1))  # +1 to cluster up to cluster_range
        elif hasattr(cluster_range, "__len__") and not isinstance(cluster_range, str):
            pass
        else:
            raise TypeError('Type not valid')
        if n_jobs == 1:
            cluster_silhouette = []
            for num_clusters in cluster_range:
                score = self.__agg_cluster_score(num_clusters, linkage=linkage, **kwargs)
                cluster_silhouette.append(score)
            clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_silhouette': cluster_silhouette})
            return clusters_df
        else:
            if Pool is None:
                raise Exception('Please install the pathos package for this feature')
            p = Pool(n_jobs)
            res = p.amap(lambda x: self.__agg_cluster_score(x, linkage, **kwargs), cluster_range)
            scores = res.get()
            clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_silhouette': scores})
            return clusters_df

    def calinski_harabaz_score(self):
        if self.labels is None:
            raise Exception('you must cluster the signatures first')
        score = metrics.calinski_harabaz_score(self.signatures.sequences, self.labels)
        return score

    #TODO: Develop gap statistics score for sequences. Maybe get elbow plot as well
