from tropical.examples.double_enzymatic.mm_two_paths_model import model
# from nose.tools import *
import numpy as np
from tropical import clustering
from pysb.testing import *


class TestClusteringBase(object):
    def setUp(self):
        self.model = model
        self.time = np.linspace(0, 50, 101)
        self.signatures = [[2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
        self.clus = clustering.ClusterSequences(self.signatures, unique_sequences=False)

    def tearDown(self):
        self.model = None
        self.time = None
        self.signatures = None
        self.clus = None


class TestClusteringSingle(TestClusteringBase):
    def test_unique_sequences(self):
        unique_seqs = self.clus.get_unique_sequences(self.clus.sequences)
        assert len(unique_seqs) == 3

    def test_diss_matrix_lcs(self):
        self.clus.diss_matrix(metric='LCS')
        seq_len = len(self.clus.sequences)
        assert self.clus.diss.shape == (seq_len, seq_len)
        assert not np.isnan(self.clus.diss).any()

    def test_diss_matrix_levenshtein(self):
        self.clus.diss_matrix(metric='levenshtein')
        seq_len = len(self.clus.sequences)
        assert self.clus.diss.shape == (seq_len, seq_len)
        assert not np.isnan(self.clus.diss).any()

    @raises(ValueError)
    def test_diss_matrix_invalid_metric(self):
        self.clus.diss_matrix(metric='bla')

    def test_hdbscan(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.hdbscan()
        assert self.clus.cluster_method == 'hdbscan'
        assert len(self.clus.labels) == len(self.clus.sequences)
        assert not np.isnan(self.clus.labels).any()

    @raises(Exception)
    def test_hdbscan_no_diss_matrix(self):
        self.clus.hdbscan()

    # def test_kmedoids(self):
    #     self.clus.diss_matrix(metric='LCS')
    #     self.clus.Kmedoids(2)
    #     assert self.clus.cluster_method == 'kmedoids'
    #     assert len(self.clus.labels) == len(self.clus.sequences)
    #     assert not np.isnan(self.clus.labels).any()

    @raises(Exception)
    def test_kmedoids_no_diss_matrix(self):
        self.clus.Kmedoids(2)

    def test_kmeans(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.Kmeans(2)
        assert self.clus.cluster_method == 'kmeans'
        assert len(self.clus.labels) == len(self.clus.sequences)
        assert not np.isnan(self.clus.labels).any()

    @raises(Exception)
    def test_kmeans_no_diss_matrix(self):
        self.clus.Kmeans(2)

    def test_silhouette_score_hdbscan(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.hdbscan(min_cluster_size=2, min_samples=1)
        score = self.clus.silhouette_score()
        assert 1 > score > -1

    def test_silhouette_score_kmeans(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.Kmeans(2)
        score = self.clus.silhouette_score()
        assert 1 > score > -1

    @raises(Exception)
    def test_silhoutte_score_without_clustering(self):
        self.clus.silhouette_score()

    def test_silhouette_score_kmeans_range_2_4(self):
        self.clus.diss_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info = self.clus.silhouette_score_kmeans_range(k_range)
        assert len(k_range) == len(clus_info['num_clusters'])
        scores = np.array([True for i in clus_info['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True

    @raises(TypeError)
    def test_silhouette_score_kmeans_invalid_range(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.silhouette_score_kmeans_range('bla')

    def test_calinski_harabaz_score(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.Kmeans(2)
        score = self.clus.calinski_harabaz_score()
        assert score == 17.0

