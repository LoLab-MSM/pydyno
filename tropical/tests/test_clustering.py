from tropical.examples.double_enzymatic.mm_two_paths_model import model
# from nose.tools import *
import numpy as np
from tropical import clustering
from pysb.testing import *


class TestClusteringBase(object):
    def setUp(self):
        self.signatures = [[2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
        self.clus = clustering.ClusterSequences(self.signatures, unique_sequences=False)

    def tearDown(self):
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

    # The k-medoids implementation throws an error when the clusters are empty
    # def test_kmedoids(self):
    #     self.clus.diss_matrix(metric='LCS')
    #     self.clus.Kmedoids(2)
    #     assert self.clus.cluster_method == 'kmedoids'
    #     assert len(self.clus.labels) == len(self.clus.sequences)
    #     assert not np.isnan(self.clus.labels).any()

    @raises(Exception)
    def test_kmedoids_no_diss_matrix(self):
        self.clus.Kmedoids(2)

    def test_agglomerative(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.agglomerative_clustering(2)
        assert self.clus.cluster_method == 'agglomerative'
        assert len(self.clus.labels) == len(self.clus.sequences)
        assert not np.isnan(self.clus.labels).any()

    def test_spectral(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.spectral_clustering(2)
        assert self.clus.cluster_method == 'spectral'
        assert len(self.clus.labels) == len(self.clus.sequences)
        assert not np.isnan(self.clus.labels).any()

    def test_silhouette_score_hdbscan(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.hdbscan(min_cluster_size=2, min_samples=1)
        score = self.clus.silhouette_score()
        assert 1 > score > -1

    def test_silhouette_score_agglomerative(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.agglomerative_clustering(2)
        score = self.clus.silhouette_score()
        assert 1 > score > -1

    @raises(Exception)
    def test_silhouette_score_without_clustering(self):
        self.clus.silhouette_score()

    def test_silhouette_spectral_range_2_4(self):
        self.clus.diss_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info_range = self.clus.silhouette_score_spectral_range(k_range)
        clust_n = k_range[-1]
        clus_info_int = self.clus.silhouette_score_spectral_range(clust_n)
        assert len(k_range) == len(clus_info_range['num_clusters'])
        scores = np.array([True for i in clus_info_range['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True
        np.testing.assert_allclose(clus_info_range['cluster_silhouette'],
                                   clus_info_int['cluster_silhouette'])

    def test_silhouette_score_agglomerative_range_2_4(self):
        self.clus.diss_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info_range = self.clus.silhouette_score_agglomerative_range(k_range)
        clust_n = k_range[-1]
        clus_info_int = self.clus.silhouette_score_agglomerative_range(clust_n)
        assert len(k_range) == len(clus_info_range['num_clusters'])
        scores = np.array([True for i in clus_info_range['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True
        np.testing.assert_allclose(clus_info_range['cluster_silhouette'],
                                   clus_info_int['cluster_silhouette'])

    @raises(TypeError)
    def test_silhouette_score_spectral_wrong_type(self):
        self.clus.silhouette_score_spectral_range('4')

    @raises(TypeError)
    def test_silhouette_score_agglomerative_wrong_type(self):
        self.clus.silhouette_score_agglomerative_range('4')

    @raises(TypeError)
    def test_silhouette_score_agglomerative_invalid_range(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.silhouette_score_agglomerative_range('bla')

    def test_calinski_harabaz_score(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.agglomerative_clustering(2)
        score = self.clus.calinski_harabaz_score()
        assert score == 17.0


    @raises(Exception)
    def test_calinski_harabaz_score_without_clustering(self):
        self.clus.calinski_harabaz_score()

    def test_neighborhood_density(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.agglomerative_clustering(n_clusters=2)
        clus_seqs0 = self.clus.sequences.iloc[self.clus.labels == 0]
        clus_seqs1 = self.clus.sequences.iloc[self.clus.labels == 1]
        clus_idx0 = clus_seqs0.index.get_level_values(0).values
        clus_idx1 = clus_seqs1.index.get_level_values(0).values
        rep0 = self.clus.neighborhood_density(proportion=0.5, sequences_idx=clus_idx0)
        rep1 = self.clus.neighborhood_density(proportion=0.5, sequences_idx=clus_idx1)
        np.testing.assert_allclose(rep0, np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1]))
        np.testing.assert_allclose(rep1, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))

    def test_centrality(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.agglomerative_clustering(n_clusters=2)
        clus_seqs0 = self.clus.sequences.iloc[self.clus.labels == 0]
        clus_seqs1 = self.clus.sequences.iloc[self.clus.labels == 1]
        clus_idx0 = clus_seqs0.index.get_level_values(0).values
        clus_idx1 = clus_seqs1.index.get_level_values(0).values
        rep0 = self.clus.centrality(sequences_idx=clus_idx0)
        rep1 = self.clus.centrality(sequences_idx=clus_idx1)
        np.testing.assert_allclose(rep0, np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        np.testing.assert_allclose(rep1, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))

    def test_frequency(self):
        self.clus.diss_matrix(metric='LCS')
        self.clus.agglomerative_clustering(n_clusters=2)
        clus_seqs0 = self.clus.sequences.iloc[self.clus.labels == 0]
        clus_seqs1 = self.clus.sequences.iloc[self.clus.labels == 1]
        clus_idx0 = clus_seqs0.index.get_level_values(0).values
        clus_idx1 = clus_seqs1.index.get_level_values(0).values
        rep0 = self.clus.frequency(sequences_idx=clus_idx0)
        rep1 = self.clus.frequency(sequences_idx=clus_idx1)
        np.testing.assert_allclose(rep0, np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1]))
        np.testing.assert_allclose(rep1, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))