import numpy as np
from pydyno.seqanalysis import SeqAnalysis
import pytest


@pytest.fixture(scope='function')
def signatures():
    seqsdata = [[2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    signatures = SeqAnalysis(seqsdata, 's0')
    return signatures


class TestClustering:
    def test_hdbscan(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.hdbscan_clustering()
        assert signatures.cluster_method == 'hdbscan'
        assert len(signatures.labels) == len(signatures.sequences)
        assert not np.isnan(signatures.labels).any()

    def test_hdbscan_no_diss_matrix(self, signatures):
        with pytest.raises(Exception):
            signatures.hdbscan_clustering()

    # The k-medoids implementation throws an error when the clusters are empty
    def test_kmedoids(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.Kmedoids(2)
        assert signatures.cluster_method == 'kmedoids'
        assert len(signatures.labels) == len(signatures.sequences)
        assert not np.isnan(signatures.labels).any()

    def test_kmedoids_no_diss_matrix(self, signatures):
        with pytest.raises(Exception):
            signatures.Kmedoids(2)

    def test_agglomerative(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.agglomerative_clustering(2)
        assert signatures.cluster_method == 'agglomerative'
        assert len(signatures.labels) == len(signatures.sequences)
        assert not np.isnan(signatures.labels).any()

    def test_spectral(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.spectral_clustering(2)
        assert signatures.cluster_method == 'spectral'
        assert len(signatures.labels) == len(signatures.sequences)
        assert not np.isnan(signatures.labels).any()

    def test_silhouette_score_hdbscan(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.hdbscan_clustering(min_cluster_size=2, min_samples=1)
        score = signatures.silhouette_score()
        assert 1 > score > -1

    def test_silhouette_score_agglomerative(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.agglomerative_clustering(2)
        score = signatures.silhouette_score()
        assert 1 > score > -1

    def test_silhouette_score_without_clustering(self):
        with pytest.raises(Exception):
            signatures.silhouette_score()

    def test_silhouette_spectral_range_2_4(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info_range = signatures.silhouette_score_spectral_range(k_range)
        clust_n = k_range[-1]
        clus_info_int = signatures.silhouette_score_spectral_range(clust_n)
        assert len(k_range) == len(clus_info_range['num_clusters'])
        scores = np.array([True for i in clus_info_range['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True
        np.testing.assert_allclose(clus_info_range['cluster_silhouette'],
                                   clus_info_int['cluster_silhouette'])

    def test_silhouette_score_agglomerative_range_2_4(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info_range = signatures.silhouette_score_agglomerative_range(k_range)
        clust_n = k_range[-1]
        clus_info_int = signatures.silhouette_score_agglomerative_range(clust_n)
        assert len(k_range) == len(clus_info_range['num_clusters'])
        scores = np.array([True for i in clus_info_range['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True
        np.testing.assert_allclose(clus_info_range['cluster_silhouette'],
                                   clus_info_int['cluster_silhouette'])

    def test_silhouette_score_spectral_wrong_type(self, signatures):
        with pytest.raises(TypeError):
            signatures.silhouette_score_spectral_range('4')

    def test_silhouette_score_agglomerative_wrong_type(self, signatures):
        with pytest.raises(TypeError):
            signatures.silhouette_score_agglomerative_range('4')

    def test_silhouette_score_agglomerative_invalid_range(self, signatures):
        with pytest.raises(TypeError):
            signatures.dissimilarity_matrix(metric='LCS')
            signatures.silhouette_score_agglomerative_range('bla')

    def test_calinski_harabaz_score(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.agglomerative_clustering(2)
        score = signatures.calinski_harabaz_score()
        assert score == 17.0

    def test_calinski_harabaz_score_without_clustering(self, signatures):
        with pytest.raises(Exception):
            signatures.calinski_harabaz_score()

