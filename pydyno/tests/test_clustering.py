# from pydyno.examples.double_enzymatic.mm_two_paths_model import model
from nose.tools import *
import numpy as np
from pydyno.sequences import Sequences
# from pysb.testing import *
import os


class TestClusteringBase(object):
    @classmethod
    def tearDownClass(cls):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        test = os.listdir(dir_name)
        for item in test:
            if item.endswith(".png") or item.endswith(".pdf"):
                os.remove(os.path.join(dir_name, item))

    def setUp(self):
        seqsdata = [[2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
        self.signatures = Sequences(seqsdata, 's0')

    def tearDown(self):
        self.signatures = None


class TestClusteringSingle(TestClusteringBase):

    def test_hdbscan(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.hdbscan_clustering()
        assert self.signatures.cluster_method == 'hdbscan'
        assert len(self.signatures.labels) == len(self.signatures.sequences)
        assert not np.isnan(self.signatures.labels).any()

    @raises(Exception)
    def test_hdbscan_no_diss_matrix(self):
        self.signatures.hdbscan_clustering()

    # The k-medoids implementation throws an error when the clusters are empty
    def test_kmedoids(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.Kmedoids(2)
        assert self.signatures.cluster_method == 'kmedoids'
        assert len(self.signatures.labels) == len(self.signatures.sequences)
        assert not np.isnan(self.signatures.labels).any()

    @raises(Exception)
    def test_kmedoids_no_diss_matrix(self):
        self.signatures.Kmedoids(2)

    def test_agglomerative(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.agglomerative_clustering(2)
        assert self.signatures.cluster_method == 'agglomerative'
        assert len(self.signatures.labels) == len(self.signatures.sequences)
        assert not np.isnan(self.signatures.labels).any()

    def test_spectral(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.spectral_clustering(2)
        assert self.signatures.cluster_method == 'spectral'
        assert len(self.signatures.labels) == len(self.signatures.sequences)
        assert not np.isnan(self.signatures.labels).any()

    def test_silhouette_score_hdbscan(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.hdbscan_clustering(min_cluster_size=2, min_samples=1)
        score = self.signatures.silhouette_score()
        assert 1 > score > -1

    def test_silhouette_score_agglomerative(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.agglomerative_clustering(2)
        score = self.signatures.silhouette_score()
        assert 1 > score > -1

    @raises(Exception)
    def test_silhouette_score_without_clustering(self):
        self.signatures.silhouette_score()

    def test_silhouette_spectral_range_2_4(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info_range = self.signatures.silhouette_score_spectral_range(k_range)
        clust_n = k_range[-1]
        clus_info_int = self.signatures.silhouette_score_spectral_range(clust_n)
        assert len(k_range) == len(clus_info_range['num_clusters'])
        scores = np.array([True for i in clus_info_range['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True
        np.testing.assert_allclose(clus_info_range['cluster_silhouette'],
                                   clus_info_int['cluster_silhouette'])

    def test_silhouette_score_agglomerative_range_2_4(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        k_range = range(2, 4)
        clus_info_range = self.signatures.silhouette_score_agglomerative_range(k_range)
        clust_n = k_range[-1]
        clus_info_int = self.signatures.silhouette_score_agglomerative_range(clust_n)
        assert len(k_range) == len(clus_info_range['num_clusters'])
        scores = np.array([True for i in clus_info_range['cluster_silhouette'] if 1 > i > -1])
        assert scores.all() == True
        np.testing.assert_allclose(clus_info_range['cluster_silhouette'],
                                   clus_info_int['cluster_silhouette'])

    @raises(TypeError)
    def test_silhouette_score_spectral_wrong_type(self):
        self.signatures.silhouette_score_spectral_range('4')

    @raises(TypeError)
    def test_silhouette_score_agglomerative_wrong_type(self):
        self.signatures.silhouette_score_agglomerative_range('4')

    @raises(TypeError)
    def test_silhouette_score_agglomerative_invalid_range(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.silhouette_score_agglomerative_range('bla')

    def test_calinski_harabaz_score(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.agglomerative_clustering(2)
        score = self.signatures.calinski_harabaz_score()
        assert score == 17.0

    @raises(Exception)
    def test_calinski_harabaz_score_without_clustering(self):
        self.signatures.calinski_harabaz_score()
