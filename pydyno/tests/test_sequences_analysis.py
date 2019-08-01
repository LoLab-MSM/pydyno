# from pydyno.examples.double_enzymatic.mm_two_paths_model import model
# from nose.tools import *
import numpy as np
from pydyno.sequences import Sequences
from pysb.testing import *
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
        seqsdata = np.array([[11, 12, 13, 14, 15], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                             [6, 7, 8, 12, 11], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])
        self.signatures = Sequences(seqsdata, 's0')

        self.labels = [0, 1, 2, 3, 0, 1]

    def tearDown(self):
        self.signatures = None
        self.labels = None


class TestSequenceAnalysis(TestClusteringBase):
    def test_unique_sequences(self):
        unique_seqs = self.signatures.unique_sequences()
        assert len(unique_seqs.sequences) == 4

    def test_diss_matrix_lcs(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        seq_len = len(self.signatures.sequences)
        assert self.signatures.diss.shape == (seq_len, seq_len)
        assert not np.isnan(self.signatures.diss).any()
        np.testing.assert_allclose(self.signatures.diss,
                                   np.array([[0., 10., 10., 8., 0., 10.],
                                             [10., 0., 10., 10., 10., 0.],
                                             [10., 10., 0., 4., 10., 10.],
                                             [8., 10., 4., 0., 8., 10.],
                                             [0., 10., 10., 8., 0., 10.],
                                             [10., 0., 10., 10., 10., 0.]]))

    def test_truncate_sequence(self):
        tseq = self.signatures.truncate_sequences(idx=2)
        assert len(tseq.sequences.columns) == 2
        assert len(tseq.sequences.columns) < len(self.signatures.sequences.columns)

    def test_diss_matrix_levenshtein(self):
        self.signatures.dissimilarity_matrix(metric='levenshtein')
        seq_len = len(self.signatures.sequences)
        assert self.signatures.diss.shape == (seq_len, seq_len)
        assert not np.isnan(self.signatures.diss).any()

    @raises(ValueError)
    def test_diss_matrix_invalid_metric(self):
        self.signatures.dissimilarity_matrix(metric='bla')

    def test_neighborhood_density(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        rep0 = self.signatures.neighborhood_density(proportion=0.5, sequences_idx=None)
        np.testing.assert_allclose(rep0, np.array([1, 2, 3, 4, 5]))

    def test_centrality(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        rep0 = self.signatures.centrality(sequences_idx=None)
        np.testing.assert_allclose(rep0[1], np.array([11, 12, 13, 14, 15]))

    def test_frequency(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        rep0 = self.signatures.frequency(sequences_idx=None)
        np.testing.assert_allclose(rep0, np.array([6,  7,  8, 12, 11]))

    # def test_cluster_percentage_color(self):
    #     self.clus.diss_matrix(metric='LCS')
    #     self.clus.agglomerative_clustering(n_clusters=2)
    #     self.clus.cluster_percentage_color()
    #
    # def test_modal_plot(self):
    #     self.clus.diss_matrix(metric='LCS')
    #     self.clus.agglomerative_clustering(n_clusters=2)
    #     pl = plot_signatures.PlotSequences(self.clus)
    #     pl.plot_sequences(type_fig='modal')

    def test_all_trajectories(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.plot_sequences(type_fig='trajectories', plot_all=True)

    def test_entropy(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.plot_sequences(type_fig='entropy', plot_all=True)

    def test_modal(self):
        self.signatures.dissimilarity_matrix(metric='LCS')
        self.signatures.plot_sequences(type_fig='modal', plot_all=True)
