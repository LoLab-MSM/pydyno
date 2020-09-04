import os
import numpy as np
from pydyno.seqanalysis import SeqAnalysis
import pytest


@pytest.fixture(scope='function')
def signatures():
    seqsdata = np.array([[11, 12, 13, 14, 15], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                         [6, 7, 8, 12, 11], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])
    signatures = SeqAnalysis(seqsdata, 's0')
    return signatures


@pytest.fixture(scope='class')
def labels():
    return [0, 1, 2, 3, 0, 1]


class TestClustering:
    def test_unique_sequences(self, signatures):
        unique_seqs = signatures.unique_sequences()
        assert len(unique_seqs.sequences) == 4

    def test_diss_matrix_lcs(self, signatures):
        signatures.dissimilarity_matrix(metric='LCS')
        seq_len = len(signatures.sequences)
        assert signatures.diss.shape == (seq_len, seq_len)
        assert not np.isnan(signatures.diss).any()
        np.testing.assert_allclose(signatures.diss,
                                   np.array([[0., 10., 10., 8., 0., 10.],
                                             [10., 0., 10., 10., 10., 0.],
                                             [10., 10., 0., 4., 10., 10.],
                                             [8., 10., 4., 0., 8., 10.],
                                             [0., 10., 10., 8., 0., 10.],
                                             [10., 0., 10., 10., 10., 0.]]))

    def test_truncate_sequence(self, signatures):
        tseq = signatures.truncate_sequences(idx=2)
        assert len(tseq.sequences.columns) == 2
        assert len(tseq.sequences.columns) < len(signatures.sequences.columns)

    def test_diss_matrix_levenshtein(self, signatures):
        signatures.dissimilarity_matrix(metric='levenshtein')
        seq_len = len(signatures.sequences)
        assert signatures.diss.shape == (seq_len, seq_len)
        assert not np.isnan(signatures.diss).any()

    def test_diss_matrix_invalid_metric(self, signatures):
        with pytest.raises(ValueError):
            signatures.dissimilarity_matrix(metric='bla')

    def test_neighborhood_density(self, signatures):
        diss = signatures.dissimilarity_matrix(metric='LCS')
        rep0 = signatures.seq_representativeness(diss=diss, method='density')
        np.testing.assert_allclose(signatures.sequences.iloc[rep0[0]], np.array([1, 2, 3, 4, 5]))

    def test_dist(self, signatures):
        diss = signatures.dissimilarity_matrix(metric='LCS')
        rep0 = signatures.seq_representativeness(diss=diss, method='dist')
        np.testing.assert_allclose(signatures.sequences.iloc[rep0[0]], np.array([11, 12, 13, 14, 15]))

    def test_frequency(self, signatures):
        diss = signatures.dissimilarity_matrix(metric='LCS')
        rep0 = signatures.seq_representativeness(diss=diss, method='freq')
        np.testing.assert_allclose(signatures.sequences.iloc[rep0[0]], np.array([1, 2, 3, 4, 5]))

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

    def test_all_trajectories(self, signatures, data_files_dir):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.plot_sequences(type_fig='trajectories', plot_all=True, dir_path=data_files_dir)

    def test_entropy(self, signatures, data_files_dir):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.plot_sequences(type_fig='entropy', plot_all=True, dir_path=data_files_dir)

    def test_modal(self, signatures, data_files_dir):
        signatures.dissimilarity_matrix(metric='LCS')
        signatures.plot_sequences(type_fig='modal', plot_all=True, dir_path=data_files_dir)

    def test_save_load(self, signatures, data_files_dir):
        path_dict = {'0': {'order': 0, 'name': 's3'}}
        signatures.dissimilarity_matrix(metric='LCS')
        path_signature1 = os.path.join(data_files_dir, 'test_signature1.h5')
        path_signature2 = os.path.join(data_files_dir, 'test_signature2.h5')

        # Try to save and reload when file contains only the sequences
        signatures.save(path_signature1)
        loaded_signatures1 = SeqAnalysis.load(path_signature1)

        # Try to save and reload when file contains sequences and dominant paths
        signatures.save(path_signature2, path_dict)
        loaded_signatures2, dom_paths = SeqAnalysis.load(path_signature2)

        # Trying to overwrite an existing dataset gives a ValueError
        with pytest.raises(IOError):
            signatures.save(path_signature1)

        # Check sequences and dom path dictionaries are the same
        loaded_signatures1.sequences.equals(signatures.sequences)
        loaded_signatures2.sequences.equals(signatures.sequences)
        assert dom_paths == path_dict
