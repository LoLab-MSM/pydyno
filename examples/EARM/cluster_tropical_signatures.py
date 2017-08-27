from tropical import clustering


signatures_path = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
                  'signatures_all_kpars/data_frames/data_frame37.csv'

a = clustering.ClusterSequences(data=signatures_path, truncate_seq=50)
a.diss_matrix()
a.hdbscan()
b = clustering.PlotSequences(a)
b.modal_plot()