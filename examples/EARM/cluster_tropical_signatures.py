from tropical import clustering


signatures_path37 = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
                  'signatures_all_kpars/data_frames/data_frame37.csv'

signatures_path3 = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
                  'signatures_all_kpars/data_frames/data_frame3.csv'

signatures_path6 = '/Users/dionisio/Documents/PhD/tropical_earm_IC/signatures_within_range_function/' \
                  'signatures_all_kpars/data_frames/data_frame6.csv'

a = clustering.ClusterSequences(data=signatures_path6, truncate_seq=50)
a.diss_matrix()
a.hdbscan(min_samples=10)
b = clustering.PlotSequences(a)
b.all_trajectories_plot()