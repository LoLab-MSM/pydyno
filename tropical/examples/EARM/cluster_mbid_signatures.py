from tropical import clustering
import pickle

with open('earm_signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

sp37_signatures = all_signatures[37]['consumption']
a = clustering.ClusterSequences(data=sp37_signatures, unique_sequences=False, truncate_seq=50)
a.diss_matrix()
a.hdbscan(min_samples=5)
a.silhouette_score()
b = clustering.PlotSequences(a)
b.modal_plot(title='Mitochondrial Bid')
b.all_trajectories_plot(title='Mitochondrial Bid')

