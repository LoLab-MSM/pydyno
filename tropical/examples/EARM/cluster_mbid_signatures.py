from tropical import clustering
import pickle
import numpy as np

with open('earm_signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

sp37_signatures = all_signatures[37]['consumption']
clus = clustering.ClusterSequences(seqdata=sp37_signatures, unique_sequences=False, truncate_seq=50)
clus.diss_matrix(n_jobs=4)
clus.spectral_clustering(n_clusters=8)
clus.silhouette_score()
b = clustering.PlotSequences(clus)
b.modal_plot(title='Mitochondrial Bid')
b.all_trajectories_plot(title='Mitochondrial Bid')

