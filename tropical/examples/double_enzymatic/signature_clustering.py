import pickle
from tropical import clustering
with open('signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

sp0_sign_reactants = all_signatures[0]['consumption']
clus = clustering.ClusterSequences(sp0_sign_reactants, unique_sequences=False)
clus.diss_matrix(n_jobs=1)
clus.agglomerative_clustering(2)
pl = clustering.PlotSequences(clus)
pl.all_trajectories_plot()