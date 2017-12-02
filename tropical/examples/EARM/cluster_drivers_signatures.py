from tropical import clustering
import pickle
from pathos.multiprocessing import ProcessingPool as Pool

with open('earm_signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)


def get_cluster_percentage_color(signatures_idx):
    signatures = all_signatures[signatures_idx]['consumption']
    clus = clustering.ClusterSequences(data=signatures, unique_sequences=False, truncate_seq=50)
    clus.diss_matrix()
    sil_df = clus.silhouette_score_kmeans_range(range(2, 31))
    n_clus = sil_df.loc[sil_df['cluster_silhouette'].idxmax(), 'num_clusters']
    print (signatures_idx, sil_df)
    clus.Kmeans(n_clusters=n_clus)
    return clus.cluster_percentage_color()

drivers = all_signatures.keys()
drivers.remove('species_combinations')
p = Pool(4)
res = p.amap(get_cluster_percentage_color, drivers)
results = res.get()

with open('results.pickle', 'wb') as fp:
    pickle.dump(results, fp)
