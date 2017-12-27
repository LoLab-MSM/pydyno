from tropical import clustering
import pickle
from pathos.multiprocessing import ProcessingPool as Pool

with open('earm_signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

sil_threshold = 0.025


def cluster_percentage_color_aggomerative(signatures_idx):
    signatures = all_signatures[signatures_idx]['consumption']
    clus = clustering.ClusterSequences(seqdata=signatures, unique_sequences=False, truncate_seq=50)
    clus.diss_matrix(n_jobs=4)
    sil_df = clus.silhouette_score_agglomerative_range(cluster_range=range(2, 31))
    if sil_threshold:
        silh_diff = sil_df['cluster_silhouette'].max() - sil_threshold
        # Define n_clus to have the minimum number of clusters when silh scores are too similar
        best_silhs = sil_df.loc[sil_df['cluster_silhouette'] > silh_diff]
        best_silh, n_clus = best_silhs.loc[best_silhs['num_clusters'].idxmin()]
    else:
        best_silh, n_clus = sil_df.loc[sil_df['cluster_silhouette'].idxmax()]
    n_clus = int(n_clus)
    clus.agglomerative_clustering(n_clusters=n_clus)
    cluster_information = {signatures_idx: clus.cluster_percentage_color(),
                           'best_silh': best_silh, 'labels': clus.labels}
    return cluster_information


def cluster_percentage_color_hdbscan(signatures_idx):
    signatures = all_signatures[signatures_idx]['consumption']
    clus = clustering.ClusterSequences(seqdata=signatures, unique_sequences=False, truncate_seq=50)
    clus.diss_matrix(n_jobs=4)
    clus.hdbscan()
    cluster_information = {signatures_idx: clus.cluster_percentage_color(),
                           'best_silh': clus.silhouette_score(), 'labels': clus.labels}
    return cluster_information


def cluster_percentage_color_spectral(signatures_idx):
    signatures = all_signatures[signatures_idx]['consumption']
    clus = clustering.ClusterSequences(seqdata=signatures, unique_sequences=False, truncate_seq=50)
    clus.diss_matrix(n_jobs=4)
    sil_df = clus.silhouette_score_spectral_range(cluster_range=range(2, 31), n_jobs=4, random_state=1234)
    if sil_threshold:
        silh_diff = sil_df['cluster_silhouette'].max() - sil_threshold
        # Define n_clus to have the minimum number of clusters when silh scores are too similar
        best_silhs = sil_df.loc[sil_df['cluster_silhouette'] > silh_diff]
        best_silh, n_clus = best_silhs.loc[best_silhs['num_clusters'].idxmin()]
    else:
        best_silh, n_clus = sil_df.loc[sil_df['cluster_silhouette'].idxmax()]
    n_clus = int(n_clus)
    clus.spectral_clustering(n_clusters=n_clus, n_jobs=4, random_state=1234)
    cluster_information = {signatures_idx: clus.cluster_percentage_color(),
                           'best_silh': best_silh, 'labels': clus.labels}
    return cluster_information


drivers = all_signatures.keys()
drivers.remove('species_combinations')
p = Pool(4)
res = p.amap(cluster_percentage_color_spectral, drivers)
results = res.get()

with open('results3.pickle', 'wb') as fp:
    pickle.dump(results, fp)
