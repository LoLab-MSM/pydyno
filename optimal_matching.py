from Bio import pairwise2
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np


def list2string(a):
    return "".join(str(elm) for elm in a)


def optimal_matching(m):
    return np.array([[-pairwise2.align.globalms(list2string(a), list2string(b), 0, -2, -1, -1, score_only=True)
                      for a in m] for b in m])

species_df = pd.read_csv('/home/oscar/home/oscar/PycharmProjects/tropical/examples/CORM/data_frames_unique_pars/data_frame3.csv',
                         header=0, index_col=0).drop_duplicates()
species_ndarray = species_df.as_matrix()

model = AgglomerativeClustering(n_clusters=2, linkage="average", affinity=optimal_matching)
bb = model.fit_predict(species_ndarray)
