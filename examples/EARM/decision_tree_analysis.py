from sklearn import tree
import numpy as np
import os
from earm.lopez_embedded import model

# bid_0: 55, bclxl_0: 56, mcl1_0: 57, bcl2_0: 58

directory = os.path.dirname(__file__)
initials_path = os.path.join(directory, 'IC_10000_pars_consumption_kpar0.npy')
initials = np.load(initials_path)

initials_antiapo = initials[:, 55:59]

clusters_info = np.loadtxt('/Users/dionisio/Desktop/data_frame37.csv', dtype=np.int, delimiter=',')

clf = tree.DecisionTreeClassifier().fit(initials_antiapo, clusters_info)