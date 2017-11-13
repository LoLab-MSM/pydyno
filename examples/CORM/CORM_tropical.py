import numpy as np
from corm import model

from old_functions.max_plus_multiprocessing_numpy import run_tropical

# tipe 1 cluster: 8509
# type 2 cluster: 7848
# type 3 cluster: 1784

all_dream_log_parames = np.load("/home/oscar/PycharmProjects/CORM/results/2015_02_02_COX2_all_traces.npy")[1784]

pysb_sampled_parameter_names = ['kr_AA_cat2', 'kcat_AA2', 'kr_AA_cat3', 'kcat_AA3', 'kr_AG_cat2', 'kr_AG_cat3',
                                'kcat_AG3', 'kr_AA_allo1', 'kr_AA_allo2', 'kr_AA_allo3', 'kr_AG_allo1', 'kr_AG_allo2']

generic_kf = np.log10(1.5e4)

param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, all_dream_log_parames)}
for pname, pvalue in param_dict.items():

    # Sub in parameter values at current location in parameter space

    if 'kr' in pname:
        model.parameters[pname].value = 10 ** (pvalue + generic_kf)

    elif 'kcat' in pname:
        model.parameters[pname].value = 10 ** pvalue


tspan = np.linspace(0, 0.005, num=100)
run_tropical(model, tspan,   sp_visualize=[3, 4])
