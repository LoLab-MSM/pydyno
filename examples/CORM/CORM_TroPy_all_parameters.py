import numpy as np
import pandas as pd
from corm import model
from drivers_diff_parameters import compare_all_drivers_signatures


sets_of_pars = 2
pars_to_use = np.zeros((sets_of_pars, len(model.parameters)))

all_dream_log_parames = np.load("/home/oscar/PycharmProjects/CORM/results/2015_02_02_COX2_all_traces.npy")
unique_dream_log_pars = pd.DataFrame(all_dream_log_parames).drop_duplicates()

sets_of_pars = len(unique_dream_log_pars.index)

pysb_sampled_parameter_names = ['kr_AA_cat2', 'kcat_AA2', 'kr_AA_cat3', 'kcat_AA3', 'kr_AG_cat2', 'kr_AG_cat3',
                                'kcat_AG3', 'kr_AA_allo1', 'kr_AA_allo2', 'kr_AA_allo3', 'kr_AG_allo1', 'kr_AG_allo2']

generic_kf = np.log10(1.5e4)

for pa in range(sets_of_pars):
    param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, unique_dream_log_pars.iloc[[pa]])}
    for pname, pvalue in param_dict.items():

        # Sub in parameter values at current location in parameter space

        if 'kr' in pname:
            model.parameters[pname].value = 10 ** (pvalue + generic_kf)

        elif 'kcat' in pname:
            model.parameters[pname].value = 10 ** pvalue

    pars_to_use[pa] = [par.value for par in model.parameters]

tspan = np.linspace(0, 10, num=100)
compare_all_drivers_signatures(model, tspan, pars_to_use, to_data_frame=True)


