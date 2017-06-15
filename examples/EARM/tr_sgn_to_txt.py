from pysb.integrate import ScipyOdeSimulator
from tropical.max_plus_multiprocessing_numpy import run_tropical
from tropical.helper_functions import parse_name
import numpy as np
import pandas as pd


def trajectories_signature_2_txt(model, tspan, sp_to_analyze=None, parameters=None, file_path=''):
    y = ScipyOdeSimulator(model, tspan=tspan, param_values=parameters).run().dataframe
    observables = [obs.name for obs in model.observables]
    y.drop(observables, axis=1, inplace=True)
    sp_short_names = [parse_name(sp) for sp in model.species]
    y.columns = sp_short_names
    # y['time'] = tspan
    signatures = run_tropical(model, tspan, parameters, diff_par=1, type_sign='consumption')
    for sp in sp_to_analyze:
        y[parse_name(model.species[sp])+'_truncated'] = signatures[sp]
    y.to_csv(file_path+'tr_sig.txt')

    if parameters is not None:
        initials = np.zeros([1, len(model.species)])
        initials_df = pd.DataFrame(data=initials, columns=sp_short_names)
        initial_pars = [ic[1] for ic in model.initial_conditions]
        for i, par in enumerate(model.parameters):
            if par in initial_pars:
                idx = initial_pars.index(par)
                ic_name = parse_name(model.initial_conditions[idx][0])
                ic_value = parameters[i]
                initials_df[ic_name] = ic_value
        # initials_idx_in_pars = np.array([i for i, j in enumerate(model.parameters)
        #                                  if j in [par[1] for par in model.initial_conditions]])
        # initials[0][initials_idx_in_pars] = parameters[initials_idx_in_pars]
        # initials_df = pd.DataFrame(data=initials, columns=sp_short_names)
        initials_df.to_csv(file_path+'initials.txt', index=False)
    return