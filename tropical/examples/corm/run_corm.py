import numpy as np
from corm import model
from util import sub_parameters
from pysb.simulator.scipyode import ScipyOdeSimulator
from pathos.multiprocessing import ProcessingPool as Pool
from pysb.simulator.base import SimulationResult

t = np.linspace(0, 10, num=100)
param_names = ['kcat_AA2', 'kcat_AA3', 'KD_AG_cat3', 'KD_AG_cat2', 'KD_AG_allo2', 'KD_AG_allo1', 'KD_AA_allo1',
               'KD_AA_allo2', 'KD_AA_allo3', 'kcat_AG3', 'KD_AA_cat3', 'KD_AA_cat2']

solver = ScipyOdeSimulator(model, tspan=t)

# def transform_parameters(pars):
#     param_dict = {name: param for name, param in zip(param_names, pars)}
#     sub_parameters(model=model, param_dict=param_dict)
#     paramvals = [param.value for param in model.parameters]
#     return paramvals
#
#
# p = Pool(4)
# calibrated_parameters = np.load('2015_02_02_COX2_all_traces.npy')
# unique_pars, unique_idx = np.unique(calibrated_parameters, axis=0, return_index=True)
# pars_ready = p.map(transform_parameters, unique_pars)
# pars_ready = np.array(pars_ready)
# np.save('unique_complete_pars.npy', pars_ready)


def simulate_corm(pars):
    y = solver.run(param_values=pars).species
    return y


pars_ready = np.load('unique_complete_pars.npy')
p = Pool(11)
sims = p.amap(simulate_corm, pars_ready)
s=SimulationResult(simulator=None, tout=np.tile(t, (len(pars_ready), 1)), trajectories=sims.get(), model=model,
                 param_values=pars_ready)
s.save('corm_unique_trajectories.h5')