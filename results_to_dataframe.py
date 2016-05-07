# creates pandas data frames to use in r of the tropical signatures
import pandas
import numpy as np
import os


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def results_to_dataframe(tspan, trop_data_path, pars_path, out_path):

    tropical_data = np.load(trop_data_path)
    drivers_all = [set(dr.keys()) for dr in tropical_data]
    drivers_all_pars = set.intersection(*drivers_all)
    drivers_to_df = {}
    for sp in drivers_all_pars:
        tmp = [0] * len(drivers_all)
        for idx, tro in enumerate(tropical_data):
            tmp[idx] = tro[sp]
        drivers_to_df[sp] = tmp

    for sp in drivers_to_df.keys():
        pandas.DataFrame(np.array(drivers_to_df[sp]),
                         index=listdir_fullpath(pars_path),
                         columns=tspan).to_csv(
            'out_path/data_frame%d' % sp + '.csv')
    return

tspan = np.linspace(0, 20000, 100)[1:]/60
tspan_tro = np.around(tspan, 1)

tropical_data = np.load('/home/oscar/tropical_project_new/drivers_all_parameters5000.npy')
drivers_all = [set(dr.keys()) for dr in tropical_data]
drivers_over_pars = set.intersection(*drivers_all)
drivers_to_df = {}
for sp in drivers_over_pars:
    tmp = [0] * len(drivers_all)
    for idx, tro in enumerate(tropical_data):
        tmp[idx] = tro[sp]
    drivers_to_df[sp] = tmp
print drivers_to_df.keys()

for sp in drivers_to_df.keys():
    pandas.DataFrame(np.array(drivers_to_df[sp]),
                     index=listdir_fullpath('/home/oscar/tropical_project_new/parameters_5000'),
                     columns=tspan_tro).to_csv(
        '/home/oscar/tropical_project_new/data_frames5000/data_frame%d' % sp + '.csv')