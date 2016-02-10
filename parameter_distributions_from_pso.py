import os
import csv
import pandas as pd


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

par_sets = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_2000')

f = open(par_sets[0])
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))

all_pars = pd.DataFrame()
for i, pat in enumerate(par_sets):
    var = pd.read_table(pat, sep=',', names=['parameters','val'])
    all_pars['par%d'%i] = var.val
all_pars.set_index(var.parameters, inplace=True)
all_pars_t = all_pars.transpose()
all_pars_t.hist(column='pore_transport_dissociate_BaxA_4_CytoCC_kc')
all_pars