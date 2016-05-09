import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def listdir_fullpath(d):
    return [os.path.join(d, fi) for fi in os.listdir(d)]

par_sets = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_2000')

f = open(par_sets[0])
data = csv.reader(f)
parames = [float(i[1]) for i in data]

all_pars = pd.DataFrame()
for i, pat in enumerate(par_sets):
    var = pd.read_table(pat, sep=',', names=['parameters', 'val'])
    all_pars['par%d' % i] = var.val
all_pars.set_index(var.parameters, inplace=True)
all_pars_t = all_pars.transpose()

plt.figure()
all_pars_t['bind_L_R_to_LR_kr'].plot.hist()
plt.show()
