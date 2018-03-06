import matplotlib
matplotlib.use('agg')
from pysb.simulator import ScipyOdeSimulator
from earm2_flat import model
import numpy as np
import pickle
import pandas as pd
from multiprocessing import Pool
from collections import OrderedDict
import seaborn as sns
from tropical.util import sig_apop, column, curve_fit_ftn

tspan = np.linspace(0,20000, 100)
calibrated_pars = np.load('calibrated_6572pars.npy')

with open('results_spectral.pickle', 'rb') as handle:
    clusters = pickle.load(handle)
clusters_sp37 = clusters[13]['labels']

num_of_clusters = set(clusters_sp37)
clus_pars = {}
for j in num_of_clusters:
    item_index = np.where(clusters_sp37 == j)
    clus_pars[j] = calibrated_pars[item_index]

perturbations2 = {56: 0.2, 57: 0.2, 58: 0.2}

solver = ScipyOdeSimulator(model, tspan=tspan)


def run_one(param_values):
    sim = solver.run(param_values=param_values).all
    return sim


def species_effect_tdeath(cluster_pars, perturbations, ftn, violinplot=True, num_processors=4):
    """
    This works only for EARM 2.0 Model
    Parameters
    ----------
    solver : pysb solver
    cluster_pars : dict
        A dictionary whose keys are the cluster label and the values are the
        parameter sets in each cluster
    perturbations : dict
        A dictionary whose keys are the indices of the parameters to be perturbed
        and the values are the percentage of perturbation
    sp_analysis : vector-like
        Indices of species to analyze
    ftn : callable
        Function to apply to trajectories to have a experimental value
    metric : str
        Metric to analyze results from function
    num_processors
    kwargs

    Returns
    -------

    """
    p = Pool(processes=num_processors)
    effects_total = OrderedDict()
    for clus, pars in cluster_pars.items():
        effects_data = {}
        for par_idx, perc in perturbations.items():
            param_values = np.array(pars)
            param_values[:, par_idx] = param_values[:, par_idx] * perc
            sims = p.map(run_one, param_values)
            cparp = [sim['__s39'] for sim in sims]
            td = curve_fit_ftn(fn=ftn, xdata=tspan, ydata=cparp, p0=[100,100,100])
            td_hist = column(td, 1)
            effects_data[par_idx] = td_hist
            # std = np.std(td_hist)
            # mean = np.average(td_hist)
            # effects.loc[str(par_idx), str(clus)] = (mean, std)
        effects_total[clus] = effects_data
    print (effects_total)
    if violinplot:
        for par in perturbations.keys():
            par_data = [td[par] for td in effects_total.values()]
            g = sns.boxplot(data=par_data, orient='h')
            g.set_xlabel('Time to MOMP')
            g.set_ylabel('Cluster label')
            # g = sns.violinplot(data=par_data, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            g.set_yticklabels(effects_total.keys())
            g.set_xlim(2000, 12000)
            fig = g.get_figure()
            fig.savefig('par_{}.png'.format(par))
            fig.clf()
    return effects_total


effects_td = species_effect_tdeath(cluster_pars=clus_pars, perturbations=perturbations2,
                                   ftn=sig_apop, num_processors=4)
