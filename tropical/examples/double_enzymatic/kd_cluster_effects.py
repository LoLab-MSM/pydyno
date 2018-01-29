import matplotlib
matplotlib.use('agg')
from pysb.simulator import ScipyOdeSimulator
from mm_two_paths_model import model
import numpy as np
import pickle
from multiprocessing import Pool
from collections import OrderedDict
import seaborn as sns
from tropical import clustering
tspan = np.linspace(0,400, 400)
calibrated_pars = np.load('calibrated_pars.npy')

with open('signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

# Obtaining signature of how the species 0 (the enzyme) is being consumend
sp0_sign_reactants = all_signatures[0]['consumption']

# Initializing clustering object
clus = clustering.ClusterSequences(sp0_sign_reactants, unique_sequences=False)
clus.diss_matrix()
clus.agglomerative_clustering(2)
# pl = clustering.PlotSequences(clus)
# pl.all_trajectories_plot()
clusters_sp = clus.labels

num_of_clusters = set(clusters_sp)
clus_pars = {}
for j in num_of_clusters:
    item_index = np.where(clusters_sp == j)
    clus_pars[j] = calibrated_pars[item_index]

perturbations2 = {0: 0.2, 1: 0.2}

solver = ScipyOdeSimulator(model, tspan=tspan)


def run_one(param_values):
    sim = solver.run(param_values=param_values).all
    return sim


def species_effect_tdeath(cluster_pars, perturbations, violinplot=True, num_processors=4):
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
            p_total = [sim['__s5'][-1] for sim in sims]
            print (clus)
            print (par_idx)
            print (p_total)
            effects_data[par_idx] = p_total
            # std = np.std(td_hist)
            # mean = np.average(td_hist)
            # effects.loc[str(par_idx), str(clus)] = (mean, std)
        effects_total[clus] = effects_data
    if violinplot:
        for par in perturbations.keys():
            par_data = [td[par] for td in effects_total.values()]
            g = sns.violinplot(data=par_data, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            g.set_yticklabels(effects_total.keys())
            g.set_ylabel('Cluster index')
            g.set_xlabel('Product concentration')
            fig = g.get_figure()
            fig.savefig('par_{}.png'.format(par))
            fig.clf()
    return effects_total


effects_td = species_effect_tdeath(cluster_pars=clus_pars, perturbations=perturbations2, num_processors=4)
