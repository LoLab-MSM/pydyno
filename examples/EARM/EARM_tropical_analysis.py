import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import pysb.integrate
from earm.lopez_embedded import model
from tropical import helper_functions as hf
import csv

# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']


# Load experimental data file
earm_path = '/home/oscar/PycharmProjects/earm'
data_path = os.path.join(earm_path, 'xpdata', 'forfits',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')
exp_data = np.genfromtxt(data_path, delimiter=',', names=True)

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_obs_total = model.parameters['Smac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

# Build time points for the integrator, using the same time scale as the
# experimental data but with greater resolution to help the integrator converge.
ntimes = len(exp_data['Time'])
# Factor by which to increase time resolution
tmul = 10
# Do the sampling such that the original experimental timepoints can be
# extracted with a slice expression instead of requiring interpolation.
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
                    (ntimes - 1) * tmul + 1)
# Initialize solver object
solver = pysb.integrate.Solver(model, tspan, rtol=1e-5, atol=1e-5)


def sig_apop(t, f, td, ts):
    """Return the amount of substrate cleaved at time t.

    Keyword arguments:
    t -- time
    f -- is the fraction cleaved at the end of the reaction
    td -- is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts -- is the switching time between initial and complete effector substrate  cleavage
    """
    return f - f / (1 + np.exp((t - td) / (4 * ts)))


def column(matrix, i):
    """Return the i column of a matrix

    Keyword arguments:
    matrix -- matrix to get the column from
    i -- column to get fro the matrix
    """
    return [row[i] for row in matrix]


def display_observables(params_estimated):
    """Save a figure of the observables with different parameter values and their distributions

        keyword arguments:
        params_estimated -- list of parameter sets
    """
    fig, axApop = plt.subplots(figsize=(5.5, 5.5))
    # Construct matrix of experimental data and variance columns of interest
    exp_obs_norm = exp_data[data_names].view(float).reshape(len(exp_data), -1).T
    var_norm = exp_data[var_names].view(float).reshape(len(exp_data), -1).T
    std_norm = var_norm ** 0.5
    obs_names_disp = obs_names + ['aSmac']


    cparp_info = [0] * len(params_estimated)
    cparp_info_fraction = [0] * len(params_estimated)

    # Plot experimental data and simulation on the same axes
    colors = ('r', 'b')
    obs_range = [0, 1]
    axApop.vlines(9810.0, -0.05, 1.05, color='g', linestyle=':', label='aSmac')
    for exp, exp_err, obs, c in zip(exp_obs_norm, std_norm, obs_range, colors):

        axApop.plot(exp_data['Time'], exp, color=c, marker='.', linestyle=':', label=obs_names[obs])
        axApop.errorbar(exp_data['Time'], exp, yerr=exp_err, ecolor=c,
                        elinewidth=0.5, capsize=0, fmt=None)

        for idx, par in enumerate(params_estimated):
            if type(par) == str:
                params = hf.read_pars(par)
            else:
                params = par

            # Total starting amounts of proteins in obs_names, for normalizing simulations
            obs_totals = [params[55],
                          params[23]]
            totals = obs_totals + [params[62]]

            # params[73] -= params[73] * 0.89
            solver.run(params)
            sim_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
            sim_obs_norm = (sim_obs / totals).T
            cparp_info[idx] = curve_fit(sig_apop, tspan, sim_obs_norm[1], p0=[100, 100, 100])[0]
            cparp_info_fraction[idx] = sim_obs_norm[1][-1]
            axApop.plot(tspan, sim_obs_norm[obs], color=c, alpha=0.4)
            axApop.plot(tspan, sim_obs_norm[2], color='g', alpha=0.4)

    plt.xticks(rotation=-30)
    plt.xlabel('Time')
    plt.ylabel('Fraction')
    fig.tight_layout()

    divider = make_axes_locatable(axApop)
    axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=axApop)
    axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=axApop)
    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
             visible=False)
    # now determine nice limits by hand:
    binwidth = 0.11
    binwidthx = 600
    xymax = np.max([np.max(np.fabs(tspan)), np.max(np.fabs(1))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    weightsx = np.ones_like(column(cparp_info, 1)) / len(column(cparp_info, 1))
    weightsy = np.ones_like(cparp_info_fraction) / len(cparp_info_fraction)

    axHisty.hist(cparp_info_fraction, orientation='horizontal', bins=np.arange(0, 1.01 + binwidth, binwidth),
                 weights=weightsy)
    axHistx.hist(column(cparp_info, 1), bins=np.arange(min(tspan), max(tspan) + binwidthx, binwidthx),
                 weights=weightsx)
    axHistx.vlines(9810.0, 0, 1, color='g', linestyle=':', label='aSmac')

    # axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 0.5, 1])
    # axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 0.5, 1])
    # axApop.legend(loc=0)
    fig.savefig('/home/oscar/Desktop/clusterBid1_earm.png', format='png', dpi=400)
    return

# Gets the parameter files and display the observables of the EARM model with the experimental values
# all_parameters_path = hf.listdir_fullpath('/home/oscar/home/oscar/tropical_project_new/parameters_5000')
# new_path = '/home/oscar/Documents/tropical_earm/parameters_5000'
# cluster2_pars = hf.list_pars_infile('/home/oscar/Documents/tropical_earm/clustered_parameters_bid_consumption/data_frame37_Type2',
#                                     new_path=new_path)

all_parameters = np.load('/home/oscar/Documents/tropical_earm_IC/IC_10000_parameters_consumption.npy')

path_cluster_type = '/home/oscar/Documents/tropical_earm_IC/bid_clusteredPAM_pars_consumption/data_frame37_Type1'
f = open(path_cluster_type)
data = csv.reader(f)
pars_idx = [int(d[0]) for d in data]

display_observables(all_parameters[pars_idx])



# clusters_path = hf.listdir_fullpath('/home/oscar/tropical_project_new/parameters_clusters')
#
# cluster_pars_path = {}
# for sc in clusters_path:
#     ff = open(sc)
#     data_paths = csv.reader(ff)
#     params_path = [dd[0] for dd in data_paths]
#     cluster_pars_path[sc.split('clusters/')[1]] = params_path
#
#
#
# def display_all_species(cluster_parameters):
#     """Saves figures of all species for each cluster of parameters
#
#         keyword arguments:
#         cluster_parameters -- list of files, where each file contains the clustered parameters
#     """
#     for cl in cluster_parameters:
#         directory = '/home/oscar/Documents/tropical_project/' + cl
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         for sp in range(len(model.species)):
#               plt.figure()
#             for idx, par in enumerate(cluster_parameters[cl]):
#                 params = hf.read_pars(par)
#                 solver.run(params)
#                 y = solver.y.T
#                 plt.plot(solver.tspan, y[sp])
#             plt.title(str(model.species[sp]))
#             plt.savefig(directory + '/' + str(model.species[sp]) + '.jpg', format='jpg', bbox_inches='tight', dpi=400)
#             plt.close()
#     return

# display_all_species(cluster_pars_path)

# species_clusters_mode1 = {'sp1': ['clus1_sp1',
#                                'clus2_sp1'],
#                        'sp2': ['clus1_sp2',
#                                'clus3_sp2'],
#                        'sp5': ['clus1_sp5',
#                                'clus2_sp5',
#                                'clus3_sp5'],
#                        'sp6': ['clus1_sp6',
#                                'clus2_sp6',
#                                'clus3_sp6']}
#
# species_clusters_mode2 = {'sp1': ['clus1_sp1',
#                                'clus2_sp1',
#                                'clus3_sp1'],
#                        'sp2': ['clus1_sp2',
#                                'clus2_sp2',
#                                'clus3_sp2'],
#                        'sp5': ['clus1_sp5',
#                                'clus2_sp5',
#                                'clus3_sp5'],
#                        'sp6': ['clus1_sp6',
#                                'clus2_sp6',
#                                'clus3_sp6']}
#
#
# all_intersections = list(itertools.product(*species_clusters_mode1.values()))
#
#
# for i in all_intersections:
#     c21 = set(cluster_pars_path[i[0]]).intersection(
#             cluster_pars_path[i[1]]).intersection(
#             cluster_pars_path[i[2]]).intersection(cluster_pars_path[i[3]])
#     if c21:
#         display(c21)

