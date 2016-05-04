from earm.lopez_embedded import model
from pysb.integrate import odesolve
import numpy as np
import csv
import pandas
import pysb.integrate
import pysb.util
import matplotlib.pyplot as plt
import os
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

# f = open('/home/carlos/Downloads/pars_embedded_911.txt')
# data = csv.reader(f)
# parames = []
# for i in data:parames.append(float(i[1]))
#
# epsilons = np.linspace(0, 4, 80)
# num_d = []
# for i in epsilons:
#     num_d.append(len(run_tropical(model,tspan,i,parames)[2]))
#
# plt.plot(epsilons, num_d)
# plt.xlabel('epsilon')
# plt.ylabel('passenger species')
# plt.show()
#

# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [model.parameters['Bid_0'].value,
              model.parameters['PARP_0'].value]

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


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


all_parameters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_5000')

clusters_path = listdir_fullpath('/home/oscar/Documents/tropical_project/parameters_in_cluster5000')

cluster_pars_path = {}
for sc in clusters_path:
    ff = open(sc)
    data_paths = csv.reader(ff)
    params_path = []
    for dd in data_paths: params_path.append(dd[0])
    cluster_pars_path[sc.split('0/')[1]] = params_path


def read_pars(par_path):
    f = open(par_path)
    data = csv.reader(f)
    param = []
    for d in data: param.append(float(d[1]))
    return param


def display_observables(params_estimated):
    fig, axApop = plt.subplots(figsize=(5.5, 5.5))
    # Construct matrix of experimental data and variance columns of interest
    exp_obs_norm = exp_data[data_names].view(float).reshape(len(exp_data), -1).T
    var_norm = exp_data[var_names].view(float).reshape(len(exp_data), -1).T
    std_norm = var_norm ** 0.5
    obs_names_disp = obs_names + ['aSmac']
    totals = obs_totals + [momp_obs_total]

    cparp_info = [0] * len(params_estimated)
    cparp_info_fraction = [0] * len(params_estimated)

    # Plot experimental data and simulation on the same axes
    colors = ('r', 'b')
    obs_range = [0, 1]
    axApop.vlines(momp_data[0], -0.05, 1.05, color='g', linestyle=':', label='aSmac')
    for exp, exp_err, obs, c in zip(exp_obs_norm, std_norm, obs_range, colors):

        axApop.plot(exp_data['Time'], exp, color=c, marker='.', linestyle=':', label=obs_names[obs])
        axApop.errorbar(exp_data['Time'], exp, yerr=exp_err, ecolor=c,
                        elinewidth=0.5, capsize=0, fmt=None)

        for idx, par in enumerate(params_estimated):
            params = read_pars(par)
            # params[62] -= params[62] * 0.89
            solver.run(params)
            sim_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
            sim_obs_norm = (sim_obs / totals).T
            cparp_info[idx] = curve_fit(sig_apop, solver.tspan, sim_obs_norm[1], p0=[100, 100, 100])[0]
            cparp_info_fraction[idx] = sim_obs_norm[1][-1]
            axApop.plot(solver.tspan, sim_obs_norm[obs], color=c, alpha=0.4)
            axApop.plot(solver.tspan, sim_obs_norm[2], color='g', alpha=0.4)

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
    xymax = np.max([np.max(np.fabs(solver.tspan)), np.max(np.fabs(1))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    weightsx = np.ones_like(column(cparp_info, 1)) / len(column(cparp_info, 1))
    weightsy = np.ones_like(cparp_info_fraction) / len(cparp_info_fraction)

    axHisty.hist(cparp_info_fraction, orientation='horizontal', bins=np.arange(0, 1.01 + binwidth, binwidth),
                 weights=weightsy)
    axHistx.hist(column(cparp_info, 1), bins=np.arange(min(solver.tspan), max(solver.tspan) + binwidthx, binwidthx),
                 weights=weightsx)

    # axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    axHistx.set_yticks([0, 0.5, 1])
    # axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 0.5, 1])
    axApop.legend(loc=0)
    fig.savefig('/home/oscar/Documents/tropical_project/all_parameters_earm.jpg', format='jpg', dpi=400)
    # return cparp_info
    # axApop.show()


# display(all_parameters_path)


def sig_apop(t, f, td, ts):
    return f - f / (1 + np.exp((t - td) / (4 * ts)))


def column(matrix, i):
    return [row[i] for row in matrix]


species_clusters_mode1 = {'sp1': ['clus1_sp1',
                               'clus2_sp1'],
                       'sp2': ['clus1_sp2',
                               'clus3_sp2'],
                       'sp5': ['clus1_sp5',
                               'clus2_sp5',
                               'clus3_sp5'],
                       'sp6': ['clus1_sp6',
                               'clus2_sp6',
                               'clus3_sp6']}

species_clusters_mode2 = {'sp1': ['clus1_sp1',
                               'clus2_sp1',
                               'clus3_sp1'],
                       'sp2': ['clus1_sp2',
                               'clus2_sp2',
                               'clus3_sp2'],
                       'sp5': ['clus1_sp5',
                               'clus2_sp5',
                               'clus3_sp5'],
                       'sp6': ['clus1_sp6',
                               'clus2_sp6',
                               'clus3_sp6']}


all_intersections = list(itertools.product(*species_clusters_mode1.values()))

# from tropicalize import run_tropical
# params = read_pars('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_4825.txt')
# run_tropical(model,tspan,params)

display_observables(all_parameters_path)


# for i in all_intersections:
#     c21 = set(cluster_pars_path[i[0]]).intersection(
#             cluster_pars_path[i[1]]).intersection(
#             cluster_pars_path[i[2]]).intersection(cluster_pars_path[i[3]])
#     if c21:
#         display(c21)

def display_all_species(cluster_parameters):
    for cl in cluster_parameters:
        directory = '/home/oscar/Documents/tropical_project/' + cl
        if not os.path.exists(directory):
            os.makedirs(directory)
        for sp in range(len(model.species)):
            plt.figure()
            for idx, par in enumerate(cluster_parameters[cl]):
                params = read_pars(par)
                solver.run(params)
                y = solver.y.T
                plt.plot(solver.tspan, y[sp])
            plt.title(str(model.species[sp]))
            plt.savefig(directory + '/' + str(model.species[sp]) + '.jpg', format='jpg', bbox_inches='tight', dpi=400)
            plt.close()

# display_all_species(cluster_pars_path)
