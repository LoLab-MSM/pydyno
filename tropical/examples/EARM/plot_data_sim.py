import os
import matplotlib.pyplot as plt
import numpy as np
import pysb.integrate
from earm.lopez_embedded import model
import tropical.helper_functions as hf

# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']

obs_totals = [model.parameters['Bid_0'].value,
              model.parameters['PARP_0'].value]
# Load experimental data file
earm_path = '/Users/dionisio/PycharmProjects/earm'
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


def display(param_values):

    exp_obs_norm = exp_data[data_names].view(float).reshape(len(exp_data), -1).T
    var_norm = exp_data[var_names].view(float).reshape(len(exp_data), -1).T
    std_norm = var_norm ** 0.5
    solver.run(param_values)
    obs_names_disp = obs_names + ['aSmac']
    for i in solver.yobs['aSmac']: print (i)
    sim_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
    totals = obs_totals + [momp_obs_total]
    sim_obs_norm = (sim_obs / totals).T
    colors = ('r', 'b')
    for exp, exp_err, sim, c in zip(exp_obs_norm, std_norm, sim_obs_norm, colors):
        plt.scatter(exp_data['Time'], exp, color=c, marker='.', linestyle=':')
        plt.errorbar(exp_data['Time'], exp, yerr=exp_err, ecolor=c,
                     elinewidth=1.0, capsize=0, fmt=None)
        plt.plot(tspan, sim, color=c, linewidth=1.8)
    plt.plot(tspan, sim_obs_norm[2], color='g', linewidth=1.5)
    plt.vlines(momp_data[0], -0.05, 1.05, color='g', linestyle=':')
    plt.xlabel('Time(s)')
    plt.ylabel('Population')
    plt.savefig('earm_trained.png', format='png', dpi=1000)
    plt.show()
    return

# directory = os.path.dirname(__file__)
# parameters_path = os.path.join(directory, "parameters_5000")
# all_parameters = hf.listdir_fullpath(parameters_path)
# parameters = hf.read_pars(all_parameters[500])
parameters = np.load('calibrated_6572pars.npy')
display(parameters[70])