from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream.convergence import Gelman_Rubin
import numpy as np
import os
from pysb.simulator import ScipyOdeSimulator
from pydyno.examples.double_enzymatic.mm_two_paths_model import model
from scipy.stats import norm

directory = os.path.dirname(__file__)
avg_data_path = os.path.join(directory, 'product_data.npy')
sd_data_path = os.path.join(directory, 'exp_sd.npy')

exp_avg = np.load(avg_data_path)
exp_sd = np.load(sd_data_path)

tspan = np.linspace(0, 10, 51)

solver = ScipyOdeSimulator(model, tspan=tspan)

like_product = norm(loc=exp_avg, scale=exp_sd)

idx_pars_calibrate = [0, 1, 2, 3, 4, 5]
rates_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

param_values = np.array([p.value for p in model.parameters])

sampled_parameter_names = [SampledParam(norm, loc=np.log10(par), scale=2) for par in param_values[rates_mask]]

nchains = 5
niterations = 50000


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_mask] = 10 ** Y

    sim = solver.run(param_values=param_values).all
    logp_product = np.sum(like_product.logpdf(sim['Product']))

    if np.isnan(logp_product):
        logp_product = -np.inf

    return logp_product


if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood,
                                       niterations=niterations, nchains=nchains, multitry=True,
                                       gamma_levels=4, adapt_gamma=True, history_thin=1,
                                       model_name='denzyme_dreamzs_5chain', verbose=False)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('pydream_results/denzyme_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('pydream_results/denzyme_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

    #Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('pydream_results/denzyme_dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)

    old_samples = sampled_params
    if np.any(GR>1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations
            sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood,
                                               niterations=niterations, nchains=nchains, start=starts, multitry=True, gamma_levels=4,
                                               adapt_gamma=True, history_thin=1, model_name='denzyme_dreamzs_5chain',
                                               verbose=False, restart=True)


            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save('pydream_results/enzyme_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
                np.save('pydream_results/enzyme_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ',total_iterations,' GR = ',GR)
            np.savetxt('pydream_results/enzyme_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

            if np.all(GR<1.2):
                converged = True

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(old_samples[0])
        burnin = total_iterations/2
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
                                  old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(sampled_parameter_names)
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
            fig.savefig('pydream_results/PyDREAM_denzyme_dimension_'+str(dim))

    except ImportError:
        pass

else:

    run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':niterations, 'nchains':nchains, \
                  'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'denzyme_dreamzs_5chain', 'verbose':False}
