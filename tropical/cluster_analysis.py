from __future__ import division

import collections
import colorsys
import csv
import numbers
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tropical.util as hf
from pysb.bng import generate_equations
from pysb.pattern import SpeciesPatternMatcher, ReactionPatternMatcher
import sympy
from tropical.distinct_colors import distinct_colors
import matplotlib.patches as mpatches
from math import e

plt.ioff()


class AnalysisCluster(object):
    """
    Class to visualize species trajectories and parameter distributions in different clusters

    Parameters
    ----------
    model: pysb.Model
        Model passed to the constructor
    clusters: vector-like or str or None
        Indices of the parameters that belong to an specific cluster. It can be a list of files that contain
        the indices of each cluster, a list of lists where each list has the parameter indices of a cluster or
        a file that contains the cluster labels to which each parameter belongs to, or None if the user want to
        analyse the sim_results as a single cluster.
    sim_results: SimulationResult or h5 file from PySB simulation
        SimulationResult object or h5 file with the dynamic solutions of the model for all the parameter sets
    """
    def __init__(self, model, sim_results, clusters):

        self.model = model
        generate_equations(model)
        # Check simulation results
        self.all_simulations, self.all_parameters, nsims, self.tspan = hf.get_simulations(sim_results)

        if clusters is not None:
            # Check clusters
            self.clusters, self.number_pars = self.check_clusters_arg(clusters)
        else:
            no_clusters = {0: range(len(self.all_parameters))}
            self.clusters = no_clusters
            self.number_pars = len(self.all_parameters)

    @staticmethod
    def check_clusters_arg(clusters):  # check clusters
        if isinstance(clusters, collections.Iterable):
            # check if clusters is a list of files containing the indices or idx of the IC that belong to that cluster
            if all(os.path.isfile(str(item)) for item in clusters):
                clus_values = {}
                number_pars = 0
                for i, clus in enumerate(clusters):
                    f = open(clus)
                    data = csv.reader(f)
                    pars_idx = [int(d[0]) for d in data]
                    clus_values[i] = pars_idx
                    number_pars += len(pars_idx)
                # self.clusters is a dictionary that contains the index of the parameter values that belong to different
                # clusters
                clusters = clus_values
                number_pars = number_pars
                return clusters, number_pars
            elif all(isinstance(item, numbers.Number) for item in clusters):
                if not isinstance(clusters, np.ndarray):
                    clusters = np.array(clusters)
                pars_clusters = clusters
                num_of_clusters = set(pars_clusters)
                clus_values = {}
                for j in num_of_clusters:
                    item_index = np.where(pars_clusters == j)
                    clus_values[j] = item_index[0].tolist()
                clusters = clus_values
                number_pars = len(pars_clusters)
                return clusters, number_pars
            else:
                raise ValueError('Mixed formats is not supported')
        # check is clusters is a file that contains the indices of the clusters for each parameter set
        elif isinstance(clusters, str):
            if os.path.isfile(clusters):
                f = open(clusters)
                data = csv.reader(f)
                pars_clusters = np.array([int(d[0]) for d in data])
                num_of_clusters = set(pars_clusters)
                clus_values = {}
                for j in num_of_clusters:
                    item_index = np.where(pars_clusters == j)
                    clus_values[j] = item_index[0].tolist()
                clusters = clus_values
                number_pars = len(pars_clusters)
                return clusters, number_pars
        else:
            raise TypeError('cluster data structure not supported')

    def plot_dynamics_cluster_types(self, species, save_path='', fig_name='', species_ftn_fit=None, norm=False,
                                    **kwargs):
        """
        Plots the dynamics of the species for each cluster

        Parameters
        ----------
        species: list-like
            Indices of PySB species that will be plotted or observable names or pysb expressions
        save_path: str
            Path to folder where the figure is going to be saved
        fig_name: str
            String used to give a name to the cluster figures
        species_ftn_fit: dict, optional
            Dictionary of species with their respective function to fit their dynamics
        norm: boolean, optional
            Normalizes species by max value in simulation
        kwargs: dict
            Arguments to pass to fitting function

        Returns
        -------

        """

        # creates a dictionary to store the different figures by cluster
        plots_dict = {}
        for sp in species:
            for clus in self.clusters:
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)] = plt.subplots()

        if norm:
            if species_ftn_fit:
                # checking if species_to_fit are present in the species that are going to be plotted
                self._plot_dynamics_cluster_types_norm_ftn_species(plots_dict=plots_dict, species=species,
                                                                   species_ftn_fit=species_ftn_fit,
                                                                   save_path=save_path, fig_label=fig_name,
                                                                   **kwargs)

            else:
                self._plot_dynamics_cluster_types_norm(plots_dict=plots_dict, species=species,
                                                       save_path=save_path, fig_label=fig_name)

        else:
            self._plot_dynamics_cluster_types(plots_dict=plots_dict, species=species,
                                              save_path=save_path, fig_label=fig_name)

        return

    def _plot_dynamics_cluster_types(self, plots_dict, species, save_path, fig_label):
        for idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            for sp in species:
                # Calculate reaction rate expression
                if isinstance(sp, sympy.Expr):
                    expr_vars = [atom for atom in sp.atoms(sympy.Symbol)]
                    expr_args = [0] * len(expr_vars)
                    for va_idx, va in enumerate(expr_vars):
                        if str(va).startswith('__'):
                            sp_idx = int(''.join(filter(str.isdigit, str(va))))
                            expr_args[va_idx] = y[:, :, sp_idx].T
                        else:
                            par_idx = self.model.parameters.index(va)
                            expr_args[va_idx] = self.all_parameters[clus][:, par_idx]
                    f_expr = sympy.lambdify(expr_vars, sp)
                    sp_trajectory = f_expr(*expr_args)
                    name = 'expr'

                # Calculate observable
                elif isinstance(sp, str):
                    sp_trajectory = self._get_observable(sp, y)
                    name = sp
                else:
                    sp_trajectory = y[:, :, sp].T
                    name = self.model.species[sp]
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan, sp_trajectory,
                                                                            color='blue',
                                                                            alpha=0.2)

                ax = plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1]
                divider = make_axes_locatable(ax)
                # axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
                axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
                plt.setp(axHisty.get_yticklabels(), visible=False)
                hist_data = sp_trajectory[-1, :]
                axHisty.hist(hist_data, normed=True, orientation='horizontal')
                shape = np.std(hist_data)
                scale = np.average(hist_data)

                pdf_pars = r'$\sigma$ =' + str(round(shape, 2)) + '\n' r'$\mu$ =' + str(round(scale, 2))
                anchored_text = AnchoredText(pdf_pars, loc=1, prop=dict(size=10))
                axHisty.add_artist(anchored_text)
                axHisty.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

                sp_max_conc = np.amax(sp_trajectory)
                sp_min_conc = np.amin(sp_trajectory)
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_xlabel('Time')
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_ylabel('Concentration')
                # plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_xlim([0, 8])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_ylim([sp_min_conc, sp_max_conc])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][0].suptitle('{0}, cluster {1}'.
                                                                                format(name, idx))
                final_save_path = os.path.join(save_path, 'plot_sp{0}_cluster{1}_{2}'.format(sp, idx, fig_label))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][0].savefig(final_save_path + '.png',
                                                                               format='png', dpi=700)

    def _plot_dynamics_cluster_types_norm(self, plots_dict, species, save_path, fig_label):
        for idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            for sp in species:
                # Calculate reaction rate expression
                if isinstance(sp, sympy.Expr):
                    expr_vars = [atom for atom in sp.atoms(sympy.Symbol)]
                    expr_args = [0] * len(expr_vars)
                    for va_idx, va in enumerate(expr_vars):
                        if str(va).startswith('__'):
                            sp_idx = int(''.join(filter(str.isdigit, str(va))))
                            expr_args[va_idx] = y[:, :, sp_idx].T
                        else:
                            par_idx = self.model.parameters.index(va)
                            expr_args[va_idx] = self.all_parameters[clus][:, par_idx]
                    f_expr = sympy.lambdify(expr_vars, sp)
                    sp_trajectory = f_expr(*expr_args)
                    name = 'expr'

                # Calculate observable
                elif isinstance(sp, str):
                    sp_trajectory = self._get_observable(sp, y)
                    name = sp
                else:
                    sp_trajectory = y[:, :, sp].T
                    name = self.model.species[sp]
                norm_trajectories = np.divide(sp_trajectory, np.amax(sp_trajectory, axis=0))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan,
                                                                            norm_trajectories,
                                                                            color='blue',
                                                                            alpha=0.2)

                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_xlabel('Time')
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_ylabel('Concentration')
                # plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_xlim([0, 8])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_ylim([0, 1])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][0].suptitle('{0}, cluster {1}'.
                                                                                format(name, idx))
                final_save_path = os.path.join(save_path, 'plot_sp{0}_cluster{1}_normed_{2}'.format(sp, idx, fig_label))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][0].savefig(final_save_path + '.png',
                                                                               format='png', dpi=700)

    def _plot_dynamics_cluster_types_norm_ftn_species(self, plots_dict, species, species_ftn_fit,
                                                      save_path, fig_label, **kwargs):
        sp_overlap = [ii for ii in species_ftn_fit if ii in species]
        if not sp_overlap:
            raise ValueError('species_to_fit must be in species list')

        for idx, clus in self.clusters.items():
            ftn_result = {}
            y = self.all_simulations[clus]
            for sp in species:
                # Calculate reaction rate expression
                if isinstance(sp, sympy.Expr):
                    expr_vars = [atom for atom in sp.atoms(sympy.Symbol)]
                    expr_args = [0] * len(expr_vars)
                    for va_idx, va in enumerate(expr_vars):
                        if str(va).startswith('__'):
                            sp_idx = int(''.join(filter(str.isdigit, str(va))))
                            expr_args[va_idx] = y[:, :, sp_idx].T
                        else:
                            par_idx = self.model.parameters.index(va)
                            expr_args[va_idx] = self.all_parameters[clus][:, par_idx]
                    f_expr = sympy.lambdify(expr_vars, sp)
                    sp_trajectory = f_expr(*expr_args)
                    name = 'expr'

                # Calculate observable
                elif isinstance(sp, str):
                    sp_trajectory = self._get_observable(sp, y)
                    name = sp
                else:
                    sp_trajectory = y[:, :, sp].T
                    name = self.model.species[sp]
                norm_trajectories = np.divide(sp_trajectory, np.amax(sp_trajectory, axis=1))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan,
                                                                            norm_trajectories,
                                                                            color='blue',
                                                                            alpha=0.2)
                if sp in sp_overlap:
                    result_fit = hf.curve_fit_ftn(fn=species_ftn_fit[sp], xdata=self.tspan,
                                                  ydata=sp_trajectory, **kwargs)
                    ftn_result[sp] = result_fit
            self._add_function_hist(plots_dict=plots_dict, idx=idx, sp_overlap=sp_overlap, ftn_result=ftn_result)

            for sp in species:
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_xlabel('Time')
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_ylabel('Concentration')
                # plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_xlim([0, 8])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].set_ylim([0, 1])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][0].suptitle('{0}, cluster {1}'.
                                                                                format(name, idx))
                final_save_path = os.path.join(save_path, 'plot_sp{0}_cluster{1}_fitted_{2}'.format(sp, idx, fig_label))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][0].savefig(final_save_path + '.png',
                                                                               format='png', dpi=700)

    def _add_function_hist(self, plots_dict, idx, sp_overlap, ftn_result):
        for sp_dist in sp_overlap:
            ax = plots_dict['plot_sp{0}_cluster{1}'.format(sp_dist, idx)][1]
            divider = make_axes_locatable(ax)
            axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
            # axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
            plt.setp(axHistx.get_xticklabels(),
                     visible=False)  # + axHisty.get_yticklabels(), visible=False)

            # This is specific for the time of death fitting in apoptosis
            hist_data = hf.column(ftn_result[sp_dist], 1)
            # TODO I should look deeper into how many trajectories have different dynamics
            hist_data_filt = hist_data[(hist_data > 0) & (hist_data < self.tspan[-1])]
            # shape, loc, scale = lognorm.fit(hist_data_filt, floc=0)
            # pdf = lognorm.pdf(np.sort(hist_data_filt), shape, loc, scale)
            shape = np.std(hist_data_filt)
            scale = np.average(hist_data_filt)

            pdf_pars = r'$\sigma$ =' + str(round(shape, 2)) + '\n' r'$\mu$ =' + str(round(scale, 2))
            anchored_text = AnchoredText(pdf_pars, loc=1, prop=dict(size=12))
            axHistx.add_artist(anchored_text)
            axHistx.hist(hist_data_filt, normed=True, bins=20)
            axHistx.vlines(10230.96, -0.05, 1.05, color='r', linestyle=':', linewidth=2)  # MOMP data
            # axHistx.plot(np.sort(hist_data_filt), pdf) # log fitting to histogram data
            for tl in axHistx.get_xticklabels():
                tl.set_visible(False)
            # yticks = [v for v in np.linspace(0, pdf.max(), 3)]
            axHistx.set_ylim(0, 1.5e-3)
            axHistx.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    def _get_observable(self, obs, y):
        """

        Parameters
        ----------
        obs: str
            pysb observable name
        y: numpy array
            Simulations from which the observable is going to be calculated

        Returns
        -------

        """
        obs_names = [ob.name for ob in self.model.observables]
        try:
            obs_idx = obs_names.index(obs)
        except ValueError:
            raise ValueError(obs + "doesn't exist in the model")
        sps = self.model.observables[obs_idx].species
        obs_values = np.sum(y[:, :, sps], axis=2)
        return obs_values.T

    def hist_plot_clusters(self, par_idxs, save_path=''):
        """
        Creates a plot for each cluster, and it has histograms of the parameters provided

        Parameters
        ----------
        par_idxs: list-like
            Indices of the model parameters that would be visualized
        save_path: str
            Path to where the file is going to be saved

        Returns
        -------

        """

        colors = self._get_colors(len(par_idxs))
        plt.figure(1)
        for c_idx, clus in self.clusters.items():
            cluster_pars = self.all_parameters[clus]
            sp_ic_all = [0] * len(par_idxs)
            sp_weights_all = [0] * len(par_idxs)
            labels = [0] * len(par_idxs)
            for idx, sp_ic in enumerate(par_idxs):
                sp_ic_values = cluster_pars[:, sp_ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values)
                sp_ic_all[idx] = sp_ic_values
                sp_weights_all[idx] = sp_ic_weights
                labels[idx] = self.model.parameters[sp_ic].name
            plt.hist(sp_ic_all, weights=sp_weights_all, alpha=0.4, color=colors, label=labels)
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            plt.legend(loc=0)
            final_save_path = os.path.join(save_path, 'hist_ic_type{0}'.format(c_idx))
            plt.savefig(final_save_path + '.png', format='png', dpi=700)
            plt.clf()
        return

    def hist_avg_sps(self, pattern, type_fig='bar', y_lim=(0, 1), save_path='', fig_name=''):
        """
        Creates a plot for each cluster. It has a stacked bar of the percentage of the interactions
        of species at each time point
        Parameters
        ----------
        pattern: pysb.Monomer or pysb.MonomerPattern or pysb.ComplexPattern
        y_lim: tuple
            y-axis limits
        fig_name: str
            Figure name


        Returns
        -------

        """
        # Matching species to pattern
        spm = SpeciesPatternMatcher(self.model)
        sps_matched = spm.match(pattern, index=True)
        colors = distinct_colors(len(sps_matched))

        for c_idx, clus in self.clusters.items():
            y_offset = np.zeros(len(self.tspan))
            y = self.all_simulations[clus]
            sps_total = np.sum(y[:, :, sps_matched], axis=(0, 2)) /(len(clus))
            sps_avg = np.zeros((len(sps_matched), len(self.tspan)))
            for idx, sp, in enumerate(sps_matched):
                sp_pctge = np.sum(y[:, :, sp], axis=0)/(sps_total*len(clus))
                sps_avg[idx] = sp_pctge

            if type_fig == 'bar':
                for sps, col in enumerate(colors):
                    sp_pctge = sps_avg[sps]
                    plt.bar(self.tspan, sp_pctge, color=col, bottom=y_offset, width=1)
                    y_offset = y_offset + sp_pctge

                plt.xlabel('Time')
                plt.ylabel('Percentage')
                plt.suptitle('Cluster {0}'.format(c_idx))
                plt.ylim(y_lim)
                plt.legend(sps_matched, loc='lower center', bbox_to_anchor=(0.50, -0.4), ncol=5,
                           title='Species indices')

            elif type_fig == 'entropy':
                entropies = [0] * len(self.tspan)
                for t_idx in range(len(self.tspan)):
                    tp_pctge = sps_avg[:, t_idx]
                    entropy = self._get_probs_entropy(tp_pctge)
                    entropies[t_idx] = entropy

                plt.plot(self.tspan, entropies)
                plt.xlabel('Time')
                plt.ylabel('Entropy')

            else:
                raise NotImplementedError('Type of visualization not implemented')

            final_save_path = os.path.join(save_path, 'hist_avg_clus{0}_{1}'.format(c_idx, fig_name))
            plt.savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')
            plt.clf()

    def hist_avg_rxns(self, pattern, type_fig='bar', y_lim=(0,1), save_path='', fig_name=''):
        """

        Parameters
        ----------
        pattern: pysb.Monomer or pysb.MonomerPattern or pysb.ComplexPattern
        y_lim: tuple
            y-axis limits
        save_path: str
        fig_name: str
            Figure name

        Returns
        -------

        """
        # Matching reactions to pattern
        rpm = ReactionPatternMatcher(self.model)
        products_matched = rpm.match_products(pattern)
        reactants_matched = rpm.match_reactants(pattern)
        products_avg = np.zeros((len(products_matched), len(self.tspan)))
        reactants_avg = np.zeros((len(reactants_matched), len(self.tspan)))
        pcolors = distinct_colors(len(products_matched))
        rcolors = distinct_colors(len(reactants_matched))

        plegend_patches = []
        plabels = []
        rlegend_patches = []
        rlabels = []

        for c_idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            y_poffset = np.zeros(len(self.tspan))
            y_roffset = np.zeros(len(self.tspan))

            pars = self.all_parameters[clus]
            for rxn_idx, rxn in enumerate(products_matched):
                rate = rxn.rate
                var = [atom for atom in rate.atoms(sympy.Symbol)]
                arg = [0] * len(var)
                for idx, va in enumerate(var):
                    if str(va).startswith('__'):
                        sp_idx = int(''.join(filter(str.isdigit, str(va))))
                        arg[idx] = y[:, :, sp_idx]
                    else:
                        arg[idx] = pars[:, self.model.parameters.index(va)][0]
                        # print (pars[:, self.model.parameters.index(va)])
                f = sympy.lambdify(var, rate)
                values = f(*arg)
                for col in range(values.shape[1]):
                    if (values[:, col] < 0).all():
                        values[:, col] = 0
                values_avg = np.average(values, axis=0, weights=values >= 0)
                products_avg[rxn_idx] = values_avg

                # Creating labels
                plabel = hf.rate_2_interactions(self.model, str(rate))
                plabels.append(plabel)
                plegend_patches.append(mpatches.Patch(color=pcolors[rxn_idx], label=plabel))

            ptotals = np.sum(products_avg, axis=0)
            products_avg = products_avg / ptotals

            for rct_idx, rct in enumerate(reactants_matched):
                rate = rct.rate
                var = [atom for atom in rate.atoms(sympy.Symbol)]
                arg = [0] * len(var)
                for idx, va in enumerate(var):
                    if str(va).startswith('__'):
                        sp_idx = int(''.join(filter(str.isdigit, str(va))))
                        arg[idx] = y[:, :, sp_idx]
                    else:
                        arg[idx] = pars[:, self.model.parameters.index(va)][0]

                f = sympy.lambdify(var, rate)
                values = f(*arg)
                for col in range(values.shape[1]):
                    if (values[:, col] > 0).all():
                        values[:, col] = 0
                values_avg = np.average(values, axis=0, weights=values <= 0)
                reactants_avg[rct_idx] = values_avg

                # Creating labels
                rlabel = hf.rate_2_interactions(self.model, str(rate))
                rlabels.append(rlabel)
                rlegend_patches.append(mpatches.Patch(color=rcolors[rct_idx], label=rlabel))

            rtotals = np.sum(reactants_avg, axis=0)
            reactants_avg = reactants_avg/rtotals

            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            if type_fig == 'bar':
                for prxn, pcol in zip(range(len(products_matched)), pcolors):
                    sp_pctge = products_avg[prxn]
                    ax1.bar(self.tspan, sp_pctge, color=pcol, bottom=y_poffset, width=1)
                    y_poffset = y_poffset + sp_pctge

                for rrxn, rcol in zip(range(len(reactants_matched)), rcolors):
                    sp_pctge = reactants_avg[rrxn]
                    ax2.bar(self.tspan, sp_pctge, color=rcol, bottom=y_roffset, width=1)
                    y_roffset = y_roffset + sp_pctge

                ax1.set(title='Reactions distributions', ylabel='Percentage')
                ax2.set(xlabel='Time', ylabel='Percentage')

                fig_plegend = plt.figure(figsize=(2, 1.25))
                fig_plegend.legend(plegend_patches, plabels, loc='center', frameon=False, ncol=4)
                plt.savefig('plegends_{0}.pdf'.format(fig_name), format='pdf', bbox_inches='tight')

                fig_rlegend = plt.figure(figsize=(2, 1.25))
                fig_rlegend.legend(rlegend_patches, rlabels, loc='center', frameon=False, ncol=4)
                plt.savefig('rlegends{0}.pdf'.format(fig_name), format='pdf', bbox_inches='tight')

            elif type_fig == 'entropy':
                pentropies = [0] * len(self.tspan)
                rentropies = [0] * len(self.tspan)
                for t_idx in range(len(self.tspan)):
                    tp_pctge = products_avg[:, t_idx]
                    pentropy = hf.get_probs_entropy(tp_pctge)
                    pentropies[t_idx] = pentropy

                    tr_pctge = reactants_avg[:, t_idx]
                    rentropy = hf.get_probs_entropy(tr_pctge)
                    rentropies[t_idx] = rentropy

                ax1.plot(self.tspan, pentropies)
                ax2.plot(self.tspan, rentropies)

                ax1.set(title='Reactions entropies', ylabel='Entropy')
                ax2.set(xlabel='Time', ylabel='Entropy')

            else:
                raise NotImplementedError('Type of visualization not implemented')

            final_save_path = os.path.join(save_path, 'hist_avg_rxn_clus{0}_{1}'.format(c_idx, fig_name))
            fig.savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')

    def violin_plot_sps(self, par_idxs, save_path=''):
        """
        Creates a plot for each model parameter. Then, makes violin plots for each cluster

        Parameters
        ----------
        par_idxs: list-like
            Indices of the parameters that would be visualized
        save_path: str
            Path to where the file is going to be saved

        Returns
        -------

        """

        for sp_ic in par_idxs:
            plt.figure()
            data_violin = [0] * len(self.clusters)
            clus_labels = [0] * len(self.clusters)
            count = 0
            for idx, clus in self.clusters.items():
                cluster_pars = self.all_parameters[clus]
                sp_ic_values = cluster_pars[:, sp_ic]
                data_violin[count] = np.log10(sp_ic_values)
                clus_labels[count] = idx
                count += 1

            g = sns.violinplot(data=data_violin, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            g.set_yticklabels(clus_labels)
            plt.xlabel('Parameter Range')
            plt.ylabel('Clusters')
            plt.suptitle('Parameter {0}'.format(self.model.parameters[sp_ic].name))
            final_save_path = os.path.join(save_path, 'violin_sp_{0}'.format(self.model.parameters[sp_ic].name))
            plt.savefig(final_save_path + '.png', format='png', dpi=700)
        return

    def violin_plot_kd(self, par_idxs, save_path=''):
        """
        Creates a plot for each kd parameter. Then, makes violin plots for each cluster

        Parameters
        ----------
        par_idxs: list-like
            Tuples of parameters indices, where the first entry is the k_reverse and
            the second entry is the k_forward parameter.
        save_path: str
            Path to where the file is going to be saved

        Returns
        -------

        """
        for kd_pars in par_idxs:
            plt.figure()
            data_violin = [0] * len(self.clusters)
            clus_labels = [0] * len(self.clusters)
            count = 0
            for idx, clus in self.clusters.items():
                cluster_pars = self.all_parameters[clus]
                kr_values = cluster_pars[:, kd_pars[0]]
                kf_values = cluster_pars[:, kd_pars[1]]
                data_violin[count] = np.log10(kr_values / kf_values)
                clus_labels[count] = idx
                count += 1

            g = sns.violinplot(data=data_violin, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            g.set_yticklabels(clus_labels)
            plt.xlabel('Parameter Range')
            plt.ylabel('Clusters')
            plt.suptitle('Parameter {0}'.format(self.model.parameters[kd_pars[0]].name))
            final_save_path = os.path.join(save_path, 'violin_sp_{0}_kd'.format(self.model.parameters[kd_pars[0]].name))
            plt.savefig(final_save_path + '.png', format='png', dpi=700)
        return

    def plot_sp_ic_overlap(self, par_idxs, save_path=''):
        """
        Creates a stacked histogram with the distributions of each of the
        clusters for each model parameter provided

        Parameters
        ----------
        par_idxs: list
            Indices of the initial conditions in model.parameter to plot
        save_path: str
            Path to where the file is going to be saved

        Returns
        -------

        """

        if type(par_idxs) == int:
            par_idxs = [par_idxs]

        for ic in par_idxs:
            plt.figure()
            sp_ic_values_all = self.all_parameters[:, ic]
            sp_ic_weights_all = np.ones_like(sp_ic_values_all) / len(sp_ic_values_all)
            n, bins, patches = plt.hist(sp_ic_values_all, weights=sp_ic_weights_all, bins=30, fill=False)

            cluster_ic_values = []
            cluster_ic_weights = []
            for clus in self.clusters.values():
                cluster_pars = self.all_parameters[clus]
                sp_ic_values = cluster_pars[:, ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values_all)
                cluster_ic_values.append(sp_ic_values)
                cluster_ic_weights.append(sp_ic_weights)

            label = ['cluster_{0}, {1}%'.format(cl, (len(self.clusters[cl]) / self.number_pars) * 100)
                     for cl in self.clusters.keys()]
            plt.hist(cluster_ic_values, bins=bins, weights=cluster_ic_weights, stacked=True, label=label,
                     histtype='bar', ec='black')
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            plt.title(self.model.parameters[ic].name)
            plt.legend(loc=0)

            final_save_path = os.path.join(save_path, 'plot_ic_overlap_{0}'.format(ic))
            plt.savefig(final_save_path + '.png', format='png', dpi=700)
        return

    def scatter_plot_pars(self, par_idxs, cluster, save_path=''):
        """

        Parameters
        ----------
        par_idxs: list
            Indices of the parameters to visualize
        cluster: list-like
        save_path: str
            Path to where the file is going to be saved


        Returns
        -------

        """
        if isinstance(cluster, int):
            cluster_idxs = self.clusters[cluster]
        elif isinstance(cluster, collections.Iterable):
            cluster_idxs = cluster
        else:
            raise TypeError('format not supported')

        sp_ic_values1 = self.all_parameters[cluster_idxs, par_idxs[0]]
        sp_ic_values2 = self.all_parameters[cluster_idxs, par_idxs[1]]
        plt.figure()
        plt.scatter(sp_ic_values1, sp_ic_values2)
        ic_name0 = self.model.parameters[par_idxs[0]].name
        ic_name1 = self.model.parameters[par_idxs[1]].name
        plt.xlabel(ic_name0)
        plt.ylabel(ic_name1)
        final_save_path = os.path.join(save_path, 'scatter_{0}_{1}_cluster_{2}'.format(ic_name0, ic_name1,
                                                                                       cluster))
        plt.savefig(final_save_path + '.png', format='png', dpi=700)

    @staticmethod
    def _get_colors(num_colors):
        colors = []
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i / 360.
            lightness = (50 + np.random.rand() * 10) / 100.
            saturation = (90 + np.random.rand() * 10) / 100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors

