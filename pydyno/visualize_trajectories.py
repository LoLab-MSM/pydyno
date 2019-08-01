import collections
import colorsys
import csv
import numbers
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pydyno.util as hf
from pysb.bng import generate_equations
from pysb.pattern import SpeciesPatternMatcher, ReactionPatternMatcher
import sympy
from pydyno.distinct_colors import distinct_colors
import matplotlib.patches as mpatches

plt.ioff()


class VisualizeTrajectories(object):
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

        self._model = model
        generate_equations(model)
        # Check simulation results
        self._all_simulations, self._all_parameters, self._nsims, self._tspan = hf.get_simulations(sim_results)

        if clusters is None:
            no_clusters = {0: range(len(self.all_parameters))}
            self._clusters = no_clusters
        else:
            # Check clusters
            self._clusters = self.check_clusters_arg(clusters, self.nsims)

    @property
    def model(self):
        return self._model

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, new_clusters):
        self._clusters = self.check_clusters_arg(new_clusters, self.nsims)

    @property
    def all_simulations(self):
        return self._all_simulations

    @property
    def all_parameters(self):
        return self._all_parameters

    @property
    def nsims(self):
        return self._nsims

    @property
    def tspan(self):
        return self._tspan

    @staticmethod
    def check_clusters_arg(clusters, nsims):  # check clusters
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
                if number_pars != nsims:
                    raise ValueError('The number of cluster indices must have the same'
                                     'length as the number of simulations')
                return clusters
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
                if number_pars != nsims:
                    raise ValueError('The number of cluster indices must have the same'
                                     'length as the number of simulations')
                return clusters
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
                if number_pars != nsims:
                    raise ValueError('The number of cluster indices must have the same'
                                     'length as the number of simulations')
                return clusters
        else:
            raise TypeError('cluster data structure not supported')

    def plot_cluster_dynamics(self, species, save_path='', fig_name='',
                              species_ftn_fit=None, norm=False, norm_value=None,
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
                                                       save_path=save_path, fig_label=fig_name, norm_value=norm_value)

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

    def _plot_dynamics_cluster_types_norm(self, plots_dict, species, save_path, fig_label, norm_value=None):
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
                if norm_value:
                    norm_trajectories = sp_trajectory/norm_value
                else:
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

    def hist_clusters_parameters(self, par_idxs, save_path=''):
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

    def plot_pattern_sps_distribution(self, pattern, type_fig='bar', y_lim=(0, 1), save_path='', fig_name=''):
        """
        Creates a plot for each cluster. It has a stacked bar or entropy plot of the percentage
        of the interactions of species at each time point
        Parameters
        ----------
        pattern: pysb.Monomer or pysb.MonomerPattern or pysb.ComplexPattern
        type_fig: str
            `bar` to get a stacked bar of the distribution of the species that match the provided pattern.
            `entropy` to get the entropy of the distributions of the species that match the provided pattern
        y_lim: tuple
            y-axis limits
        save_path: path to save the file
        fig_name: str
            Figure name


        Returns
        -------

        """
        # Matching species to pattern
        spm = SpeciesPatternMatcher(self.model)
        sps_matched = spm.match(pattern, index=True)
        colors = distinct_colors(len(sps_matched))

        plots_dict = {}
        for clus in self.clusters:
            plots_dict['plot_cluster{0}'.format(clus)] = plt.subplots()

        if type_fig == 'bar':
            self.__bar_sps(sps_matched, colors, plots_dict, save_path, fig_name)

        elif type_fig == 'entropy':
            self.__entropy_sps(sps_matched, plots_dict, save_path, fig_name)

        else:
            raise NotImplementedError('Type of visualization not implemented')

    def __bar_sps(self, sps_matched, colors, plots, save_path, fig_name):
        for c_idx, clus in self.clusters.items():
            fig = plots['plot_cluster{0}'.format(c_idx)][0]
            ax = plots['plot_cluster{0}'.format(c_idx)][1]
            y_offset = np.zeros(len(self.tspan))
            y = self.all_simulations[clus]
            sps_total = np.sum(y[:, :, sps_matched], axis=(0, 2)) / (len(clus))
            sps_avg = np.zeros((len(sps_matched), len(self.tspan)))
            for idx, sp, in enumerate(sps_matched):
                sp_pctge = np.sum(y[:, :, sp], axis=0) / (sps_total * len(clus))
                sps_avg[idx] = sp_pctge

            for sps, col in enumerate(colors):
                sp_pctge = sps_avg[sps]
                ax.bar(self.tspan, sp_pctge, color=col, bottom=y_offset, width=self.tspan[2]-self.tspan[1])
                y_offset = y_offset + sp_pctge

            ax.set(xlabel='Time', ylabel='Percentage')
            fig.suptitle('Cluster {0}'.format(c_idx))
            ax.legend(sps_matched, loc='lower center', bbox_to_anchor=(0.50, -0.4), ncol=5,
                      title='Species indices')
            final_save_path = os.path.join(save_path, 'hist_avg_clus{0}_{1}'.format(c_idx, fig_name))
            fig.savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')
        return

    def __entropy_sps(self, sps_matched, plots, save_path, fig_name):
        max_entropy = 0
        for c_idx, clus in self.clusters.items():
            ax = plots['plot_cluster{0}'.format(c_idx)][1]
            y = self.all_simulations[clus]
            sps_total = np.sum(y[:, :, sps_matched], axis=(0, 2)) / (len(clus))
            sps_avg = np.zeros((len(sps_matched), len(self.tspan)))
            for idx, sp, in enumerate(sps_matched):
                sp_pctge = np.sum(y[:, :, sp], axis=0) / (sps_total * len(clus))
                sps_avg[idx] = sp_pctge

            entropies = [0] * len(self.tspan)
            for t_idx in range(len(self.tspan)):
                tp_pctge = sps_avg[:, t_idx]
                entropy = hf.get_probs_entropy(tp_pctge)
                entropies[t_idx] = entropy
            if max(entropies) > max_entropy:
                max_entropy = max(entropies)

            ax.plot(self.tspan, entropies)
            ax.set(xlabel='Time', ylabel='Entropy')

        for c in self.clusters:
            plots['plot_cluster{0}'.format(c)][1].set(ylim=(0, max_entropy))
            final_save_path = os.path.join(save_path, 'hist_avg_clus{0}_{1}'.format(c, fig_name))
            plots['plot_cluster{0}'.format(c)][0].savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')
        return

    def plot_pattern_rxns_distribution(self, pattern, type_fig='bar', y_lim=(0, 1), save_path='', fig_name=''):
        """
        This function uses the given pattern to match it with reactions in which it is present
        as a product and as a reactant. There are two types of visualization:
        Bar: This visualization obtains all the reaction rates on which the pattern matches as a product
        and as a reactant and then finds the percentage that each of reactions has compared with the total
        Entropy: This visualization uses the percentages of each of the reactions compared with the total
        and calculates the entropy as a proxy of variance in the number of states at a given time point
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

        plots_dict = {}
        for clus in self.clusters:
            plots_dict['plot_cluster{0}'.format(clus)] = plt.subplots(2, sharex=True)

        if type_fig == 'bar':
            self.__bar_rxns(products_matched, reactants_matched, plots_dict, save_path, fig_name)

        elif type_fig == 'entropy':
            self.__entropy__rxns(products_matched, reactants_matched, plots_dict, save_path, fig_name)

        else:
            raise NotImplementedError('Type of visualization not implements')
        return

    def __get_avgs(self, y, pars, products_matched, reactants_matched):
        """
        This function uses the simulated trajectories from each cluster and obtains the reaction
        rates of the matched product reactions and reactant reactions. After obtaining the
        reaction rates values it normalizes the reaction rates by the sum of the reaction rates
        each time point. It also generates the labels to know what color represent each reaction
        in the bar graph.
        Parameters
        ----------
        y
        pars
        products_matched
        reactants_matched

        Returns
        -------

        """
        products_avg = np.zeros((len(products_matched), len(self.tspan)))
        reactants_avg = np.zeros((len(reactants_matched), len(self.tspan)))
        pcolors = distinct_colors(len(products_matched))
        rcolors = distinct_colors(len(reactants_matched))
        plegend_patches = []
        plabels = []
        rlegend_patches = []
        rlabels = []

        # Obtaining reaction rates values
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
            # values[values < 0] = 0
            values_avg = np.average(values, axis=0)
            values_avg[values_avg < 0] = 0
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
            values_avg = np.average(values, axis=0)
            values_avg[values_avg > 0] = 0
            reactants_avg[rct_idx] = values_avg

            # Creating labels
            rlabel = hf.rate_2_interactions(self.model, str(rate))
            rlabels.append(rlabel)
            rlegend_patches.append(mpatches.Patch(color=rcolors[rct_idx], label=rlabel))

        rtotals = np.sum(reactants_avg, axis=0)
        reactants_avg = reactants_avg / rtotals

        plegend_info = (pcolors, plabels, plegend_patches)
        rlegend_info = (rcolors, rlabels, rlegend_patches)

        return products_avg, reactants_avg, plegend_info, rlegend_info

    def __bar_rxns(self, products_matched, reactants_matched, plots, save_path, fig_name):
        for c_idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            pars = self.all_parameters[clus]
            y_poffset = np.zeros(len(self.tspan))
            y_roffset = np.zeros(len(self.tspan))

            products_avg, reactants_avg, plegend_info, rlegend_info = self.__get_avgs(y, pars,
                                                                                      products_matched,
                                                                                      reactants_matched)
            pcolors, plabels, plegend_patches = plegend_info
            rcolors, rlabels, rlegend_patches = rlegend_info

            fig = plots['plot_cluster{0}'.format(c_idx)][0]
            ax1, ax2 = plots['plot_cluster{0}'.format(c_idx)][1]

            for prxn, pcol in zip(range(len(products_matched)), pcolors):
                sp_pctge = products_avg[prxn]
                ax1.bar(self.tspan, sp_pctge, color=pcol, bottom=y_poffset, width=self.tspan[2]-self.tspan[1])
                y_poffset = y_poffset + sp_pctge

            for rrxn, rcol in zip(range(len(reactants_matched)), rcolors):
                sp_pctge = reactants_avg[rrxn]
                ax2.bar(self.tspan, sp_pctge, color=rcol, bottom=y_roffset, width=self.tspan[2]-self.tspan[1])
                y_roffset = y_roffset + sp_pctge

            ax1.set(title='Products reactions')
            ax2.set(title='Reactants reactions')
            ax3 = fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            ax3.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            ax3.grid(False)
            ax3.set(xlabel="Time", ylabel='Percentage')

            final_save_path = os.path.join(save_path, 'hist_avg_rxn_clus{0}_{1}'.format(c_idx, fig_name))
            fig.savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')

            fig_plegend = plt.figure(figsize=(2, 1.25))
            fig_plegend.legend(plegend_patches, plabels, loc='center', frameon=False, ncol=4)
            plt.savefig('plegends_{0}.pdf'.format(fig_name), format='pdf', bbox_inches='tight')

            fig_rlegend = plt.figure(figsize=(2, 1.25))
            fig_rlegend.legend(rlegend_patches, rlabels, loc='center', frameon=False, ncol=4)
            plt.savefig('rlegends{0}.pdf'.format(fig_name), format='pdf', bbox_inches='tight')
        return

    def __entropy__rxns(self, products_matched, reactants_matched, plots, save_path, fig_name):
        pmax_entropy = 0
        rmax_entropy = 0
        for c_idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            pars = self.all_parameters[clus]
            products_avg, reactants_avg, plegend_info, rlegend_info = self.__get_avgs(y, pars,
                                                                                      products_matched,
                                                                                      reactants_matched)

            fig = plots['plot_cluster{0}'.format(c_idx)][0]
            ax1, ax2 = plots['plot_cluster{0}'.format(c_idx)][1]

            pentropies = [0] * len(self.tspan)
            rentropies = [0] * len(self.tspan)
            for t_idx in range(len(self.tspan)):
                tp_pctge = products_avg[:, t_idx]
                pentropy = hf.get_probs_entropy(tp_pctge)
                pentropies[t_idx] = pentropy

                tr_pctge = reactants_avg[:, t_idx]
                rentropy = hf.get_probs_entropy(tr_pctge)
                rentropies[t_idx] = rentropy

            if max(pentropies) > pmax_entropy:
                pmax_entropy = max(pentropies)

            if max(rentropies) > rmax_entropy:
                rmax_entropy = max(rentropies)

            ax1.plot(self.tspan, pentropies)
            ax2.plot(self.tspan, rentropies)

            ax3 = fig.add_subplot(111, frameon=False)

            # hide tick and tick label of the big axes
            ax3.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            ax3.grid(False)
            ax3.set(xlabel="Time", ylabel="Entropy")

        for c in self.clusters:
            plots['plot_cluster{0}'.format(c)][1][0].set(ylim=(0, pmax_entropy))
            plots['plot_cluster{0}'.format(c)][1][1].set(ylim=(0, rmax_entropy))
            final_save_path = os.path.join(save_path, 'hist_avg_rxn_clus{0}_{1}'.format(c, fig_name))
            plots['plot_cluster{0}'.format(c)][0].savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')
        return

    def plot_violin_pars(self, par_idxs, save_path=''):
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
            data_violin = [0] * len(self.clusters)
            clus_labels = [0] * len(self.clusters)
            count = 0
            for idx, clus in self.clusters.items():
                cluster_pars = self.all_parameters[clus]
                sp_ic_values = cluster_pars[:, sp_ic]
                data_violin[count] = sorted(np.log10(sp_ic_values))
                clus_labels[count] = idx
                count += 1

            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            ax1.set_title('Parameter {0}'.format(self.model.parameters[sp_ic].name))
            ax1.set_ylabel('Parameter values (log10)')
            parts = ax1.violinplot(data_violin, showmeans=False, showmedians=False, showextrema=False)

            for pc in parts['bodies']:
                pc.set_facecolor('#D43F3A')
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            percentile_data = np.array([np.percentile(data, [25, 50, 75]) for data in data_violin])
            quartile1 = percentile_data[:, 0]
            medians = percentile_data[:, 1]
            quartile3 = percentile_data[:, 2]

            whiskers = np.array([_adjacent_values(sorted_array, q1, q3)
                                 for sorted_array, q1, q3 in zip(data_violin, quartile1, quartile3)])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

            # set style for the axes
            _set_axis_style(ax1, clus_labels)

            final_save_path = os.path.join(save_path, 'violin_sp_{0}'.format(self.model.parameters[sp_ic].name))
            fig.savefig(final_save_path + '.pdf', format='pdf')

            ### Code to plot violinplots with seaborn
            # g = sns.violinplot(data=data_violin, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            # g.set(yticklabels=clus_labels, xlabel='Parameter Range', ylabel='Clusters')
            # fig = g.get_figure()
            # fig.suptitle('Parameter {0}'.format(self.model.parameters[sp_ic].name))
            # final_save_path = os.path.join(save_path, 'violin_sp_{0}'.format(self.model.parameters[sp_ic].name))
            # fig.savefig(final_save_path + '.pdf', format='pdf')
        return

    def plot_violin_kd(self, par_idxs, save_path=''):
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

            fig, ax1 = plt.subplots(nrows=1, ncols=1)
            ax1.set_title('Parameter {0}'.format(self.model.parameters[kd_pars[0]].name))
            ax1.set_ylabel('Parameter values (log10)')
            parts = ax1.violinplot(data_violin, showmeans=False, showmedians=False, showextrema=False)

            for pc in parts['bodies']:
                pc.set_facecolor('#D43F3A')
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            percentile_data = np.array([np.percentile(data, [25, 50, 75]) for data in data_violin])
            quartile1 = percentile_data[:, 0]
            medians = percentile_data[:, 1]
            quartile3 = percentile_data[:, 2]

            whiskers = np.array([_adjacent_values(sorted_array, q1, q3)
                                 for sorted_array, q1, q3 in zip(data_violin, quartile1, quartile3)])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax1.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

            # set style for the axes
            _set_axis_style(ax1, clus_labels)

            final_save_path = os.path.join(save_path, 'violin_sp_{0}'.format(self.model.parameters[kd_pars[0]].name))
            fig.savefig(final_save_path + '.pdf', format='pdf')

            # g = sns.violinplot(data=data_violin, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            # g.set(yticklabels=clus_labels, xlabel='Parameter Range', ylabel='Clusters')
            # fig = g.get_figure()
            # fig.suptitle('Parameter {0}'.format(self.model.parameters[kd_pars[0]].name))
            # final_save_path = os.path.join(save_path, 'violin_sp_{0}_kd'.format(self.model.parameters[kd_pars[0]].name))
            # fig.savefig(final_save_path + '.pdf', format='pdf')
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

        number_pars = self.nsims

        if type(par_idxs) == int:
            par_idxs = [par_idxs]

        for ic in par_idxs:
            fig, ax = plt.subplots(1)
            sp_ic_values_all = self.all_parameters[:, ic]
            sp_ic_weights_all = np.ones_like(sp_ic_values_all) / len(sp_ic_values_all)
            n, bins, patches = ax.hist(sp_ic_values_all, weights=sp_ic_weights_all, bins=30, fill=False)

            cluster_ic_values = []
            cluster_ic_weights = []
            for clus in self.clusters.values():
                cluster_pars = self.all_parameters[clus]
                sp_ic_values = cluster_pars[:, ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values_all)
                cluster_ic_values.append(sp_ic_values)
                cluster_ic_weights.append(sp_ic_weights)

            label = ['cluster_{0}, {1}%'.format(cl, (len(self.clusters[cl]) / number_pars) * 100)
                     for cl in self.clusters.keys()]
            ax.hist(cluster_ic_values, bins=bins, weights=cluster_ic_weights, stacked=True, label=label,
                     histtype='bar', ec='black')
            ax.set(xlabel='Concentration', ylabel='Percentage', title=self.model.parameters[ic].name)
            ax.legend(loc=0)

            final_save_path = os.path.join(save_path, 'plot_ic_overlap_{0}'.format(ic))
            fig.savefig(final_save_path + '.png', format='png', dpi=700)
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


def _adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def _set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Clusters')
