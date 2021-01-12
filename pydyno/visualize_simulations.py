import collections
import colorsys
import csv
import numbers
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pydyno.util as hf
from pysb.bng import generate_equations
from pysb.pattern import SpeciesPatternMatcher, ReactionPatternMatcher
from pysb import Monomer, MonomerPattern, ComplexPattern
import sympy
from pydyno.distinct_colors import distinct_colors
from pydyno.discretization.pysb_discretize import calculate_reaction_rate
import matplotlib.patches as mpatches

plt.ioff()


class VisualizeSimulations(object):
    """
    Visualize PySB simulations and parameter distributions in different clusters

    Parameters
    ----------
    model: pysb.Model
        PySB model used to obtain the simulations
    sim_results: SimulationResult or h5 file from PySB simulation
        SimulationResult object or h5 file with the dynamic solutions of the model for all the parameter sets
    clusters: vector-like or str or None
        Indices of the parameters that belong to an specific cluster. It can be a list of files that contain
        the indices of each cluster, a list of lists where each list has the parameter indices of a cluster or
        a file that contains the cluster labels to which each parameter belongs to, or None if the user want to
        analyse the sim_results as a single cluster.
    truncate_idx: int
        Index at which the simulation is truncated. Only works when clusters is None. It cannot be used at the same
         time with truncate_idx.
    drop_sim_idx: array-like
        Indices of simulation to drop. Only works when clusters is None. It cannot be used at the same time with
        truncate_idx.

    Examples
    --------
    Visualize the trajectory of a simulation:

    >>> from pydyno.visualize_simulations import VisualizeSimulations
    >>> from pysb.examples.tyson_oscillator import model
    >>> from pysb.simulator import ScipyOdeSimulator
    >>> import numpy as np
    >>> sim = ScipyOdeSimulator(model, tspan=np.linspace(0, 100, 100)).run()
    >>> vs = VisualizeSimulations(model, sim, clusters=None)
    >>> vs.plot_cluster_dynamics([0, 1, 2]) \
        #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    {'comp0_cluster0': (<Figure size 640x480 with 1 Axes>, <matplotlib.axes._subplots.AxesSubplot object at ...>),
    'comp1_cluster0': (<Figure size 640x480 with 1 Axes>, <matplotlib.axes._subplots.AxesSubplot object at ...>),
    'comp2_cluster0': (<Figure size 640x480 with 1 Axes>, <matplotlib.axes._subplots.AxesSubplot object at ...>)}
    """

    def __init__(self, model, sim_results, clusters, truncate_idx=None,
                 truncate_time=None, drop_sim_idx=None):

        self._model = model
        generate_equations(model)
        # Check simulation results
        self._all_simulations, self._all_parameters, self._nsims, self._tspan = hf.get_simulations(sim_results)
        try:
            changed_parameters = sim_results.changed_parameters
            time_change = sim_results.time_change
        except AttributeError:
            changed_parameters = None
            time_change = None
        self._changed_parameters = changed_parameters
        self._time_change = time_change

        self._par_name_idx = {j.name: i for i, j in enumerate(self.model.parameters)}

        if clusters is None:
            if truncate_idx is not None and drop_sim_idx is None:
                self._all_simulations = self._all_simulations[:truncate_idx, :]
                samples = range(len(self.all_parameters) - truncate_idx)
            elif drop_sim_idx is not None and truncate_idx is None:
                self._all_simulations = np.delete(self._all_simulations, drop_sim_idx, axis=0)
                samples = range(len(self.all_parameters) - len(drop_sim_idx))
            elif truncate_idx is None and drop_sim_idx is None:
                samples = range(len(self.all_parameters))
            else:
                raise ValueError('both truncate_idx and drop_sim_idx cannot bet different than None '
                                 'at the same time')
            # remove cluster indices if drop_sim_idx is not None

            no_clusters = {0: samples}
            self._clusters = no_clusters

        else:
            if truncate_time is not None:
                self._all_simulations = self._all_simulations[:, :truncate_time, :]
                self._tspan = self._tspan[:truncate_time]
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

    @property
    def par_name_idx(self):
        return self._par_name_idx

    @property
    def changed_parameters(self):
        return self._changed_parameters

    @property
    def time_change(self):
        return self._time_change

    @staticmethod
    def check_clusters_arg(clusters, nsims):  # check clusters
        def _clusters_to_dict(cluster_labels, n):
            # Takes a list of cluster labels and create a dictionary where the keys are the labels, and
            # the values are the indices in the original list
            number_pars = len(cluster_labels)
            if number_pars != n:
                raise ValueError('The number of cluster indices must have the same'
                                 'length as the number of simulations')
            num_of_clusters = set(cluster_labels)
            clus_values = {}
            for j in num_of_clusters:
                item_index = np.where(cluster_labels == j)
                clus_values[j] = item_index[0].tolist()
            cluster_labels = clus_values
            return cluster_labels

        if isinstance(clusters, collections.Iterable):
            if all(isinstance(item, numbers.Number) for item in clusters):
                if not isinstance(clusters, np.ndarray):
                    clusters = np.array(clusters)
                pars_clusters = clusters
                clusters = _clusters_to_dict(pars_clusters, nsims)
                return clusters
            else:
                raise ValueError('Mixed formats is not supported')
        # check is clusters is a file that contains the indices of the clusters for each parameter set
        elif isinstance(clusters, str):
            if os.path.isfile(clusters):
                f = open(clusters)
                data = csv.reader(f)
                pars_clusters = np.array([int(d[0]) for d in data])
                clusters = _clusters_to_dict(pars_clusters, nsims)
                return clusters
        else:
            raise TypeError('cluster data structure not supported')

    def plot_cluster_dynamics(self, components, x_data=None, y_data=None, y_error=None, dir_path='',
                              type_fig='trajectories', add_y_histogram=False, fig_name='', plot_format='png',
                              species_ftn_fit=None, norm=False, norm_value=None, fit_options={}, figure_options={}):
        """
        Plots the dynamics of species/observables/pysb expressions for each cluster

        Parameters
        ----------
        components: list-like
            Indices of PySB species that will be plotted, or observable names, or pysb expressions
        x_data: dict
            Dictionary where the keys must be the same as the components names. Dictionary values
            correspond to the time points at which the experimental data was obtained
        y_data: dict
            Dictionary where the keys must be the same as the components names. Dictionary values
            correspond to experimental concentration data
        y_error: dict
            Dictionary where the keys must be the same as the components names. Dictionary values
            correspond to the experimental errors (e.g standard deviation)
        dir_path: str
            Path to folder where the figure is going to be saved
        type_fig: str
            Type of figure to plot. It can be `trajectories` to plot all the simulated trajectories or `mean_std`
            to plot the mean and standard deviation
        add_y_histogram: bool
            Whether to add a histogram of the concentrations at the last time point of the simulation
        fig_name: str
            String used to give a name to the cluster figures
        plot_format: str; default `png`
            Format used to save the figures: png, pdf, etc
        species_ftn_fit: dict, optional
            Dictionary of species with their respective function to fit their dynamics.
        norm: boolean, optional
            Normalizes species by max value in simulation
        norm_value: array-like or str
            Array of values used to normalized species concentrations. Must have same order
            as species
        kwargs: dict
            Arguments to pass to the fitting function

        Returns
        -------
        dict
            A dictionary whose keys are the names of the clusters and the values are arrays with the
            corresponding matplotlib Figure and Axes objects.
        """
        # Plot experimental data if provided
        if x_data is not None and y_data is not None:
            # creates a dictionary to store the different figures by cluster
            plots_dict = {}
            for comp_idx, comp in enumerate(components):
                # access experimental data
                x = x_data[comp]
                y = y_data[comp]
                for clus in self.clusters:
                    fig, ax = plt.subplots(**figure_options)
                    if y_error is not None:
                        yerr = y_error[comp]
                        ax.errorbar(x, y, yerr, color='r', alpha=1, zorder=10)
                    else:
                        ax.plot(x, y, color='r', alpha=1, zorder=10)
                    plots_dict['comp{0}_cluster{1}'.format(comp_idx, clus)] = (fig, ax)
        elif x_data is None and y_data is None:
            # creates a dictionary to store the different figures by cluster
            plots_dict = {}
            for comp_idx, comp in enumerate(components):
                for clus in self.clusters:
                    plots_dict['comp{0}_cluster{1}'.format(comp_idx, clus)] = plt.subplots(**figure_options)
        else:
            raise ValueError('both x_data and y_data must be passed to plot experimental data')

        if norm:
            if species_ftn_fit:
                # checking if species_to_fit are present in the species that are going to be plotted
                plot_data = self._plot_dynamics_cluster_types_norm_ftn_species(plots_dict=plots_dict,
                                                                               components=components,
                                                                               species_ftn_fit=species_ftn_fit,
                                                                               **fit_options)

            else:
                plot_data = self._plot_dynamics_cluster_types_norm(plots_dict=plots_dict, components=components,
                                                                   norm_value=norm_value,
                                                                   add_y_histogram=add_y_histogram)

        else:
            plot_data = self._plot_dynamics_cluster_types(plots_dict=plots_dict, components=components,
                                                          add_y_histogram=add_y_histogram,
                                                          type_fig=type_fig)

        if fig_name:
            fig_name = '_' + fig_name
        for name, plot in plot_data.items():
            final_save_path = os.path.join(dir_path, name + fig_name)
            plot[0].savefig(final_save_path + f'.{plot_format}', dpi=500)

        return plot_data

    def _calculate_expr_values(self, y, component, clus):
        # Calculates the reaction rate values, observable values or species values
        # depending in the sp type
        if isinstance(component, sympy.Expr):
            expr_vars = [atom for atom in component.atoms(sympy.Symbol)]
            expr_args = [0] * len(expr_vars)
            for va_idx, va in enumerate(expr_vars):
                if str(va).startswith('__'):
                    sp_idx = int(''.join(filter(str.isdigit, str(va))))
                    expr_args[va_idx] = y[:, :, sp_idx].T
                else:
                    par_idx = self.model.parameters.index(va)
                    expr_args[va_idx] = self.all_parameters[clus][:, par_idx]
            f_expr = sympy.lambdify(expr_vars, component)
            sp_trajectory = f_expr(*expr_args)
            name = 'expr'

        # Calculate observable
        elif isinstance(component, str):
            sp_trajectory = self._get_observable(component, y)
            name = component
        elif isinstance(component, int):
            sp_trajectory = y[:, :, component].T
            name = self.model.species[component]
        elif isinstance(component, (Monomer, MonomerPattern, ComplexPattern)):
            spm = SpeciesPatternMatcher(self.model)
            sps_matched = spm.match(component, index=True)
            sps_values = np.sum(y[:, :, sps_matched], axis=2)
            sp_trajectory = sps_values.T
            name = str(component)
        else:
            raise TypeError('Type of model component not valid for visualization')
        return sp_trajectory, name

    def _plot_dynamics_cluster_types(self, plots_dict, components, add_y_histogram, type_fig):
        for idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            for comp_idx, comp in enumerate(components):
                # Obtain component values
                sp_trajectory, name = self._calculate_expr_values(y, comp, clus)
                fig, ax = plots_dict['comp{0}_cluster{1}'.format(comp_idx, idx)]
                if type_fig == 'trajectories':
                    ax.plot(self.tspan, sp_trajectory,
                            color='blue',
                            alpha=0.2)
                elif type_fig == 'mean_std':
                    mean = np.mean(sp_trajectory, axis=1)
                    std = np.std(sp_trajectory, axis=1)
                    ax.errorbar(self.tspan, mean, yerr=std, color='blue')

                if add_y_histogram:
                    self._add_y_histogram(ax, sp_trajectory)

                sp_max_conc = np.nanmax(sp_trajectory[sp_trajectory != np.inf])
                sp_min_conc = np.nanmin(sp_trajectory[sp_trajectory != -np.inf])
                ax.set_xlabel('Time')
                ax.set_ylabel('Concentration')
                # plots_dict['plot_sp{0}_cluster{1}'.format(comp, clus)][1].set_xlim([0, 8])
                ax.set_ylim([sp_min_conc, sp_max_conc])
                fig.suptitle('{0}, cluster {1}'.
                             format(name, idx))
        return plots_dict

    def _plot_dynamics_cluster_types_norm(self, plots_dict, components, norm_value=None, add_y_histogram=False):
        for idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            for comp_idx, comp in enumerate(components):
                # Calculate reaction rate expression
                sp_trajectory, name = self._calculate_expr_values(y, comp, clus)
                if isinstance(norm_value, list):
                    norm_trajectories = sp_trajectory / norm_value[n]
                elif norm_value == 'sum_total':
                    sum_total = np.sum(y, axis=2).T
                    norm_trajectories = np.divide(sp_trajectory, sum_total)
                else:
                    norm_trajectories = np.divide(sp_trajectory, np.amax(sp_trajectory, axis=0))

                fig, ax = plots_dict['comp{0}_cluster{1}'.format(comp_idx, idx)]
                ax.plot(self.tspan,
                        norm_trajectories,
                        color='blue',
                        alpha=0.01)

                if add_y_histogram:
                    self._add_y_histogram(ax, norm_trajectories)

                ax.set_xlabel('Time')
                ax.set_ylabel('Concentration')
                # plots_dict['plot_sp{0}_cluster{1}'.format(comp, clus)][1].set_xlim([0, 8])
                ax.set_ylim([0, 1])
                fig.suptitle('{0}, cluster {1}'.format(name, idx))
        return plots_dict

    def _plot_dynamics_cluster_types_norm_ftn_species(self, plots_dict, components, species_ftn_fit, **kwargs):
        comps_idx_fit = [components.index(comp_fit) for comp_fit in species_ftn_fit]
        for idx, clus in self.clusters.items():
            ftn_result = {}
            y = self.all_simulations[clus]
            for comp_idx, comp in enumerate(components):
                # Calculate reaction rate expression
                sp_trajectory, name = self._calculate_expr_values(y, comp, clus)
                norm_trajectories = np.divide(sp_trajectory, np.amax(sp_trajectory, axis=0))
                fig, ax = plots_dict['comp{0}_cluster{1}'.format(comp_idx, idx)]
                # setting plot information
                ax.set_xlabel('Time')
                ax.set_ylabel('Concentration')
                ax.set_ylim([0, 1])
                fig.suptitle('{0}, cluster {1}'.format(name, idx))
                ax.plot(self.tspan,
                        norm_trajectories,
                        color='blue',
                        alpha=0.2)
                if comp_idx in comps_idx_fit:
                    result_fit = hf.curve_fit_ftn(fn=species_ftn_fit[components[comp_idx]], xdata=self.tspan,
                                                  ydata=sp_trajectory.T, **kwargs)
                    ftn_result[comp] = result_fit
            self._add_function_hist(plots_dict=plots_dict, idx=idx, sp_overlap=comps_idx_fit, components=components, ftn_result=ftn_result)

        return plots_dict

    @staticmethod
    def _add_y_histogram(ax, sp_trajectory):
        divider = make_axes_locatable(ax)
        # axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
        axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
        plt.setp(axHisty.get_yticklabels(), visible=False)
        hist_data = sp_trajectory[-1, :]
        axHisty.hist(hist_data, density=True, orientation='horizontal')
        shape = np.std(hist_data)
        scale = np.average(hist_data)

        pdf_pars = r'$\sigma$ =' + str(round(shape, 2)) + '\n' r'$\mu$ =' + str(round(scale, 2))
        anchored_text = AnchoredText(pdf_pars, loc=1, prop=dict(size=10))
        anchored_text.patch.set_alpha(0.5)
        axHisty.add_artist(anchored_text)
        axHisty.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

    def _add_function_hist(self, plots_dict, idx, sp_overlap, components, ftn_result):
        for sp_dist in sp_overlap:
            ax = plots_dict['comp{0}_cluster{1}'.format(sp_dist, idx)][1]
            divider = make_axes_locatable(ax)
            axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
            # axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
            plt.setp(axHistx.get_xticklabels(),
                     visible=False)  # + axHisty.get_yticklabels(), visible=False)

            # This is specific for the time of death fitting in apoptosis
            hist_data = hf.column(ftn_result[components[sp_dist]], 1)
            # TODO I should look deeper into how many trajectories have different dynamics
            hist_data_filt = hist_data[(hist_data > 0) & (hist_data < self.tspan[-1])]
            # shape, loc, scale = lognorm.fit(hist_data_filt, floc=0)
            # pdf = lognorm.pdf(np.sort(hist_data_filt), shape, loc, scale)
            shape = np.std(hist_data_filt)
            scale = np.average(hist_data_filt)

            pdf_pars = r'$\sigma$ =' + str(round(shape, 2)) + '\n' r'$\mu$ =' + str(round(scale, 2))
            anchored_text = AnchoredText(pdf_pars, loc=1, prop=dict(size=12))
            axHistx.add_artist(anchored_text)
            axHistx.hist(hist_data_filt, density=True, bins=20)
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

    def hist_clusters_parameters(self, par_idxs, ylabel='', plot_format='png', dir_path=''):
        """
        Creates a Figure for each cluster. Each figure contains a histogram for each parameter provided.
        The sum of each parameter histogram is normalized to 1

        Parameters
        ----------
        par_idxs: list-like
            Indices of the model parameters that would be visualized
        ylabel: iterable or str
            y-axis labels for each figure
        dir_path: str
            Path to directory where the file is going to be saved
        plot_format: str; default `png`
            Format used to save the figures: `png`, `pdf`, etc
        Returns
        -------

        """
        from collections.abc import Iterable
        if isinstance(ylabel, str):
            ylabels = [ylabel] * len(self.clusters)
        elif isinstance(ylabel, Iterable):
            ylabels = ylabel
        else:
            raise TypeError('ylabel must be a string or an iterable of strings')
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
            plt.xlabel(ylabels[c_idx])
            plt.ylabel('Percentage')
            plt.legend(loc=0)
            final_save_path = os.path.join(dir_path, 'hist_ic_type{0}'.format(c_idx))
            plt.savefig(final_save_path + '', format=plot_format, dpi=700)
            plt.clf()
            plt.close()
        return

    def plot_pattern_sps_distribution(self, pattern, type_fig='bar', dir_path='', fig_name=''):
        """
        Creates a Figure for each cluster. First, it obtains all the species that match the pattern argument.
        If type_fig is `bar` it creates a stacked bar at each time point of the simulation to represent the species
        percentage with respect to the sum-total of the species concentrations. If type_fig `entropy` the entropy
        is calculated from the distribution of the matched species at each time point of the simulation.

        Parameters
        ----------
        pattern: pysb.Monomer or pysb.MonomerPattern or pysb.ComplexPattern
        type_fig: str
            `bar` to get a stacked bar of the distribution of the species that match the provided pattern.
            `entropy` to get the entropy of the distributions of the species that match the provided pattern
        dir_path: str
            Path to folder where the figure is going to be saved
        fig_name: str
            String used to give a name to the cluster figures

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
            sps_avg_clusters = self.__bar_sps(sps_matched, colors, plots_dict, dir_path, fig_name)
            return sps_avg_clusters
        elif type_fig == 'entropy':
            self.__entropy_sps(sps_matched, plots_dict, dir_path, fig_name)

        else:
            raise NotImplementedError('Type of visualization not implemented')

    def __bar_sps(self, sps_matched, colors, plots, dir_path, fig_name):
        sps_avg_clusters = []
        for c_idx, clus in self.clusters.items():
            fig = plots['plot_cluster{0}'.format(c_idx)][0]
            ax = plots['plot_cluster{0}'.format(c_idx)][1]
            y_offset = np.zeros(len(self.tspan) - 1)  # ignore first time point. Concentrations of zero cause issues
            y = self.all_simulations[clus]
            sps_total = np.sum(y[:, :, sps_matched], axis=(0, 2)) / (len(clus))
            sps_avg = np.zeros((len(sps_matched), len(self.tspan)))
            for idx, sp, in enumerate(sps_matched):
                sp_pctge = np.sum(y[:, :, sp], axis=0) / (sps_total * len(clus))
                sps_avg[idx] = sp_pctge

            for sps, col in enumerate(colors):
                sp_pctge = sps_avg[sps][1:]  # ignore first time point. Concentrations of zero cause issues
                ax.bar(self.tspan[1:], sp_pctge, color=col, bottom=y_offset, width=self.tspan[2] - self.tspan[1])
                y_offset = y_offset + sp_pctge

            sps_avg_clusters.append(sps_avg)

            ax.set(xlabel='Time', ylabel='Percentage', ylim=(0, 1))
            fig.suptitle('Cluster {0}'.format(c_idx))
            ax.legend(sps_matched, loc='lower center', bbox_to_anchor=(0.50, -0.4), ncol=5,
                      title='Species indices')
            final_save_path = os.path.join(dir_path, 'hist_avg_clus{0}_{1}'.format(c_idx, fig_name))
            fig.savefig(final_save_path + '.png', format='png', bbox_inches='tight')
            plt.close(fig)
        return sps_avg_clusters

    def __entropy_sps(self, sps_matched, plots, dir_path, fig_name):
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
            final_save_path = os.path.join(dir_path, 'hist_avg_clus{0}_{1}'.format(c, fig_name))
            plots['plot_cluster{0}'.format(c)][0].savefig(final_save_path + '.png', format='png', bbox_inches='tight')
            plt.close(plots['plot_cluster{0}'.format(c)][0])
        return

    def plot_pattern_rxns_distribution(self, pattern, exclude_rxns=None, normalize=False,
                                       type_fig='bar', dir_path='', fig_name=''):
        """
        Creates a Figure for each cluster. First, it obtains the reactions in which the pattern matches the reactions
        reactants plus the reactions in which the pattern matches the reactions products. There are two types of
        visualizations: `bar`: This visualization obtains all the reaction rates where the pattern matches the
        reaction products (reactants) and then obtains the percentage that each of the reactions' rate has compared
        with the sum-total of the reactions rates. Entropy: This visualization uses the percentages of each of the
        reactions compared with the total and calculates the entropy as a proxy of variance in the number of states
        at each time point of the simulation.

        Parameters
        ----------
        pattern: pysb.Monomer or pysb.MonomerPattern or pysb.ComplexPattern
            Pattern used to obtain distributions
        exclude_rxns: list-like
            A list of reactions (sympy expressions) to exclude from the visualization.
        normalize: bool
            Whether to normalize reaction rates by their sum-total
        type_fig: str
            `bar` to get a stacked bar of the distribution of the species that match the provided pattern.
            `entropy` to get the entropy of the distributions of the species that match the provided pattern
        dir_path: str
            Path to directory where the plots are going to be saved
        fig_name: str
            Figure name

        Returns
        -------

        """
        # Matching reactions to pattern
        rpm = ReactionPatternMatcher(self.model)
        products_matched = rpm.match_products(pattern)
        reactants_matched = rpm.match_reactants(pattern)

        if exclude_rxns is not None:
            from sympy import simplify
            product_idxs_to_remove = []
            reactants_idxs_to_remove = []
            for ir in exclude_rxns:
                for idx, rxn in enumerate(products_matched):
                    if simplify(rxn.rate - ir) == 0:
                        product_idxs_to_remove.append(idx)
                        break

                for idx, rxn in enumerate(reactants_matched):
                    if simplify(rxn.rate - ir) == 0:
                        reactants_idxs_to_remove.append(idx)
                        break

            products_matched = [rxn for idx, rxn in enumerate(products_matched)
                                if idx not in product_idxs_to_remove]
            reactants_matched = [rxn for idx, rxn in enumerate(reactants_matched)
                                 if idx not in reactants_idxs_to_remove]

        rev_rxns_products = []
        rev_rxns_reactants = []
        # Add reversible reactions
        for pm in products_matched:
            if pm.reversible:
                pm_rev = copy.deepcopy(pm)
                pm_rev._rxn_dict['rev'] = True
                rev_rxns_products.append(pm_rev)
        for rm in reactants_matched:
            if rm.reversible:
                rm_rev = copy.deepcopy(rm)
                rm_rev._rxn_dict['rev'] = True
                rev_rxns_reactants.append(rm_rev)
        products_matched = products_matched + rev_rxns_reactants
        reactants_matched = reactants_matched + rev_rxns_products

        plots_dict = {}
        for clus in self.clusters:
            plots_dict['plot_cluster{0}'.format(clus)] = plt.subplots(2, sharex=True)

        if type_fig == 'bar':
            self.__bar_rxns(products_matched, reactants_matched, plots_dict, dir_path, fig_name, normalize)

        elif type_fig == 'entropy':
            self.__entropy__rxns(products_matched, reactants_matched, plots_dict, dir_path, fig_name, normalize)

        else:
            raise NotImplementedError('Type of visualization not implemented')
        return

    def _get_avgs(self, y, pars, products_matched, reactants_matched, normalize):
        """
        This function uses the simulated trajectories from each cluster and obtains the reaction
        rates of the matched product reactions and reactant reactions. After obtaining the
        reaction rates values it normalizes the reaction rates by the sum of the reaction rates at
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
        all_products_std = np.zeros((len(products_matched), len(self.tspan)))
        all_reactants_std = np.zeros((len(reactants_matched), len(self.tspan)))
        unique_products = list(dict.fromkeys(products_matched))
        unique_reactants = list(dict.fromkeys(reactants_matched))
        all_reactions = unique_products + unique_reactants
        colors = distinct_colors(len(all_reactions))
        reaction_color = {reaction.rate: colors[idx] for idx, reaction in enumerate(all_reactions)}
        pcolors = []
        rcolors = []
        plegend_patches = []
        plabels = []
        rlegend_patches = []
        rlabels = []

        # Obtaining reaction rates values
        for rxn_idx, rxn in enumerate(products_matched):
            rate = rxn.rate
            values = calculate_reaction_rate(rate, y, pars, self.par_name_idx, self.changed_parameters,
                                             self.time_change)

            # values[values < 0] = 0
            values_avg = np.average(values, axis=0)
            products_std = np.std(values, axis=0)

            if rxn.reversible:
                if 'rev' in rxn._rxn_dict.keys():
                    values_avg[values_avg < 0] = values_avg[values_avg < 0] * (-1)
                    values_avg[values_avg > 0] = 0
                    products_std[values_avg > 0] = 0
                else:
                    products_std[:, values_avg < 0] = 0
                    values_avg[values_avg < 0] = 0

            all_products_std[rxn_idx] = products_std
            products_avg[rxn_idx] = values_avg

            # Creating labels
            plabel = str(rate)
            rxn_color = reaction_color[rate]
            pcolors.append(rxn_color)
            plabels.append(plabel)
            plegend_patches.append(mpatches.Patch(color=rxn_color, label=plabel))

        if normalize:
            ptotals = np.sum(products_avg, axis=0)
            products_avg = products_avg / (ptotals + np.finfo(float).eps)  # Add small number to avoid division by zero

        for rct_idx, rct in enumerate(reactants_matched):
            rate = rct.rate
            values = calculate_reaction_rate(rate, y, pars, self.par_name_idx, self.changed_parameters,
                                             self.time_change)

            values_avg = np.average(values, axis=0)
            reactants_std = np.std(values, axis=0)

            if rct.reversible:
                if 'rev' in rct._rxn_dict.keys():
                    values_avg[values_avg < 0] = values_avg[values_avg < 0] * (-1)
                    values_avg[values_avg > 0] = 0
                    reactants_std[values_avg > 0] = 0
                else:
                    reactants_std[values_avg < 0] = 0
                    values_avg[values_avg < 0] = 0

            all_reactants_std[rct_idx] = reactants_std
            reactants_avg[rct_idx] = values_avg

            # Creating labels
            rlabel = str(rate)
            rxn_color = reaction_color[rate]
            rcolors.append(rxn_color)
            rlabels.append(rlabel)
            rlegend_patches.append(mpatches.Patch(color=rxn_color, label=rlabel))

        reactants_avg = np.abs(reactants_avg)

        if normalize:
            rtotals = np.sum(reactants_avg, axis=0)
            reactants_avg = reactants_avg / (
                        rtotals + np.finfo(float).eps)  # Add small number to avoid division by zero

        plegend_info = (pcolors, plabels, plegend_patches)
        rlegend_info = (rcolors, rlabels, rlegend_patches)

        return products_avg, all_products_std, reactants_avg, all_reactants_std, plegend_info, rlegend_info

    def __bar_rxns(self, products_matched, reactants_matched, plots, dir_path, fig_name, normalize):
        for c_idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            pars = self.all_parameters[clus]
            y_poffset = np.zeros(len(self.tspan))
            y_roffset = np.zeros(len(self.tspan))

            products_avg, all_products_ci, reactants_avg, all_reactants_ci, plegend_info, rlegend_info = \
                self._get_avgs(
                    y, pars,
                    products_matched,
                    reactants_matched,
                    normalize)

            pcolors, plabels, plegend_patches = plegend_info
            rcolors, rlabels, rlegend_patches = rlegend_info

            fig = plots['plot_cluster{0}'.format(c_idx)][0]
            ax1, ax2 = plots['plot_cluster{0}'.format(c_idx)][1]

            for prxn, pcol in zip(range(len(products_matched)), pcolors):
                sp_pctge = products_avg[prxn]
                if not normalize:
                    ax1.plot(self.tspan, sp_pctge, color=pcol)

                else:
                    ax1.bar(self.tspan, sp_pctge, color=pcol, bottom=y_poffset, width=self.tspan[2] - self.tspan[1])
                    y_poffset = y_poffset + sp_pctge

            for rrxn, rcol in zip(range(len(reactants_matched)), rcolors):
                sp_pctge = reactants_avg[rrxn]
                if not normalize:
                    ax2.plot(self.tspan, sp_pctge, color=rcol)
                else:
                    ax2.bar(self.tspan, sp_pctge, color=rcol, bottom=y_roffset, width=self.tspan[2] - self.tspan[1])
                y_roffset = y_roffset + sp_pctge

            ax1.set(title='Producing reactions')
            ax2.set(title='Consuming reactions')
            ax3 = fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            ax3.tick_params(axis='both', which='both', labelcolor='none', top=False,
                            bottom=False, left=False, right=False)
            ax3.grid(False)
            ax3.set(xlabel="Time", ylabel='Percentage')

            final_save_path = os.path.join(dir_path, 'hist_avg_rxn_clus{0}_{1}'.format(c_idx, fig_name))
            fig.savefig(final_save_path + '.png', format='png', bbox_inches='tight')
            plt.close(fig)

            fig_plegend = plt.figure(figsize=(2, 1.25))
            fig_plegend.legend(plegend_patches, plabels, loc='center', frameon=False, ncol=4)
            plt.savefig(final_save_path + 'plegends_{0}.png'.format(fig_name), format='png', bbox_inches='tight')
            plt.close(fig_plegend)

            fig_rlegend = plt.figure(figsize=(2, 1.25))
            fig_rlegend.legend(rlegend_patches, rlabels, loc='center', frameon=False, ncol=4)
            plt.savefig(final_save_path + 'rlegends{0}.png'.format(fig_name), format='png', bbox_inches='tight')
            plt.close(fig_rlegend)
        return

    def __entropy__rxns(self, products_matched, reactants_matched, plots, dir_path, fig_name, normalize):
        pmax_entropy = 0
        rmax_entropy = 0
        for c_idx, clus in self.clusters.items():
            y = self.all_simulations[clus]
            pars = self.all_parameters[clus]

            products_avg, all_products_ci, reactants_avg, all_reactants_ci, plegend_info, rlegend_info = \
                self._get_avgs(
                    y, pars,
                    products_matched,
                    reactants_matched,
                    normalize)

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
            final_save_path = os.path.join(dir_path, 'hist_avg_rxn_clus{0}_{1}'.format(c, fig_name))
            plots['plot_cluster{0}'.format(c)][0].savefig(final_save_path + '.pdf', format='pdf', bbox_inches='tight')
            plt.close(plots['plot_cluster{0}'.format(c)][0])
        return

    def plot_violin_pars(self, par_idxs, dir_path=''):
        """
        Creates a plot for each model parameter. Then, makes violin plots for each cluster

        Parameters
        ----------
        par_idxs: list-like
            Indices of the parameters that would be visualized
        dir_path: str
            Path to directory where the file is going to be saved

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

            final_save_path = os.path.join(dir_path, 'violin_sp_{0}'.format(self.model.parameters[sp_ic].name))
            fig.savefig(final_save_path + '.png', format='png', bbox_inches='tight')
            plt.close(fig)

            ### Code to plot violinplots with seaborn
            # g = sns.violinplot(data=data_violin, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            # g.set(yticklabels=clus_labels, xlabel='Parameter Range', ylabel='Clusters')
            # fig = g.get_figure()
            # fig.suptitle('Parameter {0}'.format(self.model.parameters[sp_ic].name))
            # final_save_path = os.path.join(save_path, 'violin_sp_{0}'.format(self.model.parameters[sp_ic].name))
            # fig.savefig(final_save_path + '.pdf', format='pdf')
        return

    def plot_violin_kd(self, par_idxs, dir_path=''):
        """
        Creates a plot for each kd parameter. Then, makes violin plots for each cluster

        Parameters
        ----------
        par_idxs: list-like
            Tuples of parameters indices, where the first entry is the k_reverse and
            the second entry is the k_forward parameter.
        dir_path: str
            Path to directory where the file is going to be saved

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

            final_save_path = os.path.join(dir_path, 'violin_sp_{0}'.format(self.model.parameters[kd_pars[0]].name))
            fig.savefig(final_save_path + '.pdf', format='pdf')
            plt.close(fig)

            # g = sns.violinplot(data=data_violin, orient='h', bw='silverman', cut=0, scale='count', inner='box')
            # g.set(yticklabels=clus_labels, xlabel='Parameter Range', ylabel='Clusters')
            # fig = g.get_figure()
            # fig.suptitle('Parameter {0}'.format(self.model.parameters[kd_pars[0]].name))
            # final_save_path = os.path.join(save_path, 'violin_sp_{0}_kd'.format(self.model.parameters[kd_pars[0]].name))
            # fig.savefig(final_save_path + '.pdf', format='pdf')
        return

    def plot_sp_ic_overlap(self, par_idxs, dir_path=''):
        """
        Creates a stacked histogram with the distributions of each of the
        clusters for each model parameter provided

        Parameters
        ----------
        par_idxs: list
            Indices of the initial conditions in model.parameter to plot
        dir_path: str
            Path to where the file is going to be saved

        Returns
        -------

        """

        number_pars = self.nsims

        if isinstance(par_idxs, int):
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

            final_save_path = os.path.join(dir_path, 'plot_ic_overlap_{0}'.format(ic))
            fig.savefig(final_save_path + '.png', format='png', dpi=700)
            plt.close(fig)
        return

    def scatter_plot_pars(self, par_idxs, cluster, dir_path=''):
        """

        Parameters
        ----------
        par_idxs: list
            Indices of the parameters to visualize
        cluster: list-like
        dir_path: str
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
        final_save_path = os.path.join(dir_path, 'scatter_{0}_{1}_cluster_{2}'.format(ic_name0, ic_name1,
                                                                                      cluster))
        plt.savefig(final_save_path + '.png', format='png', dpi=700)
        plt.close()

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
