from __future__ import division
import numpy as np
import csv
from pysb.integrate import ScipyOdeSimulator
import matplotlib.pyplot as plt
import colorsys
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import helper_functions as hf
from matplotlib.offsetbox import AnchoredText

plt.ioff()


class AnalysisCluster:
    """
    Class to generate the dynamics and distributions of the species concentrations in the different clusters
    """
    def __init__(self, model, tspan, parameters, clusters, sim_results=None):
        """

        :param model: PySB model
        :param tspan: time range for the simulation
        :param parameters: model parameters
        :param clusters: clusters from TroPy
        :param sim_results: Optional, trajectories of species from simulation results
        """
        self.model = model
        self.tspan = tspan
        self.sim = ScipyOdeSimulator(self.model, self.tspan)

        # check parameters
        if type(parameters) == str:
            self.all_parameters = np.load(parameters)
        elif type(parameters) == np.ndarray:
            self.all_parameters = parameters
        else:
            raise Exception('A valid set of parameters must be provided')
        if self.all_parameters.shape[1] != len(self.model.parameters):
            raise Exception("param_values must be the same length as model.parameters")

        # check if clusters is a list of files containing the indices of the IC that belong to that cluster
        if type(clusters) == list:
            clus_values = {}
            number_pars = 0
            for i, clus in enumerate(clusters):
                f = open(clus)
                data = csv.reader(f)
                pars_idx = [int(d[0]) for d in data]
                clus_values[i] = pars_idx
                number_pars += len(pars_idx)
            # self.clusters is a list of lists that contains the index of the parameter values that belong to different
            # clusters
            self.clusters = clus_values
            self.number_pars = number_pars
        elif type(clusters) == str:
            f = open(clusters)
            data = csv.reader(f)
            pars_clusters = np.array([int(d[0]) for d in data])
            num_of_clusters = set(pars_clusters)
            clus_values = {}
            for j in num_of_clusters:
                item_index = np.where(pars_clusters == j)
                clus_values[j] = item_index[0]
            self.clusters = clus_values
            self.number_pars = len(pars_clusters)
        elif clusters is None:
            no_clusters = {0: range(len(self.all_parameters))}
            self.clusters = no_clusters
            self.number_pars = len(self.all_parameters)
        else:
            raise Exception('wrong type')

        if sim_results is not None:
            self.all_simulations = sim_results
            if len(self.tspan) != self.all_simulations.shape[1]:
                raise Exception("'tspan' must be the same length as sim_results")
            if self.number_pars != self.all_simulations.shape[0]:
                raise Exception("The number of simulations must be the same as the number of parameters provided")
        else:
            self.all_simulations = self.sim.run(param_values=self.all_parameters).all

    @staticmethod
    def curve_fit_ftn(functions, species, xdata, ydata, **kwargs):
        """

        :param functions: list of functions that would be used for fitting the data
        :param species: species whose trajectories will be fitted
        :param xdata: x-axis data points (usually time span)
        :param ydata: y-axis data points (usually concentration of species in time)
        :param kwargs: Key arguments for curve_fit
        :return: array of optimized parameters
        """
        if callable(functions):
            functions = [functions]
        if isinstance(species, int):
            species = [species]
        results = [0] * len(species)
        for i, j in enumerate(species):
            results[i] = curve_fit(functions[i], xdata, ydata['__s{0}'.format(j)], p0=kwargs['p0'])[0]
        return results[0]

    def plot_dynamics_cluster_types(self, species, save_path='', species_to_fit=None, fit_ftn=None, norm=False, **kwargs):
        """

        :param species: Species that will be plotted
        :param save_path: path to file to save figures
        :param species_to_fit: index of species whose trajectory would be fitted to a function (fit_ftn)
        :param fit_ftn: Functions that will be used to fit the simulation results
        :param norm: Optional, boolean to normalize species by max value in simulation
        :param kwargs:
        :return:
        """

        # creates a dictionary to store the different figures by cluster
        plots_dict = {}
        for sp in species:
            for clus in self.clusters:
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)] = plt.subplots()

        if norm:
            if species_to_fit:
                # checking if species_to_fit are present in the species of the model
                sp_overlap = [ii for ii in species_to_fit if ii in species]
                if not sp_overlap:
                    raise Exception('species_to_fit must be in model.species')

                for idx, clus in self.clusters.items():
                    ftn_result = []
                    for i, par_idx in enumerate(clus):
                        y = self.all_simulations[par_idx]
                        try:
                            result = (self.curve_fit_ftn(functions=fit_ftn, species=species_to_fit, xdata=self.tspan,
                                                         ydata=y, **kwargs))
                        except:
                            print ("Trajectory {0} can't be fitted".format(par_idx))
                        ftn_result.append(result)
                        for i_sp, sp in enumerate(species):
                            sp_max = y['__s{0}'.format(sp)].max()
                            plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan,
                                                                                        y['__s{0}'.format(sp)] / sp_max,
                                                                                        color='blue', alpha=0.2)
                    for ind, sp_dist in enumerate(species_to_fit):
                        ax = plots_dict['plot_sp{0}_cluster{1}'.format(sp_dist, idx)][1]
                        divider = make_axes_locatable(ax)
                        axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
                        # axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
                        plt.setp(axHistx.get_xticklabels(),
                                 visible=False)  # + axHisty.get_yticklabels(), visible=False)

                        hist_data = hf.column(ftn_result, 1)
                        hist_data_filt = hist_data[(hist_data > 0) & (hist_data < self.tspan[-1])]

                        shape, loc, scale = lognorm.fit(hist_data_filt, floc=0)
                        pdf = lognorm.pdf(np.sort(hist_data_filt), shape, loc, scale)
                        # plt.figure()
                        # plt.plot(np.sort(hist_data_filt), pdf)
                        # plt.hist(hist_data_filt, normed=True, bins=20)
                        pdf_pars = 'sigma = '+str(round(shape, 2))+'\nmu = '+str(round(scale, 2))
                        anchored_text = AnchoredText(pdf_pars, loc=1)
                        axHistx.add_artist(anchored_text)
                        # weightsx = np.ones_like(hist_data_filt) / len(hist_data_filt)
                        axHistx.hist(hist_data_filt, normed=True, bins=20)
                        axHistx.plot(np.sort(hist_data_filt), pdf)
                        for tl in axHistx.get_xticklabels():
                            tl.set_visible(False)
                        yticks = [v for v in np.linspace(0, pdf.max(), 3)]
                        axHistx.set_ylim(0, 1.5e-3)
                        axHistx.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            else:
                for idx, clus in self.clusters.items():
                    y = self.all_simulations[clus]
                    for i_sp, sp in enumerate(species):
                        norm_trajectories = np.divide(y['__s{0}'.format(sp)].T, np.amax(y['__s{0}'.format(sp)], axis=1))
                        plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan,
                                                                                    norm_trajectories,
                                                                                    color='blue',
                                                                                    alpha=0.2)
        else:
            for idx, clus in self.clusters.items():
                y = self.all_simulations[clus].T
                for i_sp, sp in enumerate(species):
                    plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan, y['__s{0}'.format(sp)],
                                                                                color='blue',
                                                                                alpha=0.2)
        for ii, sp in enumerate(species):
            for clus in self.clusters:
                # ax = plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1]
                # divider = make_axes_locatable(ax)
                # # axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
                # axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
                # plt.setp(axHisty.get_yticklabels(), visible=False)
                #
                # hist_data = self.column(self.all_simulations[self.clusters[clus]]['__s{0}'.format(sp)], -1) / \
                #             self.column(self.all_parameters[self.clusters[clus]], ic_idx[ii])
                # hist_data_filt = hist_data[(hist_data > 0) & (hist_data < self.tspan[-1])]
                # weightsy = np.ones_like(hist_data_filt) / len(hist_data_filt)
                # # weightsy = np.ones_like(cparp_info_fraction) / len(cparp_info_fraction)
                # axHisty.hist(hist_data_filt, weights=weightsy, orientation='horizontal')
                # for tl in axHisty.get_yticklabels():
                #     tl.set_visible(False)
                # axHisty.set_xticks([0, 0.5, 1])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_xlabel('Time')
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_ylabel('Concentration')
                # plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_xlim([0, 8])
                # plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_ylim([0, 1])
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][0].suptitle('Species {0} cluster {1}'.
                                                                                 format(sp, clus))
                final_save_path = os.path.join(save_path, 'plot_sp{0}_cluster{1}'.format(sp, clus))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][0].savefig(final_save_path)
        return

    def plot_sp_ic_distributions(self, ic_par_idxs, save_path=''):
        """
        Creates a histogram for each cluster of the initial conditions provided

        :param ic_par_idxs: index of the initial condition in model.parameters
        :param save_path: path to save the file
        :return:
        """
        colors = self._get_colors(len(ic_par_idxs))
        for c_idx, clus in self.clusters.items():
            cluster_pars = self.all_parameters[clus]
            plt.figure(1)
            for idx, sp_ic in enumerate(ic_par_idxs):
                sp_ic_values = cluster_pars[:, sp_ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values)
                plt.hist(sp_ic_values, weights=sp_ic_weights, alpha=0.4, color=colors[idx],
                         label=self.model.parameters[sp_ic].name)
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            # plt.ylim([0, 0.4])
            # plt.xlim([0, 100000])
            plt.legend(loc=0)
            final_save_path = os.path.join(save_path, 'plot_ic_type{0}'.format(c_idx))
            plt.savefig(final_save_path)
            plt.clf()
        return

    def plot_clusters_ic_distributions(self, ic_par_idxs, save_path=''):
        colors = self._get_colors(len(ic_par_idxs))

        for idx, sp_ic in enumerate(ic_par_idxs):
            plt.figure(1)
            for c_idx, clus in self.clusters.items():
                cluster_pars = self.all_parameters[clus]
                sp_ic_values = cluster_pars[:, sp_ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values)
                plt.hist(sp_ic_values, weights=sp_ic_weights, alpha=0.4,
                         label=str(c_idx))
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            # plt.ylim([0, 0.4])
            # plt.xlim([0, 100000])
            plt.legend(loc=0)
            final_save_path = os.path.join(save_path, 'plot_sp_{0}'.format(self.model.parameters[sp_ic].name))
            plt.savefig(final_save_path)
            plt.clf()
        return

    def plot_sp_ic_overlap(self, ic_par_idxs, save_path=''):
        """
        Creates a stacked histogram with the distributions of each of the clusters for each initial condition provided

        :param ic_par_idxs: list, index of the initial conditions in model.parameters
        :param save_path: path to save the file
        :return:
        """
        if type(ic_par_idxs) == int:
            ic_par_idxs = [ic_par_idxs]

        for ic in ic_par_idxs:
            plt.figure()
            sp_ic_values_all = self.all_parameters[:, ic]
            sp_ic_weights_all = np.ones_like(sp_ic_values_all) / len(sp_ic_values_all)
            n, bins, patches = plt.hist(sp_ic_values_all, weights=sp_ic_weights_all, bins=30, fill=False)

            cluster_ic_values = []
            cluster_ic_weights = []
            for c_idx, clus in self.clusters.items():
                cluster_pars = self.all_parameters[clus]
                sp_ic_values = cluster_pars[:, ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values_all)
                cluster_ic_values.append(sp_ic_values)
                cluster_ic_weights.append(sp_ic_weights)

            label = ['cluster_{0}, {1}%'.format(cl, (len(self.clusters[cl])/self.number_pars)*100)
                     for cl in self.clusters.keys()]
            plt.hist(cluster_ic_values, bins=bins, weights=cluster_ic_weights, stacked=True, label=label,
                     histtype='bar', ec='black')
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            plt.title(self.model.parameters[ic].name)
            plt.legend(loc=0)

            final_save_path = os.path.join(save_path, 'plot_ic_overlap_{0}'.format(ic))
            plt.savefig(final_save_path)
        return

    def scatter_plot_pars(self, ic_par_idxs, cluster,  save_path=''):

        if type(cluster) == int:
            cluster_idxs = self.clusters[cluster]

        sp_ic_values1 = self.all_parameters[cluster_idxs, ic_par_idxs[0]]
        sp_ic_values2 = self.all_parameters[cluster_idxs, ic_par_idxs[1]]
        plt.figure()
        plt.scatter(sp_ic_values1, sp_ic_values2)
        ic_name0 = self.model.parameters[ic_par_idxs[0]].name
        ic_name1 = self.model.parameters[ic_par_idxs[1]].name
        plt.xlabel(ic_name0)
        plt.ylabel(ic_name1)
        final_save_path = os.path.join(save_path, 'scatter plot {0} and {1}, cluster {2}'.format(ic_name0, ic_name1,
                                                                                                 cluster))
        plt.savefig(final_save_path)

    @staticmethod
    def _get_colors(num_colors):
        colors = []
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i / 360.
            lightness = (50 + np.random.rand() * 10) / 100.
            saturation = (90 + np.random.rand() * 10) / 100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors
