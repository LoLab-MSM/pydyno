import numpy as np
import csv
from pysb.integrate import ScipyOdeSimulator
import matplotlib.pyplot as plt
import colorsys
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

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
        if sim_results is not None:
            self.all_simulations = sim_results
            if len(self.tspan) != self.all_simulations.shape[1]:
                raise Exception("'tspan' must be the same length as sim_results")
        if type(parameters) == str:
            self.all_parameters = np.load(parameters)
        elif type(parameters) == np.ndarray:
            self.all_parameters = parameters
        else:
            raise Exception('A valid set of parameters must be provided')
        if self.all_parameters.shape[1] != len(self.model.parameters):
            raise Exception("param_values must be the same length as model.parameters")

        if type(clusters) == list:
            clus_values = {}
            for i, clus in enumerate(clusters):
                f = open(clus)
                data = csv.reader(f)
                pars_idx = [int(d[0]) for d in data]
                clus_values[i] = pars_idx
            # self.clusters is a list of lists that contains the index of the parameter values that belong to different
            # clusters
            self.clusters = clus_values
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
        else:
            raise Exception('wrong type')

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

    @staticmethod
    def column(matrix, i):
        """Return the i column of a matrix

        Keyword arguments:
        matrix -- matrix to get the column from
        i -- column to get fro the matrix
        """
        return np.array([row[i] for row in matrix])

    def plot_dynamics_cluster_types(self, species, save_path='', species_to_fit=None, fit_ftn=None, ic_idx=False, **kwargs):
        """

        :param species: Species that will be plotted
        :param save_path: path to file to save figures
        :param species_to_fit: index of species whose trajectory would be fitted to a function (fit_ftn)
        :param fit_ftn: Functions that will be used to fit the simulation results
        :param ic_idx: Optional, index in model.parameters to normalize species
        :param kwargs:
        :return:
        """
        if self.all_simulations is None:
            self.all_simulations = self.sim.run(param_values=self.all_parameters).all

        # creates a dictionary to store the different figures by cluster
        plots_dict = {}
        for sp in species:
            for clus in self.clusters:
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)] = plt.subplots()

        if ic_idx:
            if species_to_fit:
                # checking if species_to_fit are present in the species of the model
                sp_overlap = [ii for ii in species_to_fit if ii in species]
                if not sp_overlap:
                    raise Exception('species_to_fit must be in species')

                for idx, clus in self.clusters.items():
                    ftn_result = [0] * len(clus)
                    for i, par_idx in enumerate(clus):
                        y = self.all_simulations[par_idx]
                        ftn_result[i] = (self.curve_fit_ftn(functions=fit_ftn, species=species_to_fit, xdata=self.tspan,
                                                            ydata=y, **kwargs))
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

                        hist_data = self.column(ftn_result, 1)
                        hist_data_filt = hist_data[(hist_data > 0) & (hist_data < self.tspan[-1])]
                        # lognorm_fit = lognorm.fit(hist_data_filt)
                        # print (lognorm_fit)
                        # ax.text(7, 0.8, str(lognorm_fit))
                        weightsx = np.ones_like(hist_data_filt) / len(hist_data_filt)
                        axHistx.hist(hist_data_filt, weights=weightsx, bins=20)
                        for tl in axHistx.get_xticklabels():
                            tl.set_visible(False)
                        axHistx.set_yticks([0, 0.5, 1])
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

    def plot_sp_IC_distributions(self, ic_par_idxs, save_path=''):
        colors = self._get_colors(len(ic_par_idxs))
        for c_idx, clus in enumerate(self.clusters):
            cluster_pars = self.all_parameters[clus]
            plt.figure(1)
            for idx, sp_ic in enumerate(ic_par_idxs):
                sp_ic_values = cluster_pars[:, sp_ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values)
                plt.hist(sp_ic_values, weights=sp_ic_weights, alpha=0.4, color=colors[idx],
                         label=self.model.parameters[sp_ic].name)
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            plt.ylim([0, 0.4])
            plt.xlim([0, 100000])
            plt.legend(loc=0)
            final_save_path = os.path.join(save_path, 'plot_ic_type{0}'.format(c_idx))
            plt.savefig(final_save_path)
            plt.clf()
        return

    @staticmethod
    def _get_colors(num_colors):
        colors = []
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i / 360.
            lightness = (50 + np.random.rand() * 10) / 100.
            saturation = (90 + np.random.rand() * 10) / 100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors
