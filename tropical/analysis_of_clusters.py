import numpy as np
import csv
from pysb.integrate import ScipyOdeSimulator
import matplotlib.pyplot as plt
import colorsys
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()


class AnalysisCluster:

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
        if type(parameters) == str:
            self.all_parameters = np.load(parameters)
        elif type(parameters) == np.ndarray:
            self.all_parameters = parameters
        else:
            raise Exception('A valid set of parameters must be provided')
        if self.all_parameters.shape[1] != len(self.model.parameters):
            raise Exception("param_values must be the same length as model.parameters")

        if type(clusters) == list:
            clus_values = [0]*len(clusters)
            for i, clus in enumerate(clusters):
                f = open(clus)
                data = csv.reader(f)
                pars_idx = [int(d[0]) for d in data]
                clus_values[i] = pars_idx
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
        results = [0]*len(species)
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

    def plot_dynamics_cluster_types(self, species, save_path, species_to_fit=None, fit_ftn=None, ic_idx=None, **kwargs):
        """

        :param species: Species that will be plotted
        :param save_path: path to file to save figures
        :param species_to_fit:
        :param fit_ftn: Functions that will be used to fit the simulation results
        :param ic_idx: Optional, index in model.parameters to normalize species
        :param kwargs:
        :return:
        """
        if self.all_simulations is None:
            self.all_simulations = self.sim.run(param_values=self.all_parameters).all

        plots_dict = {}
        for sp in species:
            for clus in range(len(self.clusters)):
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)] = plt.subplots()

        if ic_idx:
            if len(species) != len(ic_idx):
                raise Exception("length of 'ic_dx' must be the same as of 'species'")
            if species_to_fit:
                sp_overlap = [ii for ii in species_to_fit if ii in species]
                if not sp_overlap:
                    raise Exception('species_dist must be in species')
                for idx, clus in enumerate(self.clusters):
                    ftn_result = [0]*len(clus)
                    for i, par_idx in enumerate(clus):
                        parameters = self.all_parameters[par_idx]
                        y = self.all_simulations[par_idx]
                        ftn_result[i] = (self.curve_fit_ftn(fit_ftn, species_to_fit, self.tspan, y, **kwargs))
                        for i_sp, sp in enumerate(species):
                            sp_0 = parameters[ic_idx[i_sp]]
                            plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan,
                                                                                        y['__s{0}'.format(sp)] / sp_0,
                                                                                        color='blue', alpha=0.2)
                    for ind, sp_dist in enumerate(species_to_fit):
                        ax = plots_dict['plot_sp{0}_cluster{1}'.format(sp_dist, idx)][1]
                        divider = make_axes_locatable(ax)
                        axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=ax)
                        # axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=ax)
                        plt.setp(axHistx.get_xticklabels(), visible=False) #+ axHisty.get_yticklabels(), visible=False)

                        hist_data = self.column(ftn_result, 1)
                        hist_data_filt = hist_data[(hist_data > 0) & (hist_data < self.tspan[-1])]
                        weightsx = np.ones_like(hist_data_filt) / len(hist_data_filt)
                        # weightsy = np.ones_like(cparp_info_fraction) / len(cparp_info_fraction)
                        axHistx.hist(hist_data_filt, weights=weightsx)
                        for tl in axHistx.get_xticklabels():
                            tl.set_visible(False)
                        axHistx.set_yticks([0, 0.5, 1])
            else:
                for idx, clus in enumerate(self.clusters):
                    for par_idx in clus:
                        parameters = self.all_parameters[par_idx]
                        y = self.all_simulations[par_idx]
                        for i_sp, sp in enumerate(species):
                            sp_0 = parameters[ic_idx[i_sp]]
                            plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan,
                                                                                        y['__s{0}'.format(sp)] / sp_0,
                                                                                        color='blue')
        else:
            for idx, clus in enumerate(self.clusters):
                for par_idx in clus:
                    y = self.all_simulations[par_idx]
                    for i_sp, sp in enumerate(species):
                        plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan, y['__s{0}'.format(sp)],
                                                                                    color='blue')
        for sp in species:
            for clus in range(len(self.clusters)):
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_xlabel('Time')
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][1].set_ylabel('Concentration')
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][0].suptitle('Species {0} cluster {1}'.
                                                                                 format(sp, clus))
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][0].savefig(save_path + '/plot_sp{0}_cluster{1}'.
                                                                                format(sp, clus))
        return

    def plot_sp_IC_distributions(self, ic_par_idxs, save_path):
        colors = self._get_colors(len(ic_par_idxs))
        for c_idx, clus in enumerate(self.clusters):
            cluster_pars = self.all_parameters[clus]
            plt.figure(1)
            for idx, sp_ic in enumerate(ic_par_idxs):
                sp_ic_values = cluster_pars[:, sp_ic]
                sp_ic_weights = np.ones_like(sp_ic_values) / len(sp_ic_values)
                plt.hist(sp_ic_values, weights=sp_ic_weights, alpha=0.4, color=colors[idx], label=str(ic_par_idxs))
            plt.xlabel('Concentration')
            plt.ylabel('Percentage')
            plt.savefig(save_path+'/plot_ic_type{0}'.format(c_idx))
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

