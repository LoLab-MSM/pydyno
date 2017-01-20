import numpy as np
import csv
from pysb.integrate import ScipyOdeSimulator
import matplotlib.pyplot as plt
import colorsys


class AnalysisCluster:

    def __init__(self, model, tspan, parameters, clusters):
        """

        :param model: PySB model
        :param tspan: time range for the simulation
        :param parameters: model parameters
        :param clusters: clusters from TroPy
        """
        self.model = model
        self.tspan = tspan
        self.sim = ScipyOdeSimulator(self.model, self.tspan)
        if type(parameters) == str:
            self.all_parameters = np.load(parameters)
        elif type(parameters) == np.ndarray:
            self.all_parameters = parameters
        else:
            raise Exception('Is this the right exception?')
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

    def plot_dynamics_cluster_types(self, species, save_path, ic_idx=None):
        """

        :param species: Species to plot their dynamics
        :param save_path: path to file
        :param ic_idx: indices in model.parameters of initial conditions species
        :return: plot of the dynamics of the species in each cluster
        """
        plots_dict = {}
        for sp in species:
            for clus in range(len(self.clusters)):
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)] = plt.subplots()

        if ic_idx:
            if len(species) != len(ic_idx):
                raise Exception("length of 'species' must be the same as of 'ic_idx'")
            for idx, clus in enumerate(self.clusters):
                for par_idx in clus:
                    parameters = self.all_parameters[par_idx]
                    y = self.sim.run(param_values=parameters).all
                    for i_sp, sp in enumerate(species):
                        sp_0 = parameters[ic_idx[i_sp]]
                        plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan, y['__s{0}'.format(sp)] /
                                                                                    sp_0)
        else:
            for idx, clus in enumerate(self.clusters):
                for par_idx in clus:
                    parameters = self.all_parameters[par_idx]
                    y = self.sim.run(param_values=parameters).all
                    for i_sp, sp in enumerate(species):
                        plots_dict['plot_sp{0}_cluster{1}'.format(sp, idx)][1].plot(self.tspan, y['__s{0}'.format(sp)])

        for sp in species:
            for clus in range(len(self.clusters)):
                plots_dict['plot_sp{0}_cluster{1}'.format(sp, clus)][0].savefig(save_path +
                                                                                '/plot_sp{0}_cluster{1}'.format(sp, clus))
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

