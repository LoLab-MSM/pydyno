from pydyno.examples.double_enzymatic.mm_two_paths_model import model
import numpy as np
from pysb.simulator import ScipyOdeSimulator
from pydyno.visualize_simulations import VisualizeSimulations
import pytest


@pytest.fixture(scope='class')
def simulation():
    pars = [[3.42477815e-02, 3.52565624e+02, 1.04957728e+01, 6.35198054e-02, 3.14044789e+02,
             5.28598128e+01, 1.00000000e+01, 1.00000000e+02, 1.00000000e+02],
            [9.13676502e-03, 2.07253754e+00, 1.37740528e+02, 2.19960625e-01, 1.15007005e+04,
             5.16232342e+01, 1.00000000e+01, 1.00000000e+02, 1.00000000e+02]]
    tspan = np.linspace(0, 50, 101)
    sims = ScipyOdeSimulator(model, tspan=tspan, param_values=pars).run()
    return sims


class TestVisualizationInitialization:
    def test_clusters_none(self, simulation):
        VisualizeSimulations(model, clusters=None, sim_results=simulation)

    def test_truncate_idx(self, simulation):
        VisualizeSimulations(model, clusters=None, sim_results=simulation, truncate_idx=0)

    def test_drop_sim_idx(self, simulation):
        VisualizeSimulations(model, clusters=None, sim_results=simulation, drop_sim_idx=[1])

    def test_truncate_idx_drop_sim_idx(self, simulation):
        with pytest.raises(ValueError):
            VisualizeSimulations(model, clusters=None, sim_results=simulation, truncate_idx=0, drop_sim_idx=[1])


@pytest.fixture(scope='class')
def visualize():
    pars = [[3.42477815e-02, 3.52565624e+02, 1.04957728e+01, 6.35198054e-02, 3.14044789e+02,
             5.28598128e+01, 1.00000000e+01, 1.00000000e+02, 1.00000000e+02],
            [9.13676502e-03, 2.07253754e+00, 1.37740528e+02, 2.19960625e-01, 1.15007005e+04,
             5.16232342e+01, 1.00000000e+01, 1.00000000e+02, 1.00000000e+02]]
    tspan = np.linspace(0, 50, 101)
    labels = [1, 0]
    sims = ScipyOdeSimulator(model, tspan=tspan, param_values=pars).run()
    clus = VisualizeSimulations(model, clusters=labels, sim_results=sims)
    return clus


class TestClusteringAnalysis:
    def test_plot_dynamics_cluster_not_normed_types(self, visualize, data_files_dir):
        visualize.plot_cluster_dynamics(components=[0], norm=False, dir_path=data_files_dir)

    def test_plot_dynamics_cluster_normed_types(self, visualize, data_files_dir):
        visualize.plot_cluster_dynamics(components=[0], norm=True, dir_path=data_files_dir)

    def test_plot_dynamics_cluster_not_normed_expression(self, visualize, data_files_dir):
        exprs = model.reactions_bidirectional[0]['rate']
        visualize.plot_cluster_dynamics(components=[exprs], norm=False, dir_path=data_files_dir)

    def test_plot_dynamics_cluster_not_normed_observable(self, visualize, data_files_dir):
        visualize.plot_cluster_dynamics(components=['E_free'], norm=False, dir_path=data_files_dir)

    def test_plot_dynamics_cluster_normed_expression(self, visualize, data_files_dir):
        exprs = model.reactions_bidirectional[0]['rate']
        visualize.plot_cluster_dynamics(components=[exprs], norm=True, dir_path=data_files_dir)

    def test_plot_dynamics_cluster_normed_observable(self, visualize, data_files_dir):
        visualize.plot_cluster_dynamics(components=['E_free'], norm=True, dir_path=data_files_dir)

    def test_plot_dynamics_cluster_invalid_observable(self, visualize, data_files_dir):
        with pytest.raises(ValueError):
            visualize.plot_cluster_dynamics(components=['bla'], norm=True, dir_path=data_files_dir)

    def test_hist_avg_sps_bar(self, visualize, data_files_dir):
        sp = model.species[0]
        visualize.plot_pattern_sps_distribution(pattern=sp, type_fig='bar', dir_path=data_files_dir)

    def test_hist_avg_sps_entropy(self, visualize, data_files_dir):
        sp = model.species[0]
        visualize.plot_pattern_sps_distribution(pattern=sp, type_fig='entropy', dir_path=data_files_dir)

    def test_hist_avg_sps_invalid_visualization(self, visualize, data_files_dir):
        with pytest.raises(NotImplementedError):
            sp = model.species[0]
            visualize.plot_pattern_sps_distribution(pattern=sp, type_fig='bla', dir_path=data_files_dir)

    def test_hist_avg_rxn_bar(self, visualize, data_files_dir):
        sp = model.species[0]
        visualize.plot_pattern_rxns_distribution(pattern=sp, type_fig='bar', dir_path=data_files_dir)

    def test_hist_avg_rxn_entropy(self, visualize, data_files_dir):
        sp = model.species[0]
        visualize.plot_pattern_rxns_distribution(pattern=sp, type_fig='entropy', dir_path=data_files_dir)

    def test_hist_avg_rxns_invalid_visualization(self, visualize, data_files_dir):
        with pytest.raises(NotImplementedError):
            sp = model.species[0]
            visualize.plot_pattern_rxns_distribution(pattern=sp, type_fig='bla', dir_path=data_files_dir)

    def test_violin_plot_kd(self, visualize, data_files_dir):
        kd_pars = [(1, 0)]
        visualize.plot_violin_kd(par_idxs=kd_pars, dir_path=data_files_dir)

    def test_hist_plot_clusters(self, visualize, data_files_dir):
        visualize.hist_clusters_parameters(par_idxs=[7], dir_path=data_files_dir)

    def test_violin_plot_sps(self, visualize, data_files_dir):
        visualize.plot_violin_pars(par_idxs=[0], dir_path=data_files_dir)

    def test_plot_sp_ic_overlap(self, visualize, data_files_dir):
        visualize.plot_sp_ic_overlap(par_idxs=[7], dir_path=data_files_dir)

    def test_scatter_plot_pars(self, visualize, data_files_dir):
        visualize.scatter_plot_pars(par_idxs=[7, 8], cluster=0, dir_path=data_files_dir)
