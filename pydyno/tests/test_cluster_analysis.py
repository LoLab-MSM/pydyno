from pydyno.examples.double_enzymatic.mm_two_paths_model import model
# from nose.tools import *
import numpy as np
from pysb.testing import *
import os
from pysb.simulator.scipyode import ScipyOdeSimulator
from pydyno.visualize_trajectories import VisualizeTrajectories


class TestClusteringAnalysisBase(object):
    @classmethod
    def tearDownClass(cls):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        test = os.listdir(dir_name)
        for item in test:
            if item.endswith(".png") or item.endswith(".pdf"):
                os.remove(os.path.join(dir_name, item))

    def setUp(self):
        pars = [[3.42477815e-02, 3.52565624e+02, 1.04957728e+01,6.35198054e-02, 3.14044789e+02,
                5.28598128e+01, 1.00000000e+01, 1.00000000e+02, 1.00000000e+02],
                [9.13676502e-03, 2.07253754e+00, 1.37740528e+02, 2.19960625e-01, 1.15007005e+04,
                5.16232342e+01, 1.00000000e+01, 1.00000000e+02, 1.00000000e+02]]
        tspan = np.linspace(0, 50, 101)
        labels = [1, 0]
        self.model = model
        sims = ScipyOdeSimulator(self.model, tspan=tspan, param_values=pars).run()
        self.clus = VisualizeTrajectories(self.model, clusters=labels, sim_results=sims)

    def tearDown(self):
        self.signatures = None
        self.clus = None


class TestClusteringAnalysisSingle(TestClusteringAnalysisBase):
    def test_plot_dynamics_cluster_not_normed_types(self):
        self.clus.plot_cluster_dynamics(species=[0], norm=False)

    def test_plot_dynamics_cluster_normed_types(self):
        self.clus.plot_cluster_dynamics(species=[0], norm=True)

    def test_plot_dynamics_cluster_not_normed_expression(self):
        exprs = self.model.reactions_bidirectional[0]['rate']
        self.clus.plot_cluster_dynamics(species=[exprs], norm=False)

    def test_plot_dynamics_cluster_not_normed_observable(self):
        self.clus.plot_cluster_dynamics(species=['E_free'], norm=False)

    def test_plot_dynamics_cluster_normed_expression(self):
        exprs = self.model.reactions_bidirectional[0]['rate']
        self.clus.plot_cluster_dynamics(species=[exprs], norm=True)

    def test_plot_dynamics_cluster_normed_observable(self):
        self.clus.plot_cluster_dynamics(species=['E_free'], norm=True)

    @raises(ValueError)
    def test_plot_dynamics_cluster_invalid_observable(self):
        self.clus.plot_cluster_dynamics(species=['bla'], norm=True)

    def test_hist_avg_sps_bar(self):
        sp = self.model.species[0]
        self.clus.plot_pattern_sps_distribution(pattern=sp, type_fig='bar')

    def test_hist_avg_sps_entropy(self):
        sp = self.model.species[0]
        self.clus.plot_pattern_sps_distribution(pattern=sp, type_fig='entropy')

    @raises(NotImplementedError)
    def test_hist_avg_sps_invalid_visualization(self):
        sp = self.model.species[0]
        self.clus.plot_pattern_sps_distribution(pattern=sp, type_fig='bla')

    def test_hist_avg_rxn_bar(self):
        sp = self.model.species[0]
        self.clus.plot_pattern_rxns_distribution(pattern=sp, type_fig='bar')

    def test_hist_avg_rxn_entropy(self):
        sp = self.model.species[0]
        self.clus.plot_pattern_rxns_distribution(pattern=sp, type_fig='entropy')

    @raises(NotImplementedError)
    def test_hist_avg_rxns_invalid_visualization(self):
        sp = self.model.species[0]
        self.clus.plot_pattern_rxns_distribution(pattern=sp, type_fig='bla')

    def test_violin_plot_kd(self):
        kd_pars = [(1, 0)]
        self.clus.plot_violin_kd(par_idxs=kd_pars)

    def test_hist_plot_clusters(self):
        self.clus.hist_clusters_parameters(par_idxs=[7])

    def test_violin_plot_sps(self):
        self.clus.plot_violin_pars(par_idxs=[0])

    def test_plot_sp_ic_overlap(self):
        self.clus.plot_sp_ic_overlap(par_idxs=[7])

    def test_scatter_plot_pars(self):
        self.clus.scatter_plot_pars(par_idxs=[7, 8], cluster=0)