from earm2_flat import model
from tropical.analysis_of_clusters import AnalysisCluster
from tropical import util
import pickle

with open('results_spectral.pickle', 'rb') as handle:
    clusters = pickle.load(handle)
clusters_sp37 = clusters[13]['labels']

sim_0_good = 'sim_kd/earm_scipyode_sims_good0.h5'
sim_0_bad = 'sim_kd/earm_scipyode_sims_bad0.h5'
sim_1_good = 'sim_kd/earm_scipyode_sims_good1.h5'
sim_1_bad = 'sim_kd/earm_scipyode_sims_bad1.h5'
sim_2_good = 'sim_kd/earm_scipyode_sims_good2.h5'
sim_2_bad = 'sim_kd/earm_scipyode_sims_bad2.h5'
sim_3_good = 'sim_kd/earm_scipyode_sims_good3.h5'
sim_3_bad = 'sim_kd/earm_scipyode_sims_bad3.h5'
sim_4_good = 'sim_kd/earm_scipyode_sims_good4.h5'
sim_4_bad = 'sim_kd/earm_scipyode_sims_bad4.h5'
sim_5_good = 'sim_kd/earm_scipyode_sims_good5.h5'
sim_5_bad = 'sim_kd/earm_scipyode_sims_bad5.h5'
sim_6_good = 'sim_kd/earm_scipyode_sims_good6.h5'
sim_6_bad = 'sim_kd/earm_scipyode_sims_bad6.h5'
a = AnalysisCluster(model, clusters=None, sim_results=sim_0_good)

# a.hist_plot_clusters([82, 83, 84, 85, 86, 87], save_path='figures/')
# a.plot_sp_ic_overlap([82, 83, 84, 85, 86, 87], save_path='figures/')
# a.violin_plot_sps([82, 83, 84, 85, 86, 87], save_path='figures/')
a.plot_dynamics_cluster_types([39], save_path='figures/', species_ftn_fit={39: util.sig_apop},
                              norm=True, **{'p0': [100, 100, 100]})

# a.scatter_plot_pars(ic_par_idxs=[82, 85, 86, 87], cluster=3, save_path='figures/')