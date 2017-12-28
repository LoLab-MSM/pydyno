import numpy as np
from pysb.simulator.scipyode import ScipyOdeSimulator
from earm2_flat import model
import pickle
from pathos.multiprocessing import ProcessingPool as Pool

with open('results_spectral.pickle', 'rb') as handle:
    clus_info = pickle.load(handle)

# Use mBid clustering labels
clus_sp37 = clus_info[13]['labels']
unique_labels = set(clus_sp37)
pars = np.load('calibrated_6572pars.npy')
pars_ref = pars[0]
tspan = np.linspace(0, 20000, 100)


def sims_kd(label):
    if label == 0:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 58] = pars_ref[58] * 0.8  # 80% Knock down of bcl2
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label2[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 1:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 64] = pars_ref[64] * 0.8  # 80% Knock down of bak
        pars_label1[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label2[:, 64] = pars_ref[64] * 0.8  # 80% Knock down of bak
        pars_label2[:, 58] = pars_ref[58] * 0.8  # 80% Knock down of bcl2
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 2:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 64] = pars_ref[64] * 0.8 # 80% Knock down of bak
        pars_label1[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 64] = pars_ref[64] * 0.8 # 80% Knock down of bak
        pars_label2[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 3:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label2[:, 58] = pars_ref[58] * 0.8  # 80% Knock down of bcl2
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 4:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 56] = pars_ref[56] * 0.8 # 80% Knock down of bclxl
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 57] = pars_ref[57] * 0.8 # 80% Knock down of mcl1
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 5:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 57] = pars_ref[57] * 0.8 # 80% Knock down of mcl1
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 58] = pars_ref[58] * 0.8 # 80% Knock down of bcl2
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 6:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 64] = pars_ref[64] * 0.8  # 80% Knock down of bak
        pars_label1[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        pars_label1[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label2[:, 64] = pars_ref[64] * 0.8  # 80% Knock down of bak
        pars_label2[:, 58] = pars_ref[58] * 0.8  # 80% Knock down of bcl2
        pars_label2[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 7:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 64] = pars_ref[64] * 0.8 # 80% Knock down of bak
        pars_label2[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of mcl1
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 8:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 64] = pars_ref[64] * 0.8  # 80% Knock down of bak
        pars_label1[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label2[:, 64] = pars_ref[64] * 0.8  # 80% Knock down of bak
        pars_label2[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 9:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label1[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        pars_label1[:, 57] = pars_ref[57] * 0.8  # 80% Knock down of mcl1
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        pars_label2[:, 56] = pars_ref[56] * 0.8  # 80% Knock down of bclxl
        pars_label2[:, 58] = pars_ref[58] * 0.8  # 80% Knock down of bcl2
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    if label == 10:
        pars_label1 = pars[np.where(clus_sp37 == label)]
        pars_label1[:, 63] = pars_ref[63] * 0.8 # 80% Knock down of bax
        sim1 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label1).run()
        sim1.save('earm_scipyode_sims_good{0}.h5'.format(label))

        pars_label2 = pars[np.where(clus_sp37 == label)]
        pars_label2[:, 64] = pars_ref[64] * 0.8 # 80% Knock down of bak
        sim2 = ScipyOdeSimulator(model, tspan=tspan, param_values=pars_label2).run()
        sim2.save('earm_scipyode_sims_bad{0}.h5'.format(label))

    return


p = Pool(25)
res = p.amap(sims_kd, unique_labels)
res.get()


