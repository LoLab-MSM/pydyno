import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool, cpu_count
from earm.lopez_embedded import model
from pysb.simulator import ScipyOdeSimulator
from tropical import helper_functions as hf
import os

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'parameters_5000')

tspan = np.linspace(0, 20000, 100)
sim = ScipyOdeSimulator(model, tspan)


def c3_result(parameter):
    y = sim.run(param_values=parameter).species
    return y[:, 33]


def display_c3(parameters, new_path=None):
    params = hf.read_all_pars(parameters, new_path)
    p = Pool(cpu_count())
    all_c3_traces = np.asarray(p.map(c3_result, params.as_matrix()))
    fig, axApop = plt.subplots(figsize=(5.5, 5.5))
    for i in all_c3_traces:
        axApop.plot(tspan, i)

    divider = make_axes_locatable(axApop)
    axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=axApop)
    # make some labels invisible
    plt.setp(axHisty.get_yticklabels(), visible=False)
    # now determine nice limits by hand:
    binwidth = 1000
    xymax = np.max([np.max(np.fabs(tspan)), np.max(np.fabs(1))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    weightsy = np.ones_like(all_c3_traces[:, 50]) / len(all_c3_traces[:, 50])
    axHisty.hist(all_c3_traces[:, 50], orientation='horizontal', bins=np.arange(0, 10000 + binwidth, binwidth),
                 weights=weightsy)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
    axHisty.set_xticks([0, 0.5, 1])
    plt.show()
    return


new_path1 = '/home/oscar/Documents/tropical_earm/parameters_5000'
pars_cluster1 = '/home/oscar/Documents/tropical_earm/clustered_parameters_bid_consumption/data_frame37_Type3'
a = display_c3(pars_cluster1, new_path1)