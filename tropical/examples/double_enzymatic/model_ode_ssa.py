from pysb import *
from pysb.integrate import odesolve
from pysb.bng import run_ssa
from pysb.macros import catalyze_state
import numpy as np
import matplotlib.pyplot as plt

run_ode = True
n_ssa_runs = 5
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']

Model()
#######
V = 10.
#######
Parameter('kf1',   1./V)
Parameter('kr1',   1000.)
Parameter('kcat1', 10.)
Parameter('kf2',   1./V)
Parameter('kr2',   1000.)
Parameter('kcat2', 100.)


Monomer('E', ['s'])
Monomer('S', ['e', 'type'], {'type': ['1','2']})
Monomer('P')

# Rules
Rule('ReversibleBinding_1', E(s=None) + S(e=None, type='1') | E(s=1) % S(e=1, type='1'), kf1, kr1)
Rule('Production_1', E(s=1) % S(e=1, type='1') >> E(s=None) + P(), kcat1)
Rule('ReversibleBinding_2', E(s=None) + S(e=None, type='2') | E(s=1) % S(e=1, type='2'), kf2, kr2)
Rule('Production_2', E(s=1) % S(e=1, type='2') >> E(s=None) + P(), kcat2)

# Macro
# catalyze_state(E(), 's', S(), 'e', 'state', '0', '1', [kf,kr,kcat])

Observable("E_free",      E(s=None))
Observable("S1_free",     S(e=None, type='1'))
Observable("S2_free",     S(e=None, type='2'))
Observable("ES1_complex", E(s=1) % S(e=1, type='1'))
Observable("ES2_complex", E(s=1) % S(e=1, type='2'))
Observable("Product",     P())

Parameter("Etot", 1.*V)
Initial(E(s=None), Etot)

Parameter('S1_0', 10.*V)
Initial(S(e=None, type='1'), S1_0)

Parameter('S2_0', 10.*V)
Initial(S(e=None, type='2'), S2_0)

t = np.linspace(0, 50, 51)

# for i in range(n_ssa_runs):
#     print i
#     x = run_ssa(model, t[-1], len(t)-1, verbose=False)
#     for j,obs in enumerate(model.observables):
#         if j == len(model.observables)-1:
#             plt.figure('product')
#         else:
#             plt.figure('substrate_complexes')
#         if i == 0:
#             plt.plot(t, x[obs.name], label=obs.name, lw=3, color=colors[j])
#         else:
#             plt.plot(t, x[obs.name], lw=3, color=colors[j])

if run_ode:
    x = odesolve(model, t, verbose=True)
    np.save('product_data', x['Product'])
    for j,obs in enumerate(model.observables):
        if j == len(model.observables)-1:
            plt.figure('product')
        else:
            plt.figure('substrate_complexes')
        plt.plot(t, x[obs.name], 'k--', color=colors[j], label=obs.name, lw=5)

for fig in ['product', 'substrate_complexes']:
    plt.figure(fig)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Copy number', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc=0, fontsize=18)
    # plt.annotate("V = %d" % V, (0.5, 0.9), xycoords='axes fraction', fontsize=32, color='k')

    plt.savefig(fig, bbox_inches='tight')