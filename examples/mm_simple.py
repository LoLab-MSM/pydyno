from pysb import *
from pysb.integrate import odesolve
from pysb.bng import run_ssa
from pysb.macros import catalyze_state
import numpy as np
import matplotlib.pyplot as plt

run_ode = True
n_ssa_runs = 5
colors = ['blue', 'green', 'red', 'orange']

Model()
#######
V = 10.
#######
Parameter('kf',   1./V)
Parameter('kr',   1000.)
Parameter('kcat', 100.)

Monomer('E', ['s'])
Monomer('S', ['e'])
Monomer('P')

# Rules
Rule('ReversibleBinding', E(s=None) + S(e=None) <> E(s=1) % S(e=1), kf, kr)
Rule('Production', E(s=1) % S(e=1) >> E(s=None) + P(), kcat)

# Macro
# catalyze_state(E(), 's', S(), 'e', 'state', '0', '1', [kf,kr,kcat])

Observable("E_free",     E(s=None))
Observable("S_free",     S(e=None))
Observable("ES_complex", E(s=1) % S(e=1))
Observable("Product",    P())

Parameter("Etot", 1.*V)
Initial(E(s=None), Etot)

Parameter('S0', 10.*V)
Initial(S(e=None), S0)

t = np.linspace(0, 50, 501)

for i in range(n_ssa_runs):
    print i
    x = run_ssa(model, t[-1], len(t)-1, verbose=True)
    for j,obs in enumerate(model.observables):
        if j == len(model.observables)-1:
            plt.figure('product')
        else:
            plt.figure('substrate_complexes')
        if i == 0:
            plt.plot(t, x[obs.name], label=obs.name, lw=3, color=colors[j])
        else:
            plt.plot(t, x[obs.name], lw=3, color=colors[j])

if run_ode:
    x = odesolve(model, t, verbose=False)    
    for j,obs in enumerate(model.observables):
        if j == len(model.observables)-1:
            plt.figure('product')
        else:
            plt.figure('substrate_complexes')
        plt.plot(t, x[obs.name], 'k--', lw=5)

for fig in ['product', 'substrate_complexes']:
    plt.figure(fig)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Copy number', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc=0, fontsize=18)
    plt.annotate("V = %d" % V, (0.7, 0.5), xycoords='axes fraction', fontsize=32, color='k')

plt.show()
