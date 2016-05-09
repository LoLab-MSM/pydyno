from pysb import *
from pysb.integrate import odesolve
import numpy as np
import matplotlib.pyplot as plt

Model()
#######
V = 10.
#######
Parameter('kf',   1./V)
Parameter('kr',   100.)
Parameter('kcat', 1000.)

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

Parameter('S0', 1000.*V)
Initial(S(e=None), S0)


tspan = np.linspace(0, 20, 100)
y = odesolve(model, tspan)

color = ['b', 'r', 'b', 'g']

for i, j in enumerate(model.observables):
    plt.figure()
    plt.plot(tspan, y[j.name], label=j.name, linewidth=4, color=color[i])
    plt.xlabel('time', fontsize=20)
    plt.ylabel('Population', fontsize=20)
    plt.legend(loc=0)
    plt.savefig('/home/oscar/Desktop/mm%d' % i + '.eps')
plt.show()
