from pysb import *
from pysb.integrate import odesolve
from pysb.bng import run_ssa
from pysb.macros import catalyze_state
import numpy as np
import matplotlib.pyplot as plt

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

Parameter('S0', 1000.*V)
Initial(S(e=None), S0)
