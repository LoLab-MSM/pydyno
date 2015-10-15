from mm_two_paths_model import model 
from pysb.bng import generate_equations
from pysb.integrate import odesolve
import matplotlib.pyplot as plt
generate_equations(model)
print model.species


# from max_monomials_signature import run_tropical
from tropicalize import run_tropical
import numpy as np
import matplotlib.pyplot as plt


tspan = np.linspace(0, 5, 51)
# y=odesolve(model,tspan)
# plt.plot(tspan,1000.0*y['__s2']/y['__s0'])
# plt.plot(tspan,y['__s2'])
# plt.plot(tspan,y['__s0'])
# plt.plot(tspan,y['__s1'])
#   
# plt.show()
# quit()
# run_tropical(model, tspan, sp_visualize=[3,5], stoch=True)
print run_tropical(model, tspan, sp_visualize=None)


