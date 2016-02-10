from pysb.examples.tyson_oscillator import model
# from max_monomials_signature import run_tropical
from tropicalize import run_tropical
import numpy as np
import matplotlib.pyplot as plt
import pysb

tspan = np.linspace(0, 200, 200)
# run_tropical(model, tspan, sp_visualize=[3,5], stoch=True)
print run_tropical(model, tspan, sp_visualize=None)


for idx, par in enumerate(model.parameters[:10]):
    plt.figure()
    rate_params = model.parameters_rules()
    param_values = np.array([p.value for p in model.parameters])
    rate_mask = np.array([p in rate_params for p in model.parameters])
    solver = pysb.integrate.Solver(model, tspan, integrator='vode')
    solver.run(param_values)
    nummol = np.copy(solver.y[51:52].T.reshape(6))
    plt.plot(tspan, solver.y[:, 5], 'o-', label='before')
    for i in np.linspace(0.1, 1.5, 6):
        rate_params = model.parameters_rules()
        param_values = np.array([p.value for p in model.parameters])
        param_values[idx] *= i
        solver.run(param_values, y0=nummol)
        plt.plot(tspan[51:], solver.y[:-51, 5], 'x-', label=str(i))
        plt.legend(loc=0)
        plt.tight_layout()
        plt.title(par.name+'t51')
        # plt.b
plt.show()
