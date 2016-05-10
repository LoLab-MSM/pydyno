import numpy as np
from pysb.examples.tyson_oscillator import model
import matplotlib.pyplot as plt
import pysb.integrate


# CHANGES IN PARAMETER VALUE AT CERTAIN TIME POINT
def change_parameter_in_time(model, tspan, time_change, specie, parameters_to_change, fold_change, param_values=None):
    if param_values is not None:
        # accept vector of parameter values as an argument
        if len(param_values) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        if not isinstance(param_values, np.ndarray):
            param_values = np.array(param_values)
    else:
        # create parameter vector from the values in the model
        param_values = np.array([p.value for p in model.parameters])

    new_pars = dict((p.name, param_values[i]) for i, p in enumerate(model.parameters))

    for idx, par in enumerate(parameters_to_change):
        plt.figure()
        solver = pysb.integrate.Solver(model, tspan, integrator='vode')
        solver.run(new_pars)
        nummol = np.copy(solver.y[time_change:time_change+1].T.reshape(6))
        plt.plot(tspan, solver.y[:, specie], 'o-', label='before')
        for i in np.linspace(0.1, fold_change, 6):
            params = new_pars
            params[par] *= i
            print params
            solver.run(params, y0=nummol)
            plt.plot(tspan[time_change:], solver.y[:-time_change, specie], 'x-', label=str(i))
            plt.legend(loc=0)
            plt.tight_layout()
            plt.title(par + 'time' + str(time_change))
    plt.show()


tspan = np.linspace(0, 200, 200)

change_parameter_in_time(model, tspan, 51, 5, ['kp4'], 2)

