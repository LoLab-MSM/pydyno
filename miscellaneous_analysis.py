import numpy as np
import matplotlib.pyplot as plt
import pysb.integrate
import helper_functions as hf


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
        sp_before = solver.y[:, specie]
        plt.plot(tspan, sp_before, 'o-', label='before')
        for i in np.linspace(0.1, fold_change, 6):
            params = new_pars
            params[par] *= i
            solver.run(params, y0=nummol)
            sp_after = solver.y[:-time_change, specie]
            plt.plot(tspan[time_change:], sp_after, 'x-', label=str(i))
            plt.legend(loc=0)
            plt.tight_layout()
            plt.title(par + ' ' + 'time' + str(time_change))
    plt.show()


def parameter_distribution(parameters_path, par_name):
    all_parameters = hf.read_all_pars(parameters_path)
    plt.figure()
    all_parameters[par_name].plot.hist()
    plt.show()

