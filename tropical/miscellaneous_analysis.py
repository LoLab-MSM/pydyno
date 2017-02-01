import numpy as np
import matplotlib.pyplot as plt
import pysb.integrate
import helper_functions as hf
from pysb.integrate import ScipyOdeSimulator


# CHANGES IN PARAMETER VALUE AT CERTAIN TIME POINT
def change_parameter_in_time(model, tspan, time_change, specie, parameters_to_change, fold_change, param_values=None):
    """

    :param model:
    :param tspan:
    :param time_change:
    :param specie:
    :param parameters_to_change:
    :param fold_change:
    :param param_values:
    :return:
    """
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
        solver = ScipyOdeSimulator(model=model, tspan=tspan)
        y = solver.run(param_values=new_pars).species
        nummol = np.copy(y[time_change:time_change+1])#.T.reshape(len(model.species)))
        sp_before = y[:, specie]
        plt.plot(tspan, sp_before, 'o-', label='before')
        for i in np.linspace(0.1, fold_change, len(model.species)):
            params = new_pars
            params[par] *= i
            y1 = solver.run(initials=nummol, param_values=params).species
            sp_after = y1[:-time_change, specie]
            plt.plot(tspan[time_change:], sp_after, 'x-', label=str(i))
            plt.legend(loc=0)
            plt.tight_layout()
            plt.title(par + ' ' + 'time' + str(time_change))
    plt.show()


def parameter_distribution(parameters_paths, par_name, new_path):
    """

    :param parameters_paths: paths to clusters of parameters. It can be a folder or a file that contains the paths
    :param par_name: Specific PySB parameter name to look at
    :param new_path: Optional, path to where the parameters are in the file names
    :return: a matplotlib histogram of the parameters
    """

    plt.figure(1)
    for i, par_path in enumerate(parameters_paths):
        all_parameters = hf.read_all_pars(par_path, new_path)
        weights = np.ones_like(all_parameters[par_name]*100)/len(all_parameters[par_name])
        all_parameters[par_name].plot.hist(title=par_name, alpha=0.5, label='Type{0}'.format(i), weights=weights)

    plt.legend(loc=0)
    plt.savefig('/home/oscar/Documents/tropical_earm/type1_pars_distribution/{0}.{1}'.format(par_name, 'png'), bbox_inches='tight', dpi=400
                , format='png')
    plt.clf()
    # plt.show()


def sig_apop(t, f, td, ts):
    """
    Gets amount of cleaved substrate at time t
    :param t: time
    :param f: is the fraction cleaved at the end of the reaction
    :param td: is the delay period between TRAIL addition and half-maximal substrate cleavage
    :param ts: is the switching time between initial and complete effector substrate  cleavage
    :return: the amount of substrate cleaved at time t.
    """
    return f - f / (1 + np.exp((t - td) / (4 * ts)))



