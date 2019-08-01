import numpy as np
import matplotlib.pyplot as plt
import pydyno.util as hf
import pandas as pd
from pydyno.discretize import Discretize
from pysb.integrate import ScipyOdeSimulator


# CHANGES IN PARAMETER VALUE AT CERTAIN TIME POINT
def change_parameter_in_time(model, tspan, time_change, specie, parameters_to_change, fold_change, param_values=None):
    """

    Parameters
    ----------
    model : pysb.Model
        PySB model to use
    tspan : vector-like
        Time values over which to simulate.
    time_change : int
        Index in tspan at which the paramater is going to be changed
    specie : int
        Index of species that is going to be plotted
    parameters_to_change : str
        Name of the parameter whose value is going to be changed
    fold_change : float
        Fold change of the parameter values
    param_values : vector-like, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If not specified, parameter values will be taken directly from
        model.parameters.

    Returns
    -------

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
    solver = ScipyOdeSimulator(model=model, tspan=tspan)

    for idx, par in enumerate(parameters_to_change):
        plt.figure()
        y = solver.run(param_values=new_pars).species
        nummol = np.copy(y[time_change:time_change+1])#.T.reshape(len(model.species)))
        sp_before = y[:, specie]
        plt.plot(tspan, sp_before, 'o-', label='before')
        for i in np.linspace(0.1, fold_change, len(model.species)):
            params = {**new_pars}
            params[par] *= i
            y1 = solver.run(initials=nummol, param_values=params).species
            sp_after = y1[:-time_change, specie]
            plt.plot(tspan[time_change:], sp_after, 'x-', label=str(i))
            plt.legend(loc=0)
            plt.tight_layout()
            plt.title(par + ' ' + 'time' + str(time_change))
    plt.show()


def trajectories_signature_2_txt(model, tspan, sp_to_analyze=None, parameters=None, file_path=''):
    """

    Parameters
    ----------
    model : pysb.Model
        PySB model to use
    tspan : vector-like
        Time values over which to simulate.
    sp_to_analyze: vector-like
        Species whose dynamic signature is going to be obtained
    parameters : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    file_path : str
        Path for saving the file

    Returns
    -------

    """
    # TODO: make sure this functions works
    sim_result = ScipyOdeSimulator(model, tspan=tspan, param_values=parameters).run()
    y = sim_result.dataframe
    observables = [obs.name for obs in model.observables]
    y.drop(observables, axis=1, inplace=True)
    sp_short_names = [hf.parse_name(sp) for sp in model.species]
    y.columns = sp_short_names
    disc = Discretize(model, sim_result, diff_par=1)
    signatures = disc.get_signatures()
    # FIXME: This is not working. The signature data structure now uses pandas
    for sp in sp_to_analyze:
        y[hf.parse_name(model.species[sp])+'_truncated'] = signatures[sp][0]
    y.to_csv(file_path+'tr_sig.txt')

    if parameters is not None:
        initials = np.zeros([1, len(model.species)])
        initials_df = pd.DataFrame(data=initials, columns=sp_short_names)
        initial_pars = [ic[1] for ic in model.initial_conditions]
        for i, par in enumerate(model.parameters):
            if par in initial_pars:
                idx = initial_pars.index(par)
                ic_name = hf.parse_name(model.initial_conditions[idx][0])
                ic_value = parameters[i]
                initials_df[ic_name] = ic_value
        initials_df.to_csv(file_path+'initials.txt', index=False)
    return

