import numpy as np
import matplotlib.pyplot as plt
import pydyno.util as hf
import pandas as pd
from pydyno.discretize import Discretize
from pysb.integrate import ScipyOdeSimulator


# CHANGES IN PARAMETER VALUE AT CERTAIN TIME POINT
def change_parameter_in_time(model, tspan, time_change, previous_parameters, new_parameters):
    """

    Parameters
    ----------
    model: pysb.Model
        PySB model to use
    tspan: vector-like
        Time values over which to simulate.
    time_change: int
        Index in tspan at which the parameter is going to be changed
    previous_parameters: np.ndarray
        Parameter used before the time_change
    new_parameters: np.ndarray
        Parameters used after time_change

    Returns
    -------

    """
    before_change_simulation = ScipyOdeSimulator(model=model, tspan=tspan[:time_change]).\
        run(param_values=previous_parameters)
    species_before_change = np.array(before_change_simulation.species)
    concentrations_time_change = species_before_change[:, time_change-1, :]

    after_change_simulation = ScipyOdeSimulator(model=model, tspan=tspan[time_change:]).\
        run(initials=concentrations_time_change, param_values=new_parameters)

    return after_change_simulation


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

