import numpy as np
import pydyno.util as hf
import pandas as pd
from pydyno.discretize import Discretize
from pysb.simulator import ScipyOdeSimulator, SimulationResult


# CHANGES IN PARAMETER VALUE AT CERTAIN TIME POINT
def simulate_changing_parameter_in_time(model, tspan, time_change, previous_parameters, new_parameters,
                                        drop_na_sim=False, num_processors=1):
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
    drop_na_sim: bool
        Whether to drop simulations that have nan in the trajectory of
        any of the species
    num_processors: int
        Number of processors to use in the simulations

    Returns
    -------
    SimulationResult
        Simulation obtained with `previous_parameters` before the time change and using
        `new_parameters` after time change

    .. warning::
        This SimulationResult object is used only to store the species trajectories, the
        previous and new parameter sets and the time change index. This SimulationResult
        object should only be used for the visualize_simulations module.
    """

    simulation_before_change = ScipyOdeSimulator(model=model, tspan=tspan[:time_change]). \
        run(param_values=previous_parameters, num_processors=num_processors)

    if simulation_before_change.nsims == 1 and simulation_before_change.squeeze is True:
        species_before_change = np.array([simulation_before_change.species])
    else:
        species_before_change = np.array(simulation_before_change.species)

    concentrations_time_change = species_before_change[:, time_change - 1, :]

    simulation_after_change = ScipyOdeSimulator(model=model, tspan=tspan[time_change:]). \
        run(initials=concentrations_time_change, param_values=new_parameters,
            num_processors=num_processors)

    if simulation_after_change.nsims == 1 and simulation_after_change.squeeze is True:
        species_after_change = np.array([simulation_after_change.species])
    else:
        species_after_change = np.array(simulation_after_change.species)

    # This is a hack, because different parameter sets are used to obtain the
    # trajectories. I did this because the visualization module requires a
    # SimulationResult object.

    full_trajectories = np.concatenate((species_before_change, species_after_change),
                                       axis=1)
    full_touts = np.concatenate((simulation_before_change.tout,
                                 simulation_after_change.tout), axis=1)

    if drop_na_sim:
        sim_with_nan = np.isnan(full_trajectories).any(axis=(1, 2))
        full_trajectories_nan_dropped = full_trajectories[~sim_with_nan]
        full_touts_nan_dropped = full_touts[~sim_with_nan]
        pars_nan_dropped = simulation_before_change.param_values[~sim_with_nan]
        new_parameters = simulation_after_change.param_values[~sim_with_nan]
        initials_nan_dropped = concentrations_time_change[~sim_with_nan]
        full_simulation = SimulationResult(simulator=None, tout=full_touts_nan_dropped,
                                           trajectories=full_trajectories_nan_dropped,
                                           param_values=pars_nan_dropped,
                                           initials=initials_nan_dropped,
                                           model=model)
        full_simulation.changed_parameters = new_parameters
        full_simulation.time_change = time_change
        return full_simulation, np.argwhere(sim_with_nan)
    else:
        full_simulation = SimulationResult(simulator=None, tout=full_touts,
                                           trajectories=full_trajectories,
                                           param_values=simulation_before_change.param_values,
                                           initials=concentrations_time_change,
                                           model=model)
        full_simulation.changed_parameters = new_parameters
        full_simulation.time_change = time_change
        return full_simulation


def simulations_from_cluster(simulation, cluster_indices):
    """
    Create new SimulationResult object only using the cluster_indices argument

    Parameters
    ----------
    simulation: SimulationResult
        PySB simulation
    cluster_indices: list-like
        List of indices of the simulations used to create the SimulationResult object

    Returns
    -------
    SimulationResult
        Simulations from the cluster
    """
    trajectories = np.array(simulation.species)
    cluster_trajectories = trajectories[cluster_indices, :, :]
    cluster_parameters = simulation.param_values[cluster_indices, :]
    cluster_tout = simulation.tout[cluster_indices, :]
    cluster_simulation = SimulationResult(simulator=None, tout=cluster_tout,
                                          trajectories=cluster_trajectories,
                                          param_values=cluster_parameters,
                                          model=simulation._model)
    return cluster_simulation


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
        y[hf.parse_name(model.species[sp]) + '_truncated'] = signatures[sp][0]
    y.to_csv(file_path + 'tr_sig.txt')

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
        initials_df.to_csv(file_path + 'initials.txt', index=False)
    return
