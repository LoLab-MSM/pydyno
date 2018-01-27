import os
import pandas as pd
import re
import numpy as np
from collections import OrderedDict
from pysb.bng import generate_equations
import pysb
from pysb.simulator import ScipyOdeSimulator
from itertools import compress
from scipy.optimize import curve_fit
from sympy import Add
def listdir_fullpath(d):
    """

    Parameters
    ----------
    d : str
        path to directory

    Returns
    -------
    a list of paths of the files in directory
    """
    return [os.path.join(d, f) for f in os.listdir(d)]


def list_pars_infile(f, new_path=None):
    """

    Parameters
    ----------
    f : str
        File that contain paths to parameter values
    new_path : str
        New path to assign to the parameter sets

    Returns
    -------
    list of parameter paths
    """
    par_sets = pd.read_csv(f, names=['parameters'])['parameters'].tolist()
    if new_path:
        par_sets = [w.replace(w.rsplit('/', 1)[0], new_path) for w in par_sets]
    return par_sets


def read_pars(par_path):
    """

    Parameters
    ----------
    par_path : str
        Path to parameter file

    Returns
    -------
    a list of parameter values
    """
    if par_path.endswith('.txt') or par_path.endswith('.csv'):
        data = np.genfromtxt(par_path, delimiter=',', dtype=None)
        if len(data.dtype) == 0:
            pars = data
        elif len(data.dtype) == 2:
            pars = data['f1']
        else:
            raise Exception('structure of the file is not supported')
    elif par_path.endswith('.npy'):
        pars = np.load(par_path)
    else:
        raise ValueError('format not supported')

    return pars


def read_all_pars(pars_path, new_path=None):
    """

    Parameters
    ----------
    pars_path : str
        Parameter file or directory path
    new_path : np.ndarray
        Array with all the parameters

    Returns
    -------

    """
    if isinstance(pars_path, list):
        par_sets = pars_path
    elif os.path.isfile(pars_path):
        par_sets = list_pars_infile(pars_path, new_path)
    elif os.path.isdir(pars_path):
        par_sets = listdir_fullpath(pars_path)
    else:
        raise Exception("Not valid parameter file or path")
    pars_0 = read_pars(par_sets[0])
    all_pars = np.zeros((len(par_sets), len(pars_0)))
    all_pars[0] = pars_0
    for idx in range(1, len(par_sets)):
        all_pars[idx] = read_pars(par_sets[idx])
    return all_pars


def parse_name(spec):
    """

    Parameters
    ----------
    spec : pysb.ComplexPattern
        Species's whose name is going to be parsed

    Returns
    -------
    A shorter version of the species name
    """
    m = spec.monomer_patterns
    lis_m = []
    name_counts = OrderedDict()
    parsed_name = ''
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))

        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0], tmp_2[0]]))

    for name in lis_m:
        name_counts[name] = lis_m.count(name)

    for sp, counts in name_counts.items():
        if counts == 1:
            parsed_name += sp + '_'
        else:
            parsed_name += str(counts) + sp + '_'
    return parsed_name[:len(parsed_name) - 1]


def rate_2_interactions(model, rate):
    """

    Parameters
    ----------
    model : PySB model
    rate : str
    Returns
    -------

    """

    generate_equations(model)
    species_idxs = re.findall('(?<=__s)\d+', rate)
    species_idxs = [int(i) for i in species_idxs]
    if len(species_idxs) == 1:
        interaction = parse_name(model.species[species_idxs[0]])
    else:
        sp_monomers ={sp: model.species[sp].monomer_patterns for sp in species_idxs }
        sorted_intn = sorted(sp_monomers.items(), key=lambda value: len(value[1]))
        interaction = " ".join(parse_name(model.species[mons[0]]) for mons in sorted_intn[:2])
    return interaction


def pre_equilibration(model, time_search, parameters, tolerance=1e-6):
    """

    Parameters
    ----------
    model : pysb.Model
        pysb model
    time_search : np.array
        Time span array used to find the equilibrium
    parameters :  dict or np.array
        Model parameters used to find the equilibrium, it can be an array with all model parameters
        (this array must have the same order as model.parameters) or it can be a dictionary where the
        keys are the parameter names that want to be changed and the values are the new parameter
        values.
    tolerance : float
        Tolerance to define when the equilibrium has been reached

    Returns
    -------

    """
    # Solve system for the time span provided
    solver = ScipyOdeSimulator(model, tspan=time_search, param_values=parameters).run()
    y = solver.species.T
    dt = time_search[1] - time_search[0]

    time_to_equilibration = [0, 0]
    for idx, sp in enumerate(y):
        sp_eq = False
        derivative = np.diff(sp) / dt
        derivative_range = ((derivative < tolerance) & (derivative > -tolerance))
        # Indexes of values less than tolerance and greater than -tolerance
        derivative_range_idxs = list(compress(range(len(derivative_range)), derivative_range))
        for i in derivative_range_idxs:
            # Check if derivative is close to zero in the time points ahead
            if (derivative[i + 3] < tolerance) | (derivative[i + 3] > -tolerance):
                sp_eq = True
                if time_search[i] > time_to_equilibration[0]:
                    time_to_equilibration[0] = time_search[i]
                    time_to_equilibration[1] = i
            if not sp_eq:
                raise Exception('Equilibrium can not be reached within the time_search input')
            if sp_eq:
                break
        else:
            raise Exception('Species s{0} has not reached equilibrium'.format(idx))

    conc_eq = y[:, time_to_equilibration[1]]
    return time_to_equilibration, conc_eq


def find_nonimportant_nodes(model):
    """
    This function looks a the bidirectional reactions and finds the nodes that only have one incoming and outgoing
    reaction (edge)

    Parameters
    ----------
    model : pysb.Model
        PySB model to use

    Returns
    -------
    a list of non-important nodes
    """
    if not model.odes:
        pysb.bng.generate_equations(model)

    # gets the reactant and product species in the reactions
    rcts_sp = sum([i['reactants'] for i in model.reactions_bidirectional], ())
    pdts_sp = sum([i['products'] for i in model.reactions_bidirectional], ())
    # find the reactants and products that are only used once
    non_imp_rcts = set([x for x in range(len(model.species)) if rcts_sp.count(x) < 2])
    non_imp_pdts = set([x for x in range(len(model.species)) if pdts_sp.count(x) < 2])
    non_imp_nodes = set.intersection(non_imp_pdts, non_imp_rcts)
    passengers = non_imp_nodes
    return passengers


def column(matrix, i):
    """

    Parameters
    ----------
    matrix : np.ndarray
        Array to get the column from
    i : int
        Column index to get from array

    Returns
    -------

    """
    return np.array([row[i] for row in matrix])


def sig_apop(t, f, td, ts):
    """

    Parameters
    ----------
    t : list-like
        Time variable in the function
    f : float
        is the fraction cleaved at the end of the reaction
    td : float
        is the delay period between TRAIL addition and half-maximal substrate cleavage
    ts : float
        is the switching time between initial and complete effector substrate  cleavage

    Returns
    -------

    """
    return f - f / (1 + np.exp((t - td) / (4 * ts)))


def curve_fit_ftn(fn, xdata, ydata, **kwargs):
    """
    Fit simulation data to specific function

    Parameters
    ----------
    fn: callable
        function that would be used for fitting the data
    xdata: list-like,
        x-axis data points (usually time span of the simulation)
    ydata: list-like,
        y-axis data points (usually concentration of species in time)
    kwargs: dict,
        Key arguments to use in curve-fit

    Returns
    -------
    Parameter values of the functions used to fit the data

    """

    # TODO change to use for loop
    def curve_fit2(data):
        c = curve_fit(f=fn, xdata=xdata, ydata=data, **kwargs)
        return c[0]

    fit_all = np.apply_along_axis(curve_fit2, axis=1, arr=ydata)
    return fit_all