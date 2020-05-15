import pandas as pd
import re
import numpy as np
from collections import OrderedDict
from pysb.bng import generate_equations
from pysb.simulator import SimulationResult
from itertools import compress
from scipy.optimize import curve_fit
from networkx.drawing.nx_pydot import from_pydot
from pydot import graph_from_dot_data
from anytree.importer import DictImporter
from anytree.exporter import DotExporter

try:
    import h5py
except ImportError:
    h5py = None


def species_reaction_rates(model, species):
    """

    Parameters
    ----------
    model: PySB model
    species: int or str or vector-like
        Species whose reaction rates are going to be returned

    Returns
    -------
    Dict
        The reaction rates of a species sp.
    """
    monomials = {}
    counter = 0
    if isinstance(species, str):
        species_ready = model.observables.get(species).species
    elif isinstance(species, int):
        species_ready = [species]
    else:
        species_ready = species

    for sp in species_ready:
        for term in model.reactions_bidirectional:
            total_rate = 0
            for mon_type, mon_sign in zip(['products', 'reactants'], [1, -1]):
                if sp in term[mon_type]:
                    count = term[mon_type].count(sp)
                    total_rate = total_rate + (mon_sign * count * term['rate'])
            if total_rate == 0:
                continue
            monomials[counter] = (total_rate)
            counter += 1
    counter2 = 0
    mons_ready = {}
    for sp in monomials.values():
        if -1 * sp in monomials.values():
            continue
        else:
            mons_ready[counter2] = sp
            counter2 += 1

    return mons_ready


def uniquifier(numList, biggest):
    # if biggest < 10:
    #     longest = 1
    # else:
    #     longest = int(math.floor(math.log10(biggest))+1)
    longest = len(str(biggest))
    s = map(str, numList)
    q = ''.join([n.zfill(longest) for n in s])
    return int(q)


def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


def get_simulations(simulations):
    """
    Obtains trajectories, parameters, tspan from a SimulationResult object
    Parameters
    ----------
    simulations: pysb.SimulationResult, str
        Simulation result instance or h5py file with the simulation data

    Returns
    -------
    tuple
        A tuple where the first element corresponds to a np.array with the simulated
        trajectories, the second element is a np.array with the param_values used
        in the simulations, the third element is an int which  corresponds to the
        number of simulations, and the last element is the tspan used in the
        simulations.
    """
    if isinstance(simulations, str):
        if h5py is None:
            raise Exception('please install the h5py package for this feature')
        if h5py.is_hdf5(simulations):
            sims = SimulationResult.load(simulations)
        else:
            raise TypeError('File format not supported')

    elif isinstance(simulations, SimulationResult):
        sims = simulations

    else:
        raise TypeError('format not supported')
    return _get_data_from_sims(sims)


def _get_data_from_sims(simulations):
    """
    Obtain species trajectories, parameters, nsims and tspan from
    a pysb SimulationResult object

    Parameters
    ----------
    simulations : SimulationResult
        pysb simulations

    Returns
    -------
    tuple
        Tuple with the simulation trajectories, parameters, nsims, and tspan
    """
    if all_equal(simulations.tout):
        tspan = simulations.tout[0]
    else:
        raise Exception('Analysis is not supported for simulations with different time spans')
    parameters = simulations.param_values
    nsims = len(parameters)

    # SimulationResult returns an ndarray when it only runs 1 simulation
    # and the squeeze attribute is set to True
    if nsims == 1 and simulations.squeeze is True:
        trajectories = np.array([simulations.species])
    else:
        trajectories = np.array(simulations.species)
    return trajectories, parameters, nsims, tspan


def parse_name(spec):
    """
    Function that writes short names of the species to name the nodes.
    It counts how many times a monomer_pattern is present in the complex pattern an its states
    then it takes only the monomer name and its state to write a shorter name to name the nodes.

    Parameters
    ----------
    spec : pysb.ComplexPattern
        Name of species to parse

    Returns
    -------
    Parsed name of species
    """
    m = spec.monomer_patterns
    lis_m = []
    name_counts = OrderedDict()
    parsed_name = ''
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"['\"](.*?)['\"]", str(m[i]))  # Matches strings between quotes
        tmp_2 = [s.lower() for s in tmp_2]
        # tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))
        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            tmp_2.insert(0, tmp_1[0])
            tmp_2.reverse()
            lis_m.append(''.join(tmp_2))
    for name in lis_m:
        name_counts[name] = lis_m.count(name)

    for sp, counts in name_counts.items():
        if counts == 1:
            parsed_name += sp + ':'
        else:
            parsed_name += str(counts) + sp + ':'
    return parsed_name[:len(parsed_name) - 1]


def rate_2_interactions(model, rate):
    """
    Obtains the interacting protein from a rection rate
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
        sp_monomers = {sp: model.species[sp].monomer_patterns for sp in species_idxs}
        sorted_intn = sorted(sp_monomers.items(), key=lambda value: len(value[1]))
        interaction = ", ".join(parse_name(model.species[mons[0]]) for mons in sorted_intn[:2])
    return interaction


def pre_equilibration(solver, time_search=None, param_values=None, tolerance=1e-6):
    """

    Parameters
    ----------
    solver : pysb solver
        a pysb solver object
    time_search : np.array, optional
        Time span array used to find the equilibrium. If not provided, function will use tspan from the solver
    param_values :  dict or np.array, optional
        Model parameters used to find the equilibrium, it can be an array with all model parameters
        (this array must have the same order as model.parameters) or it can be a dictionary where the
        keys are the parameter names thatpara want to be changed and the values are the new parameter
        values. If not provided, function will use param_values from the solver
    tolerance : float
        Tolerance to define when the equilibrium has been reached

    Returns
    -------

    """
    # Solve system for the time span provided
    sims = solver.run(tspan=time_search, param_values=param_values)
    if not time_search:
        time_search = solver.tspan

    if sims.nsims == 1:
        simulations = [sims.species]
    else:
        simulations = sims.species

    dt = time_search[1] - time_search[0]

    all_times_eq = [0] * sims.nsims
    all_conc_eq = [0] * sims.nsims
    for n, y in enumerate(simulations):
        time_to_equilibration = [0, 0]
        for idx in range(y.shape[1]):
            sp_eq = False
            derivative = np.diff(y[:, idx]) / dt
            derivative_range = ((derivative < tolerance) & (derivative > -tolerance))
            # Indexes of values less than tolerance and greater than -tolerance
            derivative_range_idxs = list(compress(range(len(derivative_range)), derivative_range))
            for i in derivative_range_idxs:
                # Check if derivative is close to zero in the time points ahead
                if i + 3 > len(time_search):
                    raise Exception('Equilibrium can not be reached within the time_search input')
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

        conc_eq = y[time_to_equilibration[1]]
        all_times_eq[n] = time_to_equilibration
        all_conc_eq[n] = conc_eq

    return all_times_eq, all_conc_eq


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
        generate_equations(model)

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


def get_labels_entropy(labels, base=None):
    """
    Function to calculate the entropy from labels
    Parameters
    ----------
    labels: vector-like
        Vector of labels
    base: int
        Base used to calculated the entropy. If none then 2 is used.

    Returns
    -------

    """
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = 2 if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def get_probs_entropy(probs, base=None):
    """
    Function to calculate the entropy from states probabilities
    Parameters
    ----------
    probs: vector-like
        Vector of probabilities
    base: int
        Base used to calculated the entropy. If none then 2 is used.

    Returns
    -------

    """
    norm_counts = probs
    norm_counts = norm_counts[norm_counts > 0]
    base = 2 if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def path_differences(model, paths_labels, type_analysis='production'):
    """

    Parameters
    ----------
    model: PySB model
        Model used to do dominant path analysis
    paths_labels: dict
        Dictionary of pathways generated by dominant path analysis
    type_analysis: str
        Type of analysis used in the dominant path analysis.
        It can either be `production` or `consumption`

    Returns
    -------
    A pandas dataframe where the column names and row indices are the labels of the
    pathways and the cells contain the edges that are present in the
    row pathway index but not in the column pathway.
    """
    generate_equations(model)
    importer = DictImporter()
    path_edges = {}

    def find_numbers(dom_r_str):
        n = map(int, re.findall('\d+', dom_r_str))
        return n

    def nodenamefunc(node):
        node_idx = list(find_numbers(node.name))[0]
        node_sp = model.species[node_idx]
        node_name = parse_name(node_sp)
        return node_name

    def edgeattrfunc(node, child):
        return 'dir="back"'

    for keys, values in paths_labels.items():
        root = importer.import_(values)
        dot = DotExporter(root, graph='strict digraph', options=["rankdir=RL;"], nodenamefunc=nodenamefunc,
                          edgeattrfunc=edgeattrfunc)
        data = ''
        for line in dot:
            data += line
        pydot_graph = graph_from_dot_data(data)
        graph = from_pydot(pydot_graph[0])
        if type_analysis == 'production':
            graph = graph.reverse()
        edges = set(graph.edges())
        path_edges[keys] = edges

    path_diff = pd.DataFrame(index=paths_labels.keys(), columns=paths_labels.keys())
    for row in path_diff.columns:
        for col in path_diff.columns:
            path_diff.loc[row, col] = path_edges[row].difference(path_edges[col])
    return path_diff
