import re
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pydyno.discretization.base import DomPath, _dominant_paths, _reencode_signatures_paths
import numpy as np
import networkx as nx
from pysb.bng import generate_equations
from pysb import Parameter
from pysb.simulator.scipyode import SerialExecutor
import pandas as pd
import sympy
import pydyno.util as hf
from pydyno.seqanalysis import SeqAnalysis
from tqdm import tqdm


class PysbDomPath(DomPath):
    """
    Obtain dominant paths from models encoded in the PySB format.

    Parameters
    ----------
    model: PySB model
        Model to analyze
    simulations: PySB SimulationResult object or str
        Simulations used to perform the analysis. If str it should be the
        filepath to a pysb simulation result in hdf5 format

    Examples
    --------
    Obtain the discretized trajectory of the extrinsic apoptosis reaction model

    >>> from pydyno.discretization import PysbDomPath
    >>> from pydyno.examples.earm.earm2_flat import model
    >>> from pysb.simulator import ScipyOdeSimulator
    >>> import numpy as np
    >>> tspan = np.linspace(0, 20000, 100)
    >>> # Simulate model
    >>> sim = ScipyOdeSimulator(model, tspan).run()
    >>> # Obtain dominant paths that consume species 37
    >>> dp = PysbDomPath(model=model, simulations=sim)
    >>> signs, paths = dp.get_path_signatures(target='s37', type_analysis='consumption', depth=5, dom_om=1)
    >>> print(signs.sequences.iloc[:, :5]) \
        #doctest: +NORMALIZE_WHITESPACE
		          202.020203  404.040405 606.060608 808.080811 1010.101013
    seq_idx	count
          0	    1	       8	       8	      8	         8	         8

    For further information on retrieving sequences from the ``SeqAnalysis``
    object returned by :func:`get_path_signatures`, see the examples under the
    :class:`pydyno.seqanalysis.SeqAnalysis` class.
    """

    def __init__(self, model, simulations):
        super().__init__(model)
        self._trajectories, self._parameters, self._nsims, self._tspan = hf.get_simulations(simulations)
        self._par_name_idx = {j.name: i for i, j in enumerate(self.model.parameters)}
        generate_equations(self.model)

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def parameters(self):
        return self._parameters

    @property
    def nsims(self):
        return self._nsims

    @property
    def tspan(self):
        return self._tspan

    @property
    def par_name_idx(self):
        return self._par_name_idx

    def create_bipartite_graph(self):
        """
        Create bipartite graph with species and reaction nodes of the pysb model

        Returns
        -------
        nx.DiGraph
            a NetworkX directed graph
        """
        graph = nx.DiGraph(name=self.model.name)
        for i, cp in enumerate(self.model.species):
            species_node = 's%d' % i
            slabel = re.sub(r'% ', r'%\\l', str(cp))
            slabel += '\\l'
            graph.add_node(species_node,
                           label=slabel)
        for i, reaction in enumerate(self.model.reactions_bidirectional):
            reaction_node = 'r%d' % i
            graph.add_node(reaction_node,
                           label=reaction_node)
            reactants = set(reaction['reactants'])
            products = set(reaction['products'])
            modifiers = reactants & products
            reactants = reactants - modifiers
            products = products - modifiers
            attr_reversible = {'dir': 'both', 'arrowtail': 'empty'} if reaction['reversible'] else {}
            for s in reactants:
                self.r_link(graph, s, i, **attr_reversible)
            for s in products:
                self.r_link(graph, s, i, _flip=True, **attr_reversible)
            for s in modifiers:
                self.r_link(graph, s, i, _flip=True, arrowhead="odiamond")
        return graph

    @staticmethod
    def r_link(graph, s, r, **attrs):
        nodes = ('s%d' % s, 'r%d' % r)
        if attrs.get('_flip'):
            del attrs['_flip']
            nodes = reversed(nodes)
        attrs.setdefault('arrowhead', 'normal')
        graph.add_edge(*nodes, **attrs)

    def get_path_signatures(self, target, type_analysis, depth, dom_om,
                            num_processors=1, sample_simulations=None):
        """
        Obtain the dominant paths

        Parameters
        ----------
        target: str
            Species target. It has to be in a format `s1` where the number
            represents the species index
        type_analysis: str
            Type of analysis to perform. It can be `production` or `consumption`
        depth: int
            Depth of the traceback starting from the target species
        dom_om: float
            Order of magnitude to consider dominancy
        num_processors: int
            Number of cores to use in the function
        sample_simulations: int
            Number of simulations to use for the analysis

        Returns
        -------
        pydyno.SeqAnalysis
            Sequences of the discretized signatures
        """
        if sample_simulations:
            if isinstance(sample_simulations, int):
                nsims = range(sample_simulations)
            elif isinstance(sample_simulations, list):
                nsims = sample_simulations
            else:
                raise TypeError('Sample method not supported')
        else:
            nsims = range(self.nsims)

        network = self.create_bipartite_graph()

        dom_path_partial = partial(dominant_paths_pysb, param_idx_dict=self.par_name_idx,
                                   reactions_bidirectional=self.model.reactions_bidirectional, tspan=self.tspan,
                                   network=network, target=target, type_analysis=type_analysis,
                                   depth=depth, dom_om=dom_om)

        pbar = tqdm(total=len(self.parameters))

        def update(*a):
            pbar.update()

        with SerialExecutor() if num_processors == 1 else \
                ProcessPoolExecutor(max_workers=num_processors) as executor:
            results = []
            for args in zip(self.trajectories, self.parameters):
                f = executor.submit(dom_path_partial, *args)
                f.add_done_callback(update)
                results.append(f)

            try:
                signatures_labels = [r.result() for r in results]
            finally:
                for r in results:
                    r.cancel()

        signatures = [0] * len(nsims)
        labels = [0] * len(nsims)
        for idx, sl in enumerate(signatures_labels):
            signatures[idx] = sl[0]
            labels[idx] = sl[1]
        signatures_df, new_paths = _reencode_signatures_paths(signatures, labels, self.tspan)
        # signatures_labels = {'signatures': signatures, 'labels': all_labels}
        return SeqAnalysis(signatures_df, target), new_paths


def _calculate_pysb_expression(expr, trajectories, parameters, param_idx_dict):
    """Obtains simulated values of a pysb expression"""
    expanded_expr = expr.expand_expr(expand_observables=True)
    expr_variables = [atom for atom in expanded_expr.atoms(sympy.Symbol)]
    args = [0] * len(expr_variables)
    for idx2, va in enumerate(expr_variables):
        # Getting species index
        if str(va).startswith('__'):
            sp_idx = int(''.join(filter(str.isdigit, str(va))))
            args[idx2] = trajectories[:, :, sp_idx]
        else:
            par_values = parameters[:, param_idx_dict[va.name]]
            args[idx2] = par_values.reshape((len(par_values), 1))

    func = sympy.lambdify(expr_variables, expanded_expr, modules='numpy')
    expr_value = func(*args)

    return expr_value


def calculate_reaction_rate(rate_react, trajectories, parameters, param_idx_dict,
                            changed_parameters=None, time_change=None):
    """
    Get reaction rate values from simulated trajectories

    Parameters
    ----------
    rate_react
    trajectories
    parameters
    param_idx_dict

    Returns
    -------

    """
    variables = [atom for atom in rate_react.atoms(sympy.Symbol)]
    args = [0] * len(variables)  # arguments to put in the lambdify function
    for idx2, va in enumerate(variables):
        # Getting species index
        if str(va).startswith('__'):
            sp_idx = int(''.join(filter(str.isdigit, str(va))))
            args[idx2] = trajectories[:, :time_change, sp_idx]
        elif isinstance(va, Parameter):
            par_values = parameters[:, param_idx_dict[va.name]]
            args[idx2] = par_values.reshape((len(par_values), 1))
        else:
            # Calculate expressions
            args[idx2] = _calculate_pysb_expression(va, trajectories[:, :time_change, :], parameters, param_idx_dict)

    func = sympy.lambdify(variables, rate_react, modules=dict(sqrt=np.lib.scimath.sqrt))
    react_rate = func(*args)

    if changed_parameters is not None and time_change is not None:
        args = [0] * len(variables)  # arguments to put in the lambdify function
        for idx2, va in enumerate(variables):
            # Getting species index
            if str(va).startswith('__'):
                sp_idx = int(''.join(filter(str.isdigit, str(va))))
                args[idx2] = trajectories[:, time_change:, sp_idx]
            elif isinstance(va, Parameter):
                par_values = changed_parameters[:, param_idx_dict[va.name]]
                args[idx2] = par_values.reshape((len(par_values), 1))
            else:
                # Calculate expressions
                args[idx2] = _calculate_pysb_expression(va, trajectories[:, time_change:, :], changed_parameters,
                                                        param_idx_dict)

        func = sympy.lambdify(variables, rate_react, modules=dict(sqrt=np.lib.scimath.sqrt))
        react_rate2 = func(*args)
        react_rate = np.concatenate((react_rate, react_rate2), axis=1)

    return react_rate


def pysb_reaction_flux_df(reactions_bidirectional, trajectories, parameters, param_idx_dict, tspan):
    """
    Create a pandas DataFrame with the reaction rates values at each time point

    Parameters
    ----------
    reactions_bidirectional:
        PySB bidirectional reactions
    trajectories: np.ndarray
        Simulated trajectories
    parameters: np.ndarray
        Parameters used to obtain the simulations
    tspan:  np.ndarray
        Time span used in the simulations

    Returns
    -------
    pandas.DataFrame
        Dataframe with the reaction rate values of the simulations
    """

    rxns_names = ['r{0}'.format(rxn) for rxn in range(len(reactions_bidirectional))]
    rxns_df = pd.DataFrame(columns=tspan, index=rxns_names)

    for idx, reac in enumerate(reactions_bidirectional):
        react_rate = calculate_reaction_rate(reac['rate'], trajectories, parameters, param_idx_dict)
        rxns_df.loc['r{0}'.format(idx)] = react_rate
    rxns_df['Total'] = rxns_df.sum(axis=1)
    return rxns_df


def dominant_paths_pysb(trajectories, parameters, param_idx_dict, reactions_bidirectional, tspan,
                        type_analysis, network, target, depth, dom_om):
    rxns_df = pysb_reaction_flux_df(reactions_bidirectional, trajectories[np.newaxis, :, :], parameters[np.newaxis, :],
                                    param_idx_dict, tspan)
    dom_paths = _dominant_paths(rxns_df, network, tspan, target, type_analysis, depth, dom_om)
    return dom_paths
