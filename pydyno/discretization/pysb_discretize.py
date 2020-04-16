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


class PysbDomPath(DomPath):
    """
    Class to discretize simulated trajectories of a model species

    Obtains dominant paths from models encoded in the PySB format.

    Parameters
    ----------
    model: PySB model
        Model to analyze
    simulations: PySB SimulationResult object or str
        simulations used to perform the analysis. If str it should be the
        path to a simulation result in hdf5 format

    Examples
    --------
    Obtain the discretized trajectory of an apoptosis model

    >>> from pydyno.discretization.pysb_discretize import PysbDomPath
    >>> from pydyno.examples.earm.earm2_flat import model
    >>> from pysb.simulator import ScipyOdeSimulator
    >>> import numpy as np
    >>> tspan = np.linspace(0, 20000, 100)
    >>> sim = ScipyOdeSimulator(model, tspan).run()
    >>> dp = PysbDomPath(model=model, simulations=sim)
    >>> signs, paths = dp.get_path_signatures(target='s37', type_analysis='consumption', depth=5, dom_om=1)

    """

    def __init__(self, model, simulations):
        super().__init__(model)
        self._trajectories, self._parameters, self._nsims, self._tspan = hf.get_simulations(simulations)
        if self._nsims == 1:
            self._trajectories = np.array([self._trajectories])
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
        Creates bipartite graph with species and reaction nodes of the pysb model
        Returns
        -------

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

        Parameters
        ----------
        target: str
            Species target. It has to be in a format `s1` where the number
            represents the species index
        type_analysis: str
            Type of analysis to perform. It can be `production` or `consumption`
        depth: int
            Depth of the traceback starting from target
        dom_om: float
            Order of magnitude to consider dominancy
        num_processors : int
            Number of cores to use in the function
        sample_simulations : int
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

        with SerialExecutor() if num_processors == 1 else \
                ProcessPoolExecutor(max_workers=num_processors) as executor:
            dom_path_partial = partial(dominant_paths_pysb, model=self.model, tspan=self.tspan, network=network,
                                       target=target, type_analysis=type_analysis, depth=depth, dom_om=dom_om)

            results = [executor.submit(dom_path_partial, *args)
                       for args in zip(self.trajectories, self.parameters)]
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


def calculate_pysb_expression(expr, trajectories, param_dict):
    """Obtains value of a pysb expression"""
    expanded_expr = expr.expand_expr(expand_observables=True)
    expr_variables = [atom for atom in expanded_expr.atoms(sympy.Symbol)]
    args = [0] * len(expr_variables)
    for idx2, va in enumerate(expr_variables):
        # Getting species index
        if str(va).startswith('__'):
            sp_idx = int(''.join(filter(str.isdigit, str(va))))
            args[idx2] = trajectories[:, sp_idx]
        else:
            args[idx2] = param_dict[va.name]
    func = sympy.lambdify(expr_variables, expanded_expr, modules='numpy')
    expr_value = func(*args)
    return expr_value


def pysb_reaction_flux_df(trajectories, parameters, model, tspan):
    """
    Creates a data frame with the reaction rates values at each time point
    and obtains the dominant path for the passed trajectories and parameters

    Parameters
    ----------
    trajectories : np.ndarray
        Simulated trajectories
    parameters : np.ndarray
        Parameters used to obtain the simulations
    model : pysb.Model
        Model used to obtain the simulations
    tspan :  np.ndarray
        Time span used in the simulations
    Returns
    -------
    pandas.DataFrame
        Dataframe with the reaction rate values of the simulations
    """
    trajectories = trajectories
    parameters = parameters
    rxns_names = ['r{0}'.format(rxn) for rxn in range(len(model.reactions_bidirectional))]
    rxns_df = pd.DataFrame(columns=tspan, index=rxns_names)
    param_dict = dict((p.name, parameters[i]) for i, p in enumerate(model.parameters))

    for idx, reac in enumerate(model.reactions_bidirectional):
        rate_reac = reac['rate']
        # Getting species and parameters from the reaction rate
        variables = [atom for atom in rate_reac.atoms(sympy.Symbol)]
        args = [0] * len(variables)  # arguments to put in the lambdify function
        for idx2, va in enumerate(variables):
            # Getting species index
            if str(va).startswith('__'):
                sp_idx = int(''.join(filter(str.isdigit, str(va))))
                args[idx2] = trajectories[:, sp_idx]
            elif isinstance(va, Parameter):
                args[idx2] = param_dict[va.name]
            else:
                # Calculate expressions
                args[idx2] = calculate_pysb_expression(va, trajectories, param_dict)

        func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=np.lib.scimath.sqrt))
        react_rate = func(*args)
        rxns_df.loc['r{0}'.format(idx)] = react_rate
    rxns_df['Total'] = rxns_df.sum(axis=1)
    return rxns_df


def dominant_paths_pysb(trajectories, parameters, model, tspan,
                        type_analysis, network, target, depth, dom_om):
    rxns_df = pysb_reaction_flux_df(trajectories, parameters, model, tspan)
    dom_paths = _dominant_paths(rxns_df, network, tspan, target, type_analysis, depth, dom_om)
    return dom_paths
