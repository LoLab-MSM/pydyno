import re
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
from pydyno.discretization.base import DomPath, _reencode_signatures_paths, _dominant_paths
from pydyno.seqanalysis import SeqAnalysis
from pysb.simulator.scipyode import SerialExecutor


class SbmlDomPath(DomPath):
    """
    Obtain dominant paths from models encoded in the SBML format.

    Parameters
    ----------
    model: SbmlModel object
        Model to analyze. For more information about the SbmlModel object
        check :class:`pydyno.util_tellurium.SbmlModel`
    simulations: SbmlSimulation
        simulations used to perform the analysis. For more information about the
        SbmlSimulation object check: :class:`pydyno.util_tellurium.SbmlSimulation`

        Examples
    --------
    Obtain the discretized trajectory of the extrinsic apoptosis reaction model

    >>> from pydyno.discretization import SbmlDomPath
    >>> from pydyno.util_tellurium import SbmlModel, SbmlSimulation
    >>> import numpy as np
    >>> import tellurium as te
    >>> # Load model
    >>> r = te.loadSBMLModel('double_enzymatic_sbml.xml')
    >>> # Create SbmlModel and SbmlSimulation objects
    >>> model = SbmlModel(r)
    >>> sim = SbmlSimulation()
    >>> # Simulate model and add simulation results to SbmlSimulation object
    >>> r.selections = ['time', '__s0', '__s1', '__s2', '__s3', '__s4', '__s5', 'r0', 'r1', 'r2', 'r3']
    >>> r.simulate(0, 100, 100)
    >>> sim.add_simulation(r)
    >>> # Obtain dominant paths that consume species 0
    >>> dp = SbmlDomPath(model=model, simulations=sim)
    >>> signs, paths = dp.get_path_signatures(target='s0', type_analysis='consumption', depth=2, dom_om=1)
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
        self._trajectories = simulations.trajectories
        self._nsims = len(self._trajectories)
        self._parameters = simulations.parameters
        self._tspan = simulations.tspan[0]
        self.all_reaction_flux = simulations.reaction_flux_df

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
            dom_path_partial = partial(_dominant_paths, network=network, tspan=self.tspan, target=target,
                                       type_analysis=type_analysis, depth=depth, dom_om=dom_om)

            results = [executor.submit(dom_path_partial, self.all_reaction_flux[n])
                       for n in nsims]
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
