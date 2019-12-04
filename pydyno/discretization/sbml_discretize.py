import re
from concurrent.futures import ProcessPoolExecutor
from pysb.simulator.scipyode import SerialExecutor
import networkx as nx
import pydyno.discretization.base as base
from pydyno.seqanalysis import SeqAnalysis


class SbmlDomPath(base.DomPath):
    """
    Class to discretize the simulated trajectory of a model species
    Parameters
    ----------
    model: PySB model
        Model to analyze
    simulations: PySB SimulationResult object or str
        simulations used to perform the analysis. If str it should be the
        path to a simulation result in hdf5 format
    """

    def __init__(self, model, simulations):
        super().__init__(model)
        self._trajectories = simulations.trajectories
        self._nsims = len(self._trajectories)
        self._parameters = simulations.parameters
        self._tspan = simulations.tspan
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

    def get_reaction_flux_df(self, simulation_idx):
        return self.all_reaction_flux[simulation_idx]

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

        results = []
        with SerialExecutor() if num_processors == 1 else \
                ProcessPoolExecutor(max_workers=num_processors) as executor:
            for n in nsims:
                reaction_flux_df = self.get_reaction_flux_df(n)
                results.append(executor.submit(
                    base._dominant_paths,
                    network,
                    reaction_flux_df,
                    self.tspan,
                    target,
                    type_analysis,
                    depth,
                    dom_om
                ))
            signatures_labels = [r.result() for r in results]

        signatures = [0] * len(signatures_labels)
        labels = [0] * len(signatures_labels)
        for idx, sl in enumerate(signatures_labels):
            signatures[idx] = sl[0]
            labels[idx] = sl[1]
        signatures_df, new_paths = base._reencode_signatures_paths(signatures, labels, self.tspan)
        # signatures_labels = {'signatures': signatures, 'labels': all_labels}
        return SeqAnalysis(signatures_df, target), new_paths
