from pysb.simulator import ScipyOdeSimulator
import numpy as np
import networkx as nx
from pysb.bng import generate_equations
import re
import pandas as pd
from math import log10
import sympy
from pysb.simulator import SimulationResult
import pickle
try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None

try:
    import h5py
except ImportError:
    h5py = None


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

    """
    if isinstance(simulations, str):
        if h5py is None:
            raise Exception('please install the h5py package for this feature')
        if h5py.is_hdf5(simulations):
            sims = h5py.File(simulations)
            model = pickle.loads(sims.values()[0]['_model'][()])
            parameters = sims.values()[0]['result']['param_values']
            trajectories = sims.values()[0]['result']['trajectories']
            sim_tout = sims.values()[0]['result']['tout']
            if all_equal(sim_tout):
                tspan = sim_tout[0]
            else:
                raise Exception('Analysis is not supported for simulations with different time spans')
        else:
            raise TypeError('File format not supported')
    elif isinstance(simulations, SimulationResult):
        sims = simulations
        model = sims._model
        parameters = sims.param_values
        trajectories = sims.species
        tspan = sims.tout[0]
    else:
        raise TypeError('format not supported')
    nsims = len(parameters)
    return model, trajectories, parameters, nsims, tspan


class DomPath(object):
    def __init__(self, model, tspan, ref, target, depth):
        self.model = model
        self.tspan = tspan
        self.ref = ref
        self.target = target
        self.depth = depth
        self.network = self.create_bipartite_graph()

    def create_bipartite_graph(self):
        generate_equations(self.model)
        graph = nx.DiGraph(name=self.model.name)
        ic_species = [cp for cp, parameter in self.model.initial_conditions]
        for i, cp in enumerate(self.model.species):
            species_node = 's%d' % i
            slabel = re.sub(r'% ', r'%\\l', str(cp))
            slabel += '\\l'
            color = "#ccffcc"
            # color species with an initial condition differently
            if len([s for s in ic_species if s.is_equivalent_to(cp)]):
                color = "#aaffff"
            graph.add_node(species_node,
                           label=slabel,
                           shape="Mrecord",
                           fillcolor=color, style="filled", color="transparent",
                           fontsize="12",
                           margin="0.06,0")
        for i, reaction in enumerate(self.model.reactions_bidirectional):
            reaction_node = 'r%d' % i
            graph.add_node(reaction_node,
                           label=reaction_node,
                           shape="circle",
                           fillcolor="lightgray", style="filled", color="transparent",
                           fontsize="12",
                           width=".3", height=".3", margin="0.06,0")
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
                self.r_link(graph, s, i, arrowhead="odiamond")
        return graph

    @staticmethod
    def r_link(graph, s, r, **attrs):
        nodes = ('s%d' % s, 'r%d' % r)
        if attrs.get('_flip'):
            del attrs['_flip']
            nodes = reversed(nodes)
        attrs.setdefault('arrowhead', 'normal')
        graph.add_edge(*nodes, **attrs)

    def get_reaction_flux_df(self, trajectories, parameters):

        param_values = parameters
        rxns_names = ['r{0}'.format(rxn) for rxn in range(len(self.model.reactions_bidirectional))]
        rxns_df = pd.DataFrame(columns=self.tspan, index=rxns_names)
        param_dict = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))
        # sim_result = ScipyOdeSimulator(model, tspan=tspan, param_values=param_dict).run()
        # y_df = sim_result.all
        for idx, reac in enumerate(self.model.reactions_bidirectional):
            rate_reac = reac['rate']
            variables = [atom for atom in rate_reac.atoms(sympy.Symbol)]
            args = [0] * len(variables)  # arguments to put in the lambdify function
            for idx2, va in enumerate(variables):
                if str(va).startswith('__'):
                    sp_idx = int(''.join(filter(str.isdigit, str(va))))
                    args[idx2] = trajectories[:, sp_idx]
                else:
                    args[idx2] = param_dict[va.name]
            func = sympy.lambdify(variables, rate_reac, modules=dict(sqrt=np.lib.scimath.sqrt))
            react_rate = func(*args)
            # edge_name = get_edges_name(reac['reactants'], reac['products'])
            # rxns_df.rename(index={int(idx): edge_name}, inplace=True)
            rxns_df.loc['r{0}'.format(idx)] = react_rate
        rxns_df['Total'] = rxns_df.sum(axis=1)
        return rxns_df

    @staticmethod
    def get_reaction_incoming_species(network, r):
        in_edges = network.in_edges(r)
        sp_nodes = [n[0] for n in in_edges]
        return sp_nodes

    def get_dominant_paths(self, trajectories, parameters):
        """

        Parameters
        ----------
        ref: int, A number that is added to the dictionary values of the path_labels so they can be distinguished in
        different simulations
        target : Node label from network, Node from which the pathway starts
        depth : int, The depth of the pathway

        Returns
        -------

        """
        self.create_bipartite_graph()
        dom_om = 0.5 # Order of magnitude to consider dominancy
        reaction_flux_df = self.get_reaction_flux_df(trajectories, parameters)

        path_labels = {}
        signature = [0] * len(self.tspan[1:])
        prev_neg_rr = []
        for label, t in enumerate(self.tspan[1:]):
            # Get reaction rates that are negative
            neg_rr = reaction_flux_df.index[reaction_flux_df[t] < 0].tolist()
            if not neg_rr or prev_neg_rr == neg_rr:
                pass
            else:
                rr_changes = list(set(neg_rr).symmetric_difference(set(prev_neg_rr)))

                # Now we are going to flip the edges whose reaction rate value is negative
                for r_node in rr_changes:
                    in_edges = self.network.in_edges(r_node)
                    out_edges = self.network.out_edges(r_node)
                    edges_to_remove = list(in_edges) + list(out_edges)
                    self.network.remove_edges_from(edges_to_remove)
                    edges_to_add = [edge[::-1] for edge in edges_to_remove]
                    self.network.add_edges_from(edges_to_add)

                prev_neg_rr = neg_rr

            dom_nodes = {self.target: [[self.target]]}
            all_rdom_noodes = []
            t_paths = [0] * self.depth
            for d in range(self.depth):
                all_dom_nodes = {}
                for node_doms in dom_nodes.values():
                    flat_node_doms = [item for items in node_doms for item in items]
                    for node in flat_node_doms:
                        in_edges = self.network.in_edges(node)
                        # for edge in in_edges: print (node, edge)
                        fluxes_in = {edge: log10(abs(reaction_flux_df.loc[edge[0], t])) for edge in in_edges}
                        max_val = np.amax(fluxes_in.values())
                        dom_r_nodes = [n[0] for n, i in fluxes_in.items() if i > (max_val - dom_om) and max_val > -5]
                        dom_sp_nodes = [self.get_reaction_incoming_species(self.network, reaction_nodes) for reaction_nodes in dom_r_nodes]
                            # [n[0] for reaction_nodes in dom_r_nodes for n in network.in_edges(reaction_nodes)]
                        # Get the species nodes from the reaction nodes to keep back tracking the pathway
                        all_dom_nodes[node] = dom_sp_nodes
                        all_rdom_noodes.append(dom_r_nodes)
                    dom_nodes = all_dom_nodes

                t_paths[d] = dom_nodes
            all_rdom_noodes_str = str(all_rdom_noodes)
            check_paths = [all_rdom_noodes_str == path for path in path_labels.keys()]
            if not any(check_paths):
                path_labels[all_rdom_noodes_str] = label + self.ref
                signature[label] = all_rdom_noodes_str
            else:
                signature[label] = np.array(path_labels.keys())[check_paths][0]

        return signature, path_labels


def run_dompath_single(simulations, ref, target, depth):
    model, trajectories, parameters, nsims, tspan = get_simulations(simulations)
    dompath = DomPath(model, tspan, ref, target, depth)
    signatures = dompath.get_dominant_paths(trajectories, parameters[0])
    return signatures

def run_dompath_multi(simulations, ref, target, depth, cpu_cores=1):
    if Pool is None:
        raise Exception('Plese install the pathos package for this feature')
    model, trajectories, parameters, nsims, tspan = get_simulations(simulations)
    dompath = DomPath(model, tspan, ref, target, depth)
    if nsims == 1:
        trajectories = [trajectories]
    p = Pool(cpu_cores)
    res = p.amap(dompath.get_dominant_paths, trajectories, parameters)
    signatures_labels = res.get()
    signatures = [0] * len(signatures_labels)
    labels = [0] * len(signatures_labels)
    for idx, sl in enumerate(signatures_labels):
        signatures[idx] = sl[0]
        labels[idx] = sl[1]
    labels = {k: v for d in labels for k, v in d.items()}
    signatures = [[labels[label] for label in signa] for signa in signatures]
    return signatures
