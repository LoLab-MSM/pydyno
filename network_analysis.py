import pysb.bng
import networkx
from earm.lopez_embedded import model
import matplotlib.pyplot as plt


def r_link(graph, s, r, **attrs):
    nodes = (s, r)
    if attrs.get('_flip'):
        del attrs['_flip']
        nodes = reversed(nodes)
    attrs.setdefault('arrowhead', 'normal')
    graph.add_edge(*nodes, **attrs)


def construct_graph(model):
    if model.odes is None or model.odes == []:
        pysb.bng.generate_equations(model)

    graph_start = networkx.MultiDiGraph(rankdir="LR")
    labels = {}
    for i, cp in enumerate(model.species):
        labels[i] = str(i)
        species_node = i
        graph_start.add_node(species_node, label=cp)
    for i, reaction in enumerate(model.reactions):
        #             reactants = set([str(self.model.species[j]) for j in reaction['reactants']])
        #             products = set([str(self.model.species[j]) for j in reaction['products']])

        reactants = set(reaction['reactants'])
        products = set(reaction['products'])
        attr_reversible = {}
        for s in reactants:
            for p in products:
                r_link(graph_start, s, p, **attr_reversible)
    return graph_start, labels


graph, labels = construct_graph(model)
pos = networkx.spring_layout(graph)
central = networkx.degree_centrality(graph)
networkx.draw(graph, pos, with_labels=False)
networkx.draw_networkx_labels(graph, pos, labels, font_size=10)
plt.axis('off')
plt.show()

