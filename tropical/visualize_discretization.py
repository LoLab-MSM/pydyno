import numpy as np
import matplotlib.pyplot as plt
import sympy
from tropical.util import parse_name, rate_2_interactions, label2rr
import re
from pysb.bng import generate_equations
from anytree.importer import DictImporter
from anytree.exporter import DotExporter


def visualization_sp(model, tspan, y, sp_to_vis, all_signatures, plot_type, param_values):
    """

    :param model: pysb model
    :param tspan: vector-like, Time of the simulation
    :param y: species simulation
    :param sp_to_vis: Int, species index to visualize
    :param all_signatures: signatures from tropical
    :param plot_type: str, `p` for production and `c` and consumption
    :param param_values: Parameters used for the simulation
    :return:
    """
    mach_eps = np.finfo(float).eps
    species_ready = list(set(sp_to_vis).intersection(all_signatures.keys()))
    par_name_idx = {j.name: i for i, j in enumerate(model.parameters)}
    if not species_ready:
        raise Exception('None of the input species is a driver')

    for sp in species_ready:
        sp = int(sp)
        sp_plot = '__s{0}_{1}'.format(sp, plot_type)

        # Setting up figure
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        fig.subplots_adjust(hspace=0.4)

        signature = all_signatures.loc[sp_plot].values[0]

        axs[2].scatter(tspan, [str(s) for s in signature])
        # plt.yticks(list(set(signature)))
        axs[2].set_ylabel('Dominant terms', fontsize=12)
        axs[2].set_xlabel('Time(s)', fontsize=14)
        axs[2].set_xlim(0, tspan[-1])
        # plt.ylim(0, max(y_pos))

        reaction_rates = label2rr(model, sp)
        for rr_idx, rr in reaction_rates.items():
            mon = rr
            var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
            arg_f1 = [0] * len(var_to_study)
            for idx, va in enumerate(var_to_study):
                if str(va).startswith('__'):
                    sp_idx = int(''.join(filter(str.isdigit, str(va))))
                    arg_f1[idx] = np.maximum(mach_eps, y[:, sp_idx])
                else:
                    arg_f1[idx] = param_values[par_name_idx[va.name]]

            f1 = sympy.lambdify(var_to_study, mon)
            mon_values = f1(*arg_f1)
            mon_name = rate_2_interactions(model, str(mon))
            axs[1].plot(tspan, mon_values, label='{0}: {1}'.format(rr_idx, mon_name))
        axs[1].set_ylabel(r'Rate [$\mu$M/s]', fontsize=12)
        axs[1].legend(bbox_to_anchor=(1., 0.85), ncol=3, title='Reaction rates')

        # TODO: fix this for observables.
        axs[0].plot(tspan, y[:, sp], label=parse_name(model.species[sp]))
        axs[0].set_ylabel(r'Concentration [$\mu$M]', fontsize=12)
        axs[0].legend(bbox_to_anchor=(1.32, 0.85), ncol=1)
        fig.suptitle('Discretization' + ' ' + parse_name(model.species[sp]), y=1.0)

        # plt.tight_layout()
        fig.savefig('s{0}'.format(sp) + '.pdf', format='pdf', bbox_inches='tight')


def visualization_path(model, path, type_analysis, filename):
    """
    Visualize dominant path
    Parameters
    ----------
    model: pysb.Model
        pysb model used for analysis
    path: Dict
        Dictionary that have the tree structure of the path
    type_analysis: str
        Type of analysis done to obtain the path. It can either be `production` or `consumption`
    filename: str
        File name including the extension of the image file

    Returns
    -------

    """
    generate_equations(model)

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

    importer = DictImporter()
    root = importer.import_(path)

    if type_analysis == 'production':
        DotExporter(root, graph='strict digraph', options=["rankdir=TB;"], nodenamefunc=nodenamefunc,
                    edgeattrfunc=edgeattrfunc).to_picture(filename)
    elif type_analysis == 'consumption':
        DotExporter(root, graph='strict digraph', options=["rankdir=TB;"], nodenamefunc=nodenamefunc,
                    edgeattrfunc=None).to_picture(filename)
    else:
        raise ValueError('Type of visualization not implemented')


