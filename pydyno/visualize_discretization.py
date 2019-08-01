import numpy as np
import matplotlib.pyplot as plt
import sympy
from pydyno.util import parse_name, rate_2_interactions, label2rr
import re
from pysb.bng import generate_equations
from anytree.importer import DictImporter
from anytree.exporter import DotExporter


def visualization_sp(sim, sim_idx, sp_to_vis, all_signatures, plot_type):
    """

    Parameters
    ----------
    sim : pysb.simulator.SimulationResult
        Simulation from which the signatures were obtained
    sim_idx : Int,
        Index of the simulation to use for the visualization
    sp_to_vis : Int
        Index of the species discretization to visualize
    all_signatures: pd.DataFrame
        Pandas dataframe that contains the signatures
    plot_type: str
        `p` for production and `c` for consumption

    """

    model = sim._model
    tspan = sim.tout[sim_idx]
    param_values = sim.param_values[sim_idx]
    mach_eps = np.finfo(float).eps
    par_name_idx = {j.name: i for i, j in enumerate(model.parameters)}

    for sp in sp_to_vis:
        sp = int(sp)
        sp_plot = '__s{0}_{1}'.format(sp, plot_type)

        # Setting up figure
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
        fig.subplots_adjust(hspace=0.4)

        signature = all_signatures.loc[sp_plot].values[sim_idx]

        axs[2].scatter(tspan, [str(s) for s in signature])
        # plt.yticks(list(set(signature)))
        axs[2].set_ylabel('Dom rxns', fontsize=12)
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
                    sp_idx = str(va)
                    arg_f1[idx] = np.maximum(mach_eps, sim.all[sim_idx][sp_idx])
                else:
                    arg_f1[idx] = param_values[par_name_idx[va.name]]

            f1 = sympy.lambdify(var_to_study, mon)
            mon_values = f1(*arg_f1)
            mon_name = rate_2_interactions(model, str(mon))
            axs[1].plot(tspan, mon_values, label='{0}: {1}'.format(rr_idx, mon_name))
        axs[1].set_ylabel(r'Rate [$\mu$M/s]', fontsize=12)
        axs[1].legend(bbox_to_anchor=(1., 0.85), ncol=3, title='Reaction rates')

        # TODO: fix this for observables.
        axs[0].plot(tspan, sim.all[sim_idx]['__s{0}'.format(sp)], label=parse_name(model.species[sp]))
        axs[0].set_ylabel(r'Conc [$\mu$M]', fontsize=12)
        axs[0].legend(bbox_to_anchor=(1, 0.85), ncol=1)
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


def visualization_seq_paths(sim, sim_idx, all_signatures):
    """

    Parameters
    ----------
    sim : pysb.simulator.SimulationResult
        Simulation from which the signatures were obtained
    sim_idx : Int,
        Index of the simulation to use for the visualization
    all_signatures: tropical.sequences.Sequences
        Sequences class that contains the signatures

    """

    model = sim._model
    tspan = sim.tout[sim_idx]
    sp_plot = int(''.join(filter(str.isdigit, all_signatures.target)))

    # Setting up figure
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.subplots_adjust(hspace=0.4)

    signature = all_signatures.sequences.loc[sim_idx].values[0]
    axs[1].scatter(tspan[1:], signature)
    # axs[1].set_yticks(list(set(signature)))
    axs[1].set_ylabel('Dom paths', fontsize=12)
    axs[1].set_xlabel('Time(s)', fontsize=14)
    axs[1].set_xlim(0, tspan[-1])
    # plt.ylim(0, max(y_pos))

    # TODO: fix this for observables.
    if sim.nsims == 1 and sim.squeeze:
        sp_sim = sim.all['__s{0}'.format(sp_plot)]
    else:
        sp_sim = sim.all[sim_idx]['__s{0}'.format(sp_plot)]

    axs[0].plot(tspan, sp_sim, label=parse_name(model.species[sp_plot]))
    axs[0].set_ylabel(r'Conc [$\mu$M]', fontsize=12)
    axs[0].legend(bbox_to_anchor=(1, 0.85), ncol=1)
    fig.suptitle('Discretization' + ' ' + parse_name(model.species[sp_plot]), y=1.0)

    # plt.tight_layout()
    fig.savefig('s{0}'.format(sp_plot) + '.pdf', format='pdf', bbox_inches='tight')


