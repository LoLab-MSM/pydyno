import numpy as np
import matplotlib.pyplot as plt
import sympy
from tropical.util import parse_name, rate_2_interactions, label2rr
from future.utils import iteritems


def visualization(model, tspan, y, sp_to_vis, all_signatures, plot_type, param_values):
    mach_eps = np.finfo(float).eps
    species_ready = list(set(sp_to_vis).intersection(all_signatures.keys()))
    par_name_idx = {j.name: i for i, j in enumerate(model.parameters)}
    if not species_ready:
        raise Exception('None of the input species is a driver')

    for sp in species_ready:

        # Setting up figure
        plt.figure(1)
        plt.subplot(313)

        signature = all_signatures[sp][plot_type]
        # if not signature:
        #     continue

        # mon_val = OrderedDict()
        # merged_mon_comb = self.merge_dicts(*self.all_comb[sp].values())
        # merged_mon_comb.update({'ND': 'N'})
        #
        # for idx, mon in enumerate(list(set(signature))):
        #     mon_val[merged_mon_comb[mon]] = idx
        #
        # mon_rep = [0] * len(signature)
        # for i, m in enumerate(signature):
        #     mon_rep[i] = mon_val[merged_mon_comb[m]]
        # mon_rep = [mon_val[self.all_comb[sp][m]] for m in signature]
        plt.scatter(tspan, [str(s) for s in signature])
        # plt.yticks(list(set(signature)))
        plt.ylabel('Dominant terms', fontsize=12)
        plt.xlabel('Time(s)', fontsize=14)
        plt.xlim(0, tspan[-1])
        # plt.ylim(0, max(y_pos))
        plt.subplot(312)
        reaction_rates = label2rr(model, sp)
        for rr_idx, rr in iteritems(reaction_rates):
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
            plt.plot(tspan, mon_values, label='{0}: {1}'.format(rr_idx, mon_name))
        plt.ylabel(r'Rate [$\mu$M/s]', fontsize=12)
        plt.legend(bbox_to_anchor=(1., 0.85), ncol=3, title='Reaction rates')
        plt.xlim(0, tspan[-1])

        plt.subplot(311)
        # TODO: fix this for observables.
        plt.plot(tspan, y[:, sp], label=parse_name(model.species[sp]))
        plt.ylabel(r'Concentration [$\mu$M]', fontsize=12)
        plt.xlim(0, tspan[-1])
        plt.legend(bbox_to_anchor=(1.23, 0.85), ncol=1)
        plt.suptitle('Discretization' + ' ' + parse_name(model.species[sp]), y=1.08)

        plt.tight_layout()
        plt.savefig('s%d' % 27 + '.pdf', format='pdf', bbox_inches='tight')
