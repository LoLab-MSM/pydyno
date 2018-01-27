import numpy as np
import matplotlib.pyplot as plt
import sympy
from tropical.util import parse_name

def visualization(model, tspan, y, sp_to_vis, all_signatures, all_comb, param_values):
    mach_eps = np.finfo(float).eps
    species_ready = list(set(sp_to_vis).intersection(all_signatures.keys()))
    par_name_idx = {j.name: i for i, j in enumerate(model.parameters)}
    if not species_ready:
        raise Exception('None of the input species is a driver')

    for sp in species_ready:

        # Setting up figure
        plt.figure(1)
        plt.subplot(313)

        signature = all_signatures[sp][1]
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
        plt.scatter(tspan, signature)
        plt.yticks(list(set(signature)))
        plt.ylabel('Dominant terms', fontsize=14)
        plt.xlabel('Time(s)', fontsize=14)
        plt.xlim(0, tspan[-1])
        # plt.ylim(0, max(y_pos))
        plt.subplot(312)
        for val, rr in all_comb[sp]['reactants'][1].items():
            mon = rr[0]
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
            mon_name = str(val)
            plt.plot(tspan, mon_values, label=mon_name)
        plt.ylabel('Rate(m/sec)', fontsize=14)
        plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=3)
        plt.xlim(0, tspan[-1])

        plt.subplot(311)
        plt.plot(tspan, y[:, sp], label=parse_name(model.species[sp]))
        plt.ylabel('Molecules', fontsize=14)
        plt.xlim(0, tspan[-1])
        plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=1)
        plt.suptitle('Tropicalization' + ' ' + str(model.species[sp]), y=1.08)

        plt.tight_layout()
        plt.savefig('s%d' % sp + '.png', bbox_inches='tight', dpi=400)
