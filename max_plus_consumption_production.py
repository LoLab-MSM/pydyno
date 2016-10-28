from __future__ import print_function
from collections import OrderedDict
import matplotlib.pylab as plt
import numpy
import sympy
import seaborn as sns
from pysb.integrate import odesolve
from helper_functions import parse_name
from pysb.simulator.base import SimulatorException
import itertools
import pandas as pd
import math


class Tropical:
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):
        """
        Constructor of tropical function
        :param model: PySB model
        """
        self.model = model
        self.tspan = None
        self.y = None
        self.param_values = {}
        self.passengers = []
        self.eqs_for_tropicalization = {}
        self.all_sp_signatures_prod = {}
        self.all_sp_signatures_cons = {}
        self.all_comb = {}

    def tropicalize(self, tspan=None, param_values=None, diff_par=1, ignore=1, epsilon=1,
                    find_passengers_by='imp_nodes',
                    plot_imposed_trace=False, verbose=False):
        """
        tropicalization of driver species
        :param diff_par: Parameter that defines when a monomial or combination of monomials is larger than the others
        :param find_passengers_by: Option to find passenger species. 'imp_nodes' finds the nodes that only have one edge.
        'qssa' finds passenger species using the quasi steady state approach
        :param plot_imposed_trace: Option to plot imposed trace
        :param tspan: Time span
        :param param_values: PySB model parameter values
        :param ignore: Initial time points to ignore
        :param epsilon: Order of magnitude difference between solution of ODE and imposed trace to consider species as
        passenger
        :param verbose: Verbose
        :return:
        """

        if verbose:
            print("Solving Simulation")

        if tspan is not None:
            self.tspan = tspan
        else:
            raise SimulatorException("'tspan' must be defined.")

        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values = numpy.array([p.value for p in self.model.parameters])

        # convert model parameters into dictionary
        self.param_values = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))

        self.y = odesolve(self.model, self.tspan, self.param_values)

        if verbose:
            print("Getting Passenger species")
        if find_passengers_by == 'qssa':
            if plot_imposed_trace:
                self.find_passengers(self.y[ignore:], epsilon, plot=plot_imposed_trace)
            else:
                self.find_passengers(self.y[ignore:], epsilon)
        elif find_passengers_by == 'imp_nodes':
            self.find_nonimportant_nodes()
        else:
            raise Exception("equations to tropicalize must be chosen")

        if verbose:
            print("equation to tropicalize")
        self.equations_to_tropicalize()

        if verbose:
            print("Getting signatures")
        self.signal_signature(self.y, diff_par=diff_par)
        return

    @staticmethod
    def merge_dicts(*dict_args):
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    @staticmethod
    def choose_max2(pd_series, diff_par, prod_comb, cons_comb, type_sign='consumption'):
        """

        :param type_sign: Type of signature. It can be 'consumption' or 'consumption'
        :param cons_comb: combinations of monomials that consume certain species
        :param prod_comb: combinations of monomials that produce certain species
        :param pd_series: Pandas series whose axis labels are the monomials and the data is their values at a specific
        time point
        :param diff_par: Parameter to define when a monomial is larger
        :return: monomial or combination of monomials that dominate at certain time point
        """

        # chooses the larger monomial or combination of monomials that satisfy diff_par
        if type_sign == 'production':
            prod = pd_series[pd_series > 0]
            largest_prod = 'ND'
            for comb in prod_comb.keys():
                # comb is an integer that represents the number of monomials in a combination
                if comb == prod_comb.keys()[-1]:
                    break

                if len(prod_comb[comb].keys()) == 1:
                    largest_prod = prod_comb[comb].keys()[0]
                    break

                monomials_values = {}
                for idx in prod_comb[comb].keys():
                    value = 0
                    for j in prod_comb[comb][idx]:
                        if j not in list(prod.index):
                            value += 1e-100
                        else:
                            value += prod.loc[j]
                    monomials_values[idx] = value

                foo2 = pd.Series(monomials_values).sort_values(ascending=False)
                comb_largest = prod_comb[comb][list(foo2.index)[0]]
                for cm in list(foo2.index):
                    # Compares the largest combination of monomials to other combinations whose monomials that are not
                    # present in comb_largest
                    if len(set(comb_largest) - set(prod_comb[comb][cm])) == len(comb_largest):
                        value_prod_largest = math.log10(foo2.loc[list(foo2.index)[0]])
                        if abs(value_prod_largest - math.log10(foo2.loc[cm])) > diff_par and value_prod_largest > -1:
                            largest_prod = list(foo2.index)[0]
                            break
                if largest_prod != 'ND':
                    break
            return largest_prod

        # chooses the larger monomial or combination of monomials that satisfy diff_par
        elif type_sign == 'consumption':
            cons = pd_series[pd_series < 0]
            largest_cons = 'ND'
            for comb in cons_comb.keys():
                if comb == cons_comb.keys()[-1]:
                    break

                if len(cons_comb[comb].keys()) == 1:
                    largest_cons = cons_comb[comb].keys()

                monomials_values = {}
                for idx in cons_comb[comb].keys():
                    value = 0
                    for j in cons_comb[comb][idx]:
                        if j not in list(cons.index):
                            value += -1e-100
                        else:
                            value += cons.loc[j]
                    monomials_values[idx] = value

                foo2 = pd.Series(monomials_values).sort_values(ascending=True)
                comb_largest = cons_comb[comb][list(foo2.index)[0]]
                for cm in list(foo2.index):
                    # Compares the largest combination of monomials to other combinations whose monomials that are not
                    # present in comb_largest
                    if len(set(comb_largest) - set(cons_comb[comb][cm])) == len(comb_largest):
                        value_cons_largest = math.log10(-foo2.loc[list(foo2.index)[0]])
                        if abs(value_cons_largest - math.log10(-foo2.loc[cm])) > diff_par and value_cons_largest > -1:
                            largest_cons = list(foo2.index)[0]
                            break
                if largest_cons != 'ND':
                    break
            return largest_cons
        else:
            raise Exception('type_sign must be defined')

    def find_nonimportant_nodes(self):
        """

        :return: a list of non-important nodes
        """
        rcts_sp = list(sum([i['reactants'] for i in self.model.reactions_bidirectional], ()))
        pdts_sp = list(sum([i['products'] for i in self.model.reactions_bidirectional], ()))
        imp_rcts = set([x for x in rcts_sp if rcts_sp.count(x) > 1])
        imp_pdts = set([x for x in pdts_sp if pdts_sp.count(x) > 1])
        imp_nodes = set.union(imp_pdts, imp_rcts)
        idx = list(set(range(len(self.model.odes))) - set(imp_nodes))
        self.passengers = idx
        return self.passengers

    def find_passengers(self, y, epsilon=None, plot=False, verbose=False):
        """
        Finds passenger species based in the Quasi Steady State Approach (QSSA) in the model
        :param verbose: Verbose
        :param y: Solution of the differential equations
        :param epsilon: Minimum difference between the imposed trace and the dynamic solution to be considered passenger
        :param plot: Boolean, True to plot the dynamic solution and the imposed trace.
        :return: The passenger species
        """
        sp_imposed_trace = []
        assert not self.passengers

        # Loop through all equations
        for i, eq in enumerate(self.model.odes):
            # Solve equation of imposed trace. It can have more than one solution (Quadratic solutions)
            sol = sympy.solve(eq, sympy.Symbol('__s%d' % i))
            sp_imposed_trace.append(sol)
        for sp_idx, trace_soln in enumerate(sp_imposed_trace):
            distance_imposed = 999
            for idx, solu in enumerate(trace_soln):
                # Check is solution is time independent
                if solu.is_real:
                    imp_trace_values = [float(solu) + self.mach_eps] * (len(self.tspan) - 1)
                else:
                    # If the imposed trace depends on the value of other species, then we replace species and parameter
                    # values to get the imposed trace
                    for p in self.param_values:
                        solu = solu.subs(p, self.param_values[p])

                    # After replacing parameter for its values, then we get the species in the equation and pass
                    # their dynamic solution
                    variables = [atom for atom in solu.atoms(sympy.Symbol)]
                    f = sympy.lambdify(variables, solu, modules=dict(sqrt=numpy.lib.scimath.sqrt))
                    args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
                    imp_trace_values = f(*args)

                if any(isinstance(n, complex) for n in imp_trace_values):
                    if verbose:
                        print("solution {0} from equation {1} is complex".format(idx, sp_idx))
                    continue
                elif any(n < 0 for n in imp_trace_values):
                    if verbose:
                        print("solution {0} from equation {1} is negative".format(idx, sp_idx))
                    continue
                diff_trace_ode = abs(numpy.log10(imp_trace_values) - numpy.log10(y['__s%d' % sp_idx]))
                if max(diff_trace_ode) < distance_imposed:
                    distance_imposed = max(diff_trace_ode)

                if plot:
                    self.plot_imposed_trace(y=y, tspan=self.tspan[1:], imp_trace_values=imp_trace_values,
                                            sp_idx=sp_idx, diff_trace_ode=diff_trace_ode, epsilon=epsilon)

            if distance_imposed < epsilon:
                self.passengers.append(sp_idx)

        return self.passengers

    def plot_imposed_trace(self, y, tspan, imp_trace_values, sp_idx, diff_trace_ode, epsilon):
        """

        :param y: Solution of the differential equations
        :param tspan: time span of the solution of the differential equations
        :param imp_trace_values: Imposed trace values
        :param sp_idx: Index of the molecular species to be plotted
        :param diff_trace_ode: Maxmimum difference between the dynamic and the imposed trace
        :param epsilon: Order of magnitude difference between solution of ODE and imposed trace to consider species as
        passenger
        :return: Plot of the imposed trace and the dnamic solution
        """
        plt.figure()
        plt.semilogy(tspan, imp_trace_values, 'r--', linewidth=5, label='imposed')
        plt.semilogy(tspan, y['__s{0}'.format(sp_idx)], label='full')
        plt.legend(loc=0)
        plt.xlabel('time', fontsize=20)
        plt.ylabel('population', fontsize=20)
        if max(diff_trace_ode) < epsilon:
            plt.title(str(self.model.species[sp_idx]) + 'passenger', fontsize=20)
        else:
            plt.title(self.model.species[sp_idx], fontsize=20)
        plt.savefig('s%d' % sp_idx + '_imposed_trace' + '.png', bbox_inches='tight', dpi=400)

    def equations_to_tropicalize(self):
        """

        :return: Dict, keys are the index of the driver species, values are the differential equations
        """

        idx = list(set(range(len(self.model.odes))) - set(self.passengers))
        eqs = {i: self.model.odes[i] for i in idx}
        self.eqs_for_tropicalization = eqs
        return

    def signal_signature(self, y, diff_par):
        # Dictionary whose keys are species and values are the monomial signatures
        all_signatures_prod = {}
        all_signatures_cons = {}
        for sp in self.eqs_for_tropicalization:

            # Production terms
            prod = []
            # Consumption terms
            cons = []

            for term in self.model.reactions_bidirectional:
                if sp in term['reactants'] and term['reversible'] is True:
                    prod.append((-1) * term['rate'])
                    cons.append((-1) * term['rate'])
                elif sp in term['products']:
                    prod.append(term['rate'])
                elif sp in term['reactants']:
                    cons.append(term['rate'])
            print(sp, self.model.odes[sp])

            # Dictionary whose keys are the symbolic monomials and the values are the simulation results
            mons_dict = {}
            for mon_p in prod:
                mon_p_values = mon_p.subs(self.param_values)
                var_prod = [atom for atom in mon_p_values.atoms(sympy.Symbol)]  # Variables of monomial
                arg_prod = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_prod]
                f_prod = sympy.lambdify(var_prod, mon_p_values)
                prod_values = f_prod(*arg_prod)
                mons_dict[mon_p] = prod_values
            for mon_c in cons:
                if mon_c in prod:
                    continue
                mon_c_values = -mon_c.subs(self.param_values)
                var_cons = [atom for atom in mon_c_values.atoms(sympy.Symbol)]  # Variables of monomial
                arg_cons = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_cons]
                f_cons = sympy.lambdify(var_cons, mon_c_values, modules="numpy")
                cons_values = f_cons(*arg_cons)
                mons_dict[mon_c] = cons_values
            # Dataframe whose rownames are the monomials and the columns contain their values at each time point
            mons_df = pd.DataFrame(mons_dict).T

            prod_comb = OrderedDict()
            cons_comb = OrderedDict()
            prod_idx = 0
            cons_idx = 0

            for L in range(1, len(prod) + 1):
                prod_comb_names = {}
                for subset in itertools.combinations(prod, L):
                    prod_comb_names['P{0}{1}'.format(L, prod_idx)] = subset
                    prod_idx += 1
                prod_comb[L] = prod_comb_names
            self.all_comb[sp] = self.merge_dicts(*prod_comb.values())
            for L in range(1, len(cons) + 1):
                cons_comb_names = {}
                for subset in itertools.combinations(cons, L):
                    cons_comb_names['C{0}{1}'.format(L, cons_idx)] = subset
                    cons_idx += 1
                cons_comb[L] = cons_comb_names
            self.all_comb[sp].update(self.merge_dicts(*cons_comb.values()))
            self.all_comb[sp].update({'ND': 'N'})

            # Substitution matrix
            len_ND = len(max(self.all_comb[sp].values(), key=len)) + 1
            sm = numpy.zeros((len(self.all_comb[sp].keys()), len(self.all_comb[sp].keys())))
            for i, a in enumerate(self.all_comb[sp]):
                for j, b in enumerate(self.all_comb[sp]):
                    if a == 'ND' and b == 'ND':
                        sm[i, j] = 0
                    elif a == 'ND':
                        sm[i, j] = 2 * len_ND - len(self.all_comb[sp][b])
                    elif b == 'ND':
                        sm[i, j] = 2 * len_ND - len(self.all_comb[sp][a])
                    else:
                        sm[i, j] = self.sub_value(self.all_comb[sp][a], self.all_comb[sp][b])
                        # max(len(self.all_comb[sp][a]), len(self.all_comb[sp][b])) - len(
                        #             set(self.all_comb[sp][a]).intersection(self.all_comb[sp][b]))
            sm_df = pd.DataFrame(data=sm, index=self.all_comb[sp].keys(), columns=self.all_comb[sp].keys())
            sm_df.to_csv('/home/oscar/Documents/tropical_earm/subs_matrix/sm_{0}.{1}'.format(sp, 'csv'))

            signature_species_prod = mons_df.apply(self.choose_max2,
                                                   args=(diff_par, prod_comb, cons_comb, 'production'))
            signature_species_cons = mons_df.apply(self.choose_max2,
                                                   args=(diff_par, prod_comb, cons_comb, 'consumption'))

            all_signatures_prod[sp] = list(signature_species_prod)
            all_signatures_cons[sp] = list(signature_species_cons)
        self.all_sp_signatures_prod = all_signatures_prod
        self.all_sp_signatures_cons = all_signatures_cons
        return

    @staticmethod
    def sub_value(a, b):
        value = 2 * len(max(a, b, key=len)) - len(min(a, b, key=len)) - len(set(a).intersection(b))
        return value

    def visualization2(self, sp_to_vis=None):
        if sp_to_vis:
            species_ready = list(set(sp_to_vis).intersection(self.all_sp_signatures_prod.keys()))
        else:
            raise Exception('list of driver species must be defined')

        if not species_ready:
            raise Exception('None of the input species is a driver')

        for sp in species_ready:
            # Setting up figure
            plt.figure(1)
            plt.subplot(414)

            mon_val = OrderedDict()
            signature = self.all_sp_signatures_prod[sp]
            for idx, mon in enumerate(list(set(signature))):
                if mon[0] == 'C':
                    mon_val[self.all_comb[sp][mon] + (-1,)] = idx
                else:
                    mon_val[self.all_comb[sp][mon]] = idx

            mon_rep = [0] * len(signature)
            for i, m in enumerate(signature):
                if m[0] == 'C':
                    mon_rep[i] = mon_val[self.all_comb[sp][m] + (-1,)]
                else:
                    mon_rep[i] = mon_val[self.all_comb[sp][m]]
            # mon_rep = [mon_val[self.all_comb[sp][m]] for m in signature]

            y_pos = numpy.arange(len(mon_val.keys()))
            plt.scatter(self.tspan, mon_rep)
            plt.yticks(y_pos, mon_val.keys())
            plt.ylabel('Monomials', fontsize=16)
            plt.xlabel('Time(s)', fontsize=16)
            plt.xlim(0, self.tspan[-1])
            plt.ylim(0, max(y_pos))

            plt.subplot(413)

            mon_val_cons = OrderedDict()
            signature_cons = self.all_sp_signatures_cons[sp]
            for idx, mon in enumerate(list(set(signature_cons))):
                if mon[0] == 'C':
                    mon_val_cons[self.all_comb[sp][mon] + (-1,)] = idx
                else:
                    mon_val_cons[self.all_comb[sp][mon]] = idx

            mon_rep_cons = [0] * len(signature_cons)
            for i, m in enumerate(signature_cons):
                if m[0] == 'C':
                    mon_rep_cons[i] = mon_val_cons[self.all_comb[sp][m] + (-1,)]
                else:
                    mon_rep_cons[i] = mon_val_cons[self.all_comb[sp][m]]
            # mon_rep = [mon_val[self.all_comb[sp][m]] for m in signature]

            y_pos_cons = numpy.arange(len(mon_val_cons.keys()))
            plt.scatter(self.tspan, mon_rep_cons)
            plt.yticks(y_pos_cons, mon_val_cons.keys())
            plt.ylabel('Monomials', fontsize=16)
            plt.xlim(0, self.tspan[-1])
            plt.ylim(0, max(y_pos_cons))

            plt.subplot(412)
            for name in self.model.odes[sp].as_coefficients_dict():
                mon = name
                mon = mon.subs(self.param_values)
                var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
                arg_f1 = [numpy.maximum(self.mach_eps, self.y[str(va)]) for va in var_to_study]
                f1 = sympy.lambdify(var_to_study, mon)
                mon_values = f1(*arg_f1)
                mon_name = str(name).partition('__')[2]
                plt.plot(self.tspan, mon_values, label=mon_name)
            plt.ylabel('Rate(m/sec)', fontsize=16)
            plt.legend(bbox_to_anchor=(-0.1, 0.85), loc='upper right', ncol=3)

            plt.subplot(411)
            plt.plot(self.tspan, self.y['__s%d' % sp], label=parse_name(self.model.species[sp]))
            plt.ylabel('Molecules', fontsize=16)
            plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=1)
            plt.suptitle('Tropicalization' + ' ' + str(self.model.species[sp]))

            # plt.show()
            plt.savefig('s%d' % sp + '.png', bbox_inches='tight', dpi=400)
            plt.clf()

    def get_passenger(self):
        """

        :return: Passenger species of the systems
        """
        return self.passengers

    def get_species_signatures_prod(self):
        """

        :return: Signatures of the molecular species
        """
        return self.all_sp_signatures_prod

    def get_species_signatures_cons(self):
        """

        :return: Signatures of the molecular species
        """
        return self.all_sp_signatures_cons


def run_tropical(model, tspan, parameters=None, diff_par=1, sp_visualize=None):
    """

    :param diff_par:
    :param model: PySB model of a biological system
    :param tspan: Time of the simulation
    :param parameters: Parameter values of the PySB model
    :param sp_visualize: Species to visualize
    :return: The tropical signatures of all non-passenger species
    """
    tr = Tropical(model)
    tr.tropicalize(tspan=tspan, param_values=parameters, diff_par=diff_par)
    if sp_visualize is not None:
        tr.visualization2(sp_to_vis=sp_visualize)
        # tr.visualization(driver_species=sp_visualize)
    return tr.get_species_signatures_prod(), tr.get_species_signatures_cons()
