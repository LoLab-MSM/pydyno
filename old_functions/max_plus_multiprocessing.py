from __future__ import print_function
import functools
import itertools
import math
import time
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
import matplotlib
import numpy
import pandas as pd
import sympy
import tropical.helper_functions as hf
from pysb.simulator import ScipyOdeSimulator, SimulatorException

matplotlib.use('AGG')
import matplotlib.pyplot as plt

# This is a global function that takes the class object as a parameter to compute the dynamical signature.
# This global function is necessary to use the multiprocessing module.


def dynamic_signatures(param_values, tropical_object, tspan=None, type_sign='production', diff_par=1, ignore=1,
                       epsilon=1, find_passengers_by='imp_nodes', max_comb=None, sp_to_visualize=None,
                       plot_imposed_trace=False, verbose=False):
    """

    :param param_values: Parameter values needed to simulate the PySB model
    :param tropical_object: Instance of Tropical class
    :param tspan: Time of simulation
    :param type_sign: Type of signature. It can be 'consumption' or 'production'
    :param diff_par: Magnitude difference that defines that a reaction is dominant over others
    :param ignore: Number of time points to ignore in simulation (related to equilibration)
    :param epsilon:
    :param find_passengers_by: str, it can be 'imp_nodes' or 'qssa' is the way to find the passenger species
    :param max_comb: int, maximum number of combination of monomials to find dominant monomials
    :param sp_to_visualize: Molecular species to visualize its signature
    :param plot_imposed_trace: Boolean, to see the imposed trace in the QSSA approach
    :param verbose: Boolean
    :return: Dynamical signatures of all driver species
    """
    if tropical_object.is_setup is False:
        tropical_object.setup_tropical(tspan, type_sign, find_passengers_by, max_comb, verbose)

    if find_passengers_by == 'qssa':
        all_signatures = tropical_object.qssa_signal_signature(param_values, diff_par, epsilon, ignore,
                                                               plot_imposed_trace, sp_to_visualize)

    elif find_passengers_by == 'imp_nodes':
        all_signatures = tropical_object.signal_signature(param_values, diff_par=diff_par, sp_to_visualize=sp_to_visualize)
    else:
        raise Exception('A valid way to get the signatures must be provided')
    return all_signatures


class Tropical:
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):
        """
        Constructor of DynSign function
        :param model: PySB model
        """
        self.model = model
        self.tspan = None
        self.passengers = []
        self.eqs_for_tropicalization = {}
        self.all_comb = {}
        self.sim = None
        self.is_setup = False
        self.type_sign = ''

    def tropicalize(self, tspan=None, param_values=None, type_sign='production', diff_par=1, ignore=1, epsilon=1,
                    find_passengers_by='imp_nodes', max_comb=None, sp_to_visualize=None, plot_imposed_trace=False,
                    verbose=False):
        """
        tropicalization of driver species
        :param tspan: Time span
        :param param_values: PySB model parameter values
        :param type_sign: Type of max-plus signature. This is to see the way a species is being produced or consumed
        :param diff_par: Parameter that defines when a monomial or combination of monomials is larger than the others
        :param ignore: Initial time points to ignore
        :param epsilon: Order of magnitude difference between solution of ODE and imposed trace to consider species as
         passenger
        :param find_passengers_by: Option to find passenger species. 'imp_nodes' finds the nodes that only have one edge
        'qssa' finds passenger species using the quasi steady state approach
        :param max_comb:
        :param sp_to_visualize:
        :param plot_imposed_trace: Option to plot imposed trace
        :param verbose: Verbose
        :return:
        """
        all_signatures = dynamic_signatures(param_values, self, tspan=tspan, type_sign=type_sign, diff_par=diff_par,
                                            ignore=ignore, epsilon=epsilon, find_passengers_by=find_passengers_by,
                                            max_comb=max_comb, sp_to_visualize=sp_to_visualize,
                                            plot_imposed_trace=plot_imposed_trace, verbose=verbose)

        return all_signatures

    def _setup_tropical(self, tspan, type_sign, find_passengers_by, max_comb, verbose):
        """

        :param max_comb:
        :param verbose:
        :param tspan: time of simulation
        :param type_sign: type of dynamical signature. It can either 'production' or 'consumption
        :param find_passengers_by: Method to find non important species
        :return:
        """
        if verbose:
            print('setting up time span')
        if tspan is not None:
            self.tspan = tspan
        else:
            raise SimulatorException("'tspan' must be defined.")

        if verbose:
            print('setting up type signature')
        if type_sign not in ['production', 'consumption']:
            raise ValueError('Wrong type_sign')
        else:
            self.type_sign = type_sign

        if verbose:
            print('setting up the simulator class')
        self.sim = ScipyOdeSimulator(self.model, self.tspan)

        if verbose:
            print('setting up the important nodes')
        if find_passengers_by is 'imp_nodes':
            self.find_nonimportant_nodes()
            self.equations_to_tropicalize()
            if not self.all_comb:
                self.set_combinations_sm(max_comb=max_comb)

        self.is_setup = True
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
    def choose_max2(pd_series, diff_par, mon_comb, type_sign):
        """

        :param type_sign: Type of signature. It can be 'consumption' or 'consumption'
        :param mon_comb: combinations of monomials that produce certain species
        :param pd_series: Pandas series whose axis labels are the monomials and the data is their values at a specific
        time point
        :param diff_par: Parameter to define when a monomial is larger
        :return: monomial or combination of monomials that dominate at certain time point
        """
        if type_sign == 'production':
            monomials = pd_series[pd_series > 0]
            value_to_add = 1e-100
            sign = 1
            ascending = False
        elif type_sign == 'consumption':
            monomials = pd_series[pd_series < 0]
            value_to_add = -1e-100
            sign = -1
            ascending = True
        else:
            raise Exception('Wrong type_sign')

        # chooses the larger monomial or combination of monomials that satisfy diff_par
        largest_prod = 'ND'
        for comb in mon_comb.keys():
            # comb is an integer that represents the number of monomials in a combination
            if comb == mon_comb.keys()[-1]:
                break

            if len(mon_comb[comb].keys()) == 1:
                largest_prod = mon_comb[comb].keys()[0]
                break

            monomials_values = {}
            for idx in mon_comb[comb].keys():
                value = 0
                for j in mon_comb[comb][idx]:
                    # j(reversible) might not be in the prod df because it has a negative value
                    if j not in list(monomials.index):
                        value += value_to_add
                    else:
                        value += monomials.loc[j]
                monomials_values[idx] = value

            foo2 = pd.Series(monomials_values).sort_values(ascending=ascending)
            comb_largest = mon_comb[comb][list(foo2.index)[0]]
            for cm in list(foo2.index):
                # Compares the largest combination of monomials to other combinations whose monomials that are not
                # present in comb_largest
                if len(set(comb_largest) - set(mon_comb[comb][cm])) == len(comb_largest):
                    value_prod_largest = math.log10(sign * foo2.loc[list(foo2.index)[0]])
                    if abs(value_prod_largest - math.log10(sign * foo2.loc[cm])) > diff_par and value_prod_largest > -5:
                        largest_prod = list(foo2.index)[0]
                        break
            if largest_prod != 'ND':
                break
        return largest_prod

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

    def find_passengers(self, y, params_ready, epsilon=1, ignore=1, plot=False, verbose=False):
        """
        Finds passenger species based in the Quasi Steady State Approach (QSSA) in the model
        :param params_ready:
        :param verbose: Verbose
        :param y: Solution of the differential equations
        :param epsilon: Minimum difference between the imposed trace and the dynamic solution to be considered passenger
        :param ignore:
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
                    imp_trace_values = [float(solu) + self.mach_eps] * (len(self.tspan) - ignore)
                else:
                    # If the imposed trace depends on the value of other species, then we replace species and parameter
                    # values to get the imposed trace
                    solu = solu.subs(params_ready)

                    # After replacing parameter for its values, then we get the species in the equation and pass
                    # their dynamic solution
                    variables = [atom for atom in solu.atoms(sympy.Symbol)]
                    f = sympy.lambdify(variables, solu, modules=dict(sqrt=numpy.lib.scimath.sqrt))
                    args = [y[str(l)][ignore:] for l in variables]  # arguments to put in the lambdify function
                    imp_trace_values = f(*args)

                if any(isinstance(n, complex) for n in imp_trace_values):
                    if verbose:
                        print("solution {0} from equation {1} is complex".format(idx, sp_idx))
                    continue
                elif any(n < 0 for n in imp_trace_values):
                    if verbose:
                        print("solution {0} from equation {1} is negative".format(idx, sp_idx))
                    continue
                diff_trace_ode = abs(numpy.log10(imp_trace_values) - numpy.log10(y['__s%d' % sp_idx][ignore:]))
                if max(diff_trace_ode) < distance_imposed:
                    distance_imposed = max(diff_trace_ode)

                if plot:
                    self.plot_imposed_trace(y=y, tspan=self.tspan, imp_trace_values=imp_trace_values,
                                            sp_idx=sp_idx, diff_trace_ode=diff_trace_ode, ignore=ignore, epsilon=epsilon)

            if distance_imposed < epsilon:
                self.passengers.append(sp_idx)

        return self.passengers

    def plot_imposed_trace(self, y, tspan, imp_trace_values, sp_idx, diff_trace_ode, ignore, epsilon):
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
        plt.semilogy(tspan[ignore:], y['__s{0}'.format(sp_idx)][ignore:], label='full')
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
        if self.model.has_synth_deg():
            for i, j in enumerate(self.model.species):
                if str(j) == '__sink()' or str(j) == '__source()' and i in idx:
                    idx.remove(i)

        eqs = {i: self.model.odes[i] for i in idx}
        self.eqs_for_tropicalization = eqs
        return

    def _check_param_values(self, param_values):
        if param_values is not None:
            if type(param_values) is str:
                pars_to_check = hf.read_pars(param_values)
            else:
                pars_to_check = param_values
            # accept vector of parameter values as an argument
            if len(pars_to_check) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            # convert model parameters into dictionary
            pars_checked = dict((p.name, pars_to_check[i]) for i, p in enumerate(self.model.parameters))
        else:
            # create parameter vector from the values in the model
            pars_checked = dict((p.name, p.value) for i, p in enumerate(self.model.parameters))
        return pars_checked

    def _signature(self, y, eqs_for_analysis, pars_ready, diff_par=1):
        all_signatures = {}

        if self.type_sign == 'production':
            mon_type = 'products'
            mon_sign = 1
        elif self.type_sign == 'consumption':
            mon_type = 'reactants'
            mon_sign = -1
        else:
            raise Exception("type sign must be 'production' or 'consumption'")

        for sp in eqs_for_analysis:

            # reaction terms
            monomials = []

            for term in self.model.reactions_bidirectional:
                if sp in term[mon_type] and term['reversible'] is True:
                    monomials.append(mon_sign * term['rate'])
                elif sp in term[mon_type] and term['reversible'] is False:
                    monomials.append(mon_sign * term['rate'])
            # Dictionary whose keys are the symbolic monomials and the values are the simulation results
            mons_dict = {}
            for mon_p in monomials:
                mon_p_values = mon_p
                # TODO Figure out a way that doesnt require an if statement here
                if mon_p_values == 0:
                    mons_dict[mon_p] = [0] * self.tspan
                else:
                    var_prod = [atom for atom in mon_p_values.atoms(sympy.Symbol)]  # Variables of monomial
                    arg_prod = [0]*len(var_prod)
                    for idx, va in enumerate(var_prod):
                        if str(va).startswith('__'):
                            arg_prod[idx] = numpy.maximum(self.mach_eps, y[str(va)])
                        else:
                            arg_prod[idx] = pars_ready[str(va)]
                    # arg_prod = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_prod]
                    f_prod = sympy.lambdify(var_prod, mon_p_values)

                    prod_values = f_prod(*arg_prod)
                    mons_dict[mon_p] = prod_values

            # Dataframe whose rownames are the monomials and the columns contain their values at each time point
            mons_df = pd.DataFrame(mons_dict).T

            signature_species = mons_df.apply(self.choose_max2, axis=0, reduce=True,
                                              args=(diff_par, self.all_comb[sp], self.type_sign))
            all_signatures[sp] = list(signature_species)
        return all_signatures

    def signal_signature(self, param_values, diff_par=1, sp_to_visualize=None):
        pars_ready = self._check_param_values(param_values)
        y = self.sim.run(param_values=pars_ready).dataframe
        all_signatures = self._signature(y, self.eqs_for_tropicalization, pars_ready, diff_par)

        if sp_to_visualize:
            self.visualization2(y, all_signatures, pars_ready, sp_to_visualize)

        return all_signatures

    def qssa_signal_signature(self, param_values, diff_par=1, epsilon=1, ignore=1, plot_imposed_trace=False,
                              sp_to_visualize=None):
        pars_ready = self._check_param_values(param_values)
        y = self.sim.run(param_values=pars_ready).dataframe

        self.find_passengers(y, pars_ready, epsilon, ignore=ignore, plot=plot_imposed_trace)
        self.equations_to_tropicalize()
        self.set_combinations_sm()

        all_signatures = self._signature(y, self.eqs_for_tropicalization, pars_ready, diff_par)

        if sp_to_visualize:
            self.visualization2(y, all_signatures, pars_ready, sp_to_visualize)

        return all_signatures

    def set_combinations_sm(self, max_comb=None, create_sm=False):
        if self.type_sign == 'production':
            mon_type = 'products'
            mon_sign = 1
        elif self.type_sign == 'consumption':
            mon_type = 'reactants'
            mon_sign = -1
        else:
            raise Exception("type sign must be 'production' or 'consumption'")

        for sp in self.eqs_for_tropicalization:
            # reaction terms
            monomials = []

            for term in self.model.reactions_bidirectional:
                if sp in term[mon_type] and term['reversible'] is True:
                    monomials.append(mon_sign * term['rate'])
                elif sp in term[mon_type] and term['reversible'] is False:
                    monomials.append(mon_sign * term['rate'])

            if max_comb:
                combs = max_comb
            else:
                combs = len(monomials) + 1

            mon_comb = OrderedDict()
            prod_idx = 0

            for L in range(1, combs):
                prod_comb_names = {}
                for subset in itertools.combinations(monomials, L):
                    prod_comb_names['M{0}{1}'.format(L, prod_idx)] = subset
                    prod_idx += 1
                mon_comb[L] = prod_comb_names
            self.all_comb[sp] = mon_comb
            #
            # merged_mon_comb = self.merge_dicts(*mon_comb.values())
            # merged_mon_comb.update({'ND': 'N'})
            # # Substitution matrix
            # len_ND = len(max(merged_mon_comb.values(), key=len)) + 1
            # sm = numpy.zeros((len(merged_mon_comb.keys()), len(merged_mon_comb.keys())))
            # for i, a in enumerate(merged_mon_comb):
            #     for j, b in enumerate(merged_mon_comb):
            #         if a == 'ND' and b == 'ND':
            #             sm[i, j] = 0
            #         elif a == 'ND':
            #             sm[i, j] = 2 * len_ND - len(merged_mon_comb[b])
            #         elif b == 'ND':
            #             sm[i, j] = 2 * len_ND - len(merged_mon_comb[a])
            #         else:
            #             sm[i, j] = self.sub_value(merged_mon_comb[a], merged_mon_comb[b])
            #             # max(len(self.all_comb[sp][a]), len(self.all_comb[sp][b])) - len(
            #             #             set(self.all_comb[sp][a]).intersection(self.all_comb[sp][b]))
            #
            # if create_sm:
            #     sm_df = pd.DataFrame(data=sm, index=merged_mon_comb.keys(), columns=merged_mon_comb.keys())
            #     sm_df.to_csv('/home/oscar/Documents/tropical_earm/subs_matrix_consumption/sm_{0}.{1}'.format(sp, 'csv'))

    @staticmethod
    def sub_value(a, b):
        value = 2 * len(max(a, b, key=len)) - len(min(a, b, key=len)) - len(set(a).intersection(b))
        return value

    def visualization2(self, y, all_signatures, param_values, sp_to_vis=None):
        if sp_to_vis:
            species_ready = list(set(sp_to_vis).intersection(all_signatures.keys()))
        else:
            raise Exception('list of driver species must be defined')

        if not species_ready:
            raise Exception('None of the input species is a driver')

        for sp in species_ready:

            # Setting up figure
            plt.figure(1)
            plt.subplot(313)

            mon_val = OrderedDict()
            signature = all_signatures[sp]

            if not signature:
                continue

            merged_mon_comb = self.merge_dicts(*self.all_comb[sp].values())
            merged_mon_comb.update({'ND': 'N'})

            for idx, mon in enumerate(list(set(signature))):
                mon_val[merged_mon_comb[mon]] = idx

            mon_rep = [0] * len(signature)
            for i, m in enumerate(signature):
                mon_rep[i] = mon_val[merged_mon_comb[m]]
            # mon_rep = [mon_val[self.all_comb[sp][m]] for m in signature]

            y_pos = numpy.arange(len(mon_val.keys()))
            plt.scatter(self.tspan, mon_rep)
            plt.yticks(y_pos, mon_val.keys())
            plt.ylabel('Monomials', fontsize=16)
            plt.xlabel('Time(s)', fontsize=16)
            plt.xlim(0, self.tspan[-1])
            plt.ylim(0, max(y_pos))

            plt.subplot(312)
            for name in self.model.odes[sp].as_coefficients_dict():
                mon = name
                var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
                arg_f1 = [0] * len(var_to_study)
                for idx, va in enumerate(var_to_study):
                    if str(va).startswith('__'):
                        arg_f1[idx] = numpy.maximum(self.mach_eps, y[str(va)])
                    else:
                        arg_f1[idx] = param_values[str(va)]

                f1 = sympy.lambdify(var_to_study, mon)
                mon_values = f1(*arg_f1)
                mon_name = str(name).partition('__')[2]
                plt.plot(self.tspan, mon_values, label=mon_name)
            plt.ylabel('Rate(m/sec)', fontsize=16)
            plt.legend(bbox_to_anchor=(-0.1, 0.85), loc='upper right', ncol=3)

            plt.subplot(311)
            plt.plot(self.tspan, y['__s%d' % sp], label=hf.parse_name(self.model.species[sp]))
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


def run_tropical(model, tspan, parameters=None, diff_par=1, find_passengers_by='imp_nodes', type_sign='production', max_comb=None, sp_visualize=None,
                 verbose=False):
    """

    :param model: PySB model of a biological system
    :param tspan: Time of the simulation
    :param parameters: Parameter values of the PySB model
    :param diff_par:
    :param find_passengers_by:
    :param type_sign:
    :param max_comb:
    :param sp_visualize: Species to visualize
    :param verbose:
    :return: The DynSign signatures of all non-passenger species
    """
    tr = Tropical(model)
    signatures = tr.tropicalize(tspan=tspan, param_values=parameters, diff_par=diff_par, type_sign=type_sign,
                                find_passengers_by=find_passengers_by,max_comb=max_comb,
                                sp_to_visualize=sp_visualize, verbose=verbose)
    return signatures
    # return tr.get_species_signatures()


def run_tropical_multiprocessing(model, tspan, parameters=None, diff_par=1, find_passengers_by='imp_nodes', type_sign='production',
                                 to_data_frame=False, dir_path=None, verbose=False):
    """

    :param model:
    :param tspan:
    :param parameters:
    :param diff_par:
    :param find_passengers_by:
    :param type_sign:
    :param to_data_frame:
    :param dir_path:
    :param verbose:
    :return:
    """
    tr = Tropical(model)
    dynamic_signatures_partial = functools.partial(dynamic_signatures, tropical_object=tr, tspan=tspan,
                                                   type_sign=type_sign, diff_par=diff_par, find_passengers_by=find_passengers_by ,verbose=verbose)
    p = Pool(cpu_count() - 1)
    all_drivers = p.map_async(dynamic_signatures_partial, parameters)
    while not all_drivers.ready():
        remaining = all_drivers._number_left
        print ("Waiting for", remaining, "tasks to complete...")
        time.sleep(5)

    if to_data_frame:
        hf.sps_signature_to_df(all_drivers, dir_path, tspan, parameters)
    return all_drivers
