from __future__ import print_function
from __future__ import division
import functools
import itertools
import time
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
import numpy
import sympy
from sympy import default_sort_key
import helper_functions as hf
from pysb.simulator import ScipyOdeSimulator, SimulatorException
import matplotlib.pyplot as plt
from pysb import Parameter
import pandas as pd

def dynamic_signatures(param_values, tropical_object, tspan=None, type_sign='production', diff_par=1, ignore=1,
                       epsilon=1, find_passengers_by='imp_nodes', pre_equilibrate=False, max_comb=None, sp_to_visualize=None,
                       plot_imposed_trace=False, verbose=False):
    """
    This is a global function that takes the class object as a parameter to compute the dynamical signature.
    This global function is necessary to use the multiprocessing module.

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
                                                               plot_imposed_trace, sp_to_visualize, verbose=verbose)

    elif find_passengers_by == 'imp_nodes':
        all_signatures = tropical_object.signal_signature(param_values, diff_par=diff_par, pre_equilibrate=pre_equilibrate,
                                                          sp_to_visualize=sp_to_visualize)
    else:
        raise Exception('A valid way to get the signatures must be provided')
    return all_signatures


class Tropical:
    mach_eps = 1e-11 # numpy.finfo(float).eps

    def __init__(self, model):
        """
        Constructor of tropical function
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
        self.value_to_add = None
        self.sign = None
        self.ascending = None
        self.mon_type = None

    def tropicalize(self, tspan=None, param_values=None, type_sign='production', diff_par=1, ignore=1, epsilon=1,
                    find_passengers_by='imp_nodes', pre_equilibrate=False, max_comb=None, sp_to_visualize=None, plot_imposed_trace=False,
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
        :return: a dictionary whose keys are the important species and the values are the tropical signatures
        """
        all_signatures = dynamic_signatures(param_values, self, tspan=tspan, type_sign=type_sign, diff_par=diff_par,
                                            ignore=ignore, epsilon=epsilon, find_passengers_by=find_passengers_by,
                                            max_comb=max_comb, pre_equilibrate=pre_equilibrate, sp_to_visualize=sp_to_visualize,
                                            plot_imposed_trace=plot_imposed_trace, verbose=verbose)

        return all_signatures

    def setup_tropical(self, tspan, type_sign, find_passengers_by, max_comb, verbose):
        """
        A function to set up the parameters of the Tropical class

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

        if type_sign == 'production':
            self.type_sign = type_sign
            self.value_to_add = 1e-100
            self.sign = 1
            self.ascending = True
            self.mon_type = 'products'

        elif type_sign == 'consumption':
            self.type_sign = type_sign
            self.value_to_add = -1e-100
            self.sign = -1
            self.ascending = False
            self.mon_type = 'reactants'
        else:
            raise ValueError('Wrong type_sign value')

        if verbose:
            print('setting up the simulator class')
        self.sim = ScipyOdeSimulator(self.model, self.tspan)

        if verbose:
            print('setting up the important nodes')
        if find_passengers_by == 'imp_nodes':
            self.find_nonimportant_nodes()
            self.equations_to_tropicalize()
            if not self.all_comb:
                self.set_combinations_sm(max_comb=max_comb)

        self.is_setup = True
        return

    def find_nonimportant_nodes(self):
        """
        This function looks a the bidirectional reactions and finds the nodes that only have one incoming and outgoing
        reaction (edge)
        :return: a list of non-important nodes
        """
        # gets the reactant and product species in the reactions
        rcts_sp = sum([i['reactants'] for i in self.model.reactions_bidirectional], ())
        pdts_sp = sum([i['products'] for i in self.model.reactions_bidirectional], ())
        # find the reactants and products that are only used once
        non_imp_rcts = set([x for x in range(len(self.model.species)) if rcts_sp.count(x) < 2])
        non_imp_pdts = set([x for x in range(len(self.model.species)) if pdts_sp.count(x) < 2])
        non_imp_nodes = set.intersection(non_imp_pdts, non_imp_rcts)
        self.passengers = non_imp_nodes
        return self.passengers

    def find_passengers_qssa(self, y, params_ready, epsilon=1, ignore=1, plot=False, verbose=False):
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

        odes = []
        for ode in self.model.odes:
            for symbol in ode.atoms():
                if isinstance(symbol, Parameter):
                    ode = ode.subs(symbol, sympy.Symbol(symbol.name))
            odes.append(ode)

        # Loop through all equations
        for i, eq in enumerate(odes):
            # Solve equation of imposed trace. It can have more than one solution (Quadratic solutions)
            sol = sympy.solve(eq, sympy.Symbol('__s%d' % i))
            sp_imposed_trace.append(sol)
        for sp_idx, trace_soln in enumerate(sp_imposed_trace):
            distance_imposed = 999
            for idx, solu in enumerate(trace_soln):
                # Check if solution is time independent
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
                    args = [y[str(l)].iloc[ignore:] for l in variables]  # arguments to put in the lambdify function
                    imp_trace_values = f(*args) + self.mach_eps
                if any(isinstance(n, complex) for n in imp_trace_values):
                    if verbose:
                        print("solution {0} from equation {1} is complex".format(idx, sp_idx))
                    continue
                elif any(n < 0 for n in imp_trace_values):
                    if verbose:
                        print("solution {0} from equation {1} is negative".format(idx, sp_idx))
                    continue
                diff_trace_ode = abs(numpy.log10(imp_trace_values) - numpy.log10(y['__s%d' % sp_idx].iloc[ignore:] +
                                                                                 self.mach_eps))
                if max(diff_trace_ode) < distance_imposed:
                    distance_imposed = max(diff_trace_ode)

                if plot:
                    self.plot_imposed_trace(y=y, tspan=self.tspan, imp_trace_values=imp_trace_values,
                                            sp_idx=sp_idx, diff_trace_ode=diff_trace_ode, ignore=ignore,
                                            epsilon=epsilon)

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
        plt.semilogy(tspan[ignore:], imp_trace_values, 'r--', linewidth=5, label='imposed')
        plt.semilogy(tspan[ignore:], y['__s{0}'.format(sp_idx)].iloc[ignore:]+self.mach_eps, nonposy='clip', label='full')
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
        # removing source and sink species
        if self.model.has_synth_deg():
            for i, j in enumerate(self.model.species):
                if str(j) == '__sink()' or str(j) == '__source()' and i in idx:
                    idx.remove(i)

        eqs = {i: self.model.odes[i] for i in idx}
        self.eqs_for_tropicalization = eqs
        return

    def get_monomials_idx(self, array):
        if self.type_sign == 'production':
            monomials_idx = numpy.where(array > 0)[0]
            return monomials_idx
        else:
            monomials_idx = numpy.where(array < 0)[0]
            return monomials_idx

    def choose_max_numpy(self, array, mon_names, diff_par, mon_comb):
        """

        :param array:
        :param mon_names:
        :param diff_par:
        :param mon_comb:
        :param type_sign:
        :return:
        """
        monomials_idx = self.get_monomials_idx(array)

        if len(monomials_idx) == 0:
            largest_prod = mon_comb.values()[-1].keys()[0] + 1
        else:
            monomials_values = {mon_names[idx]:
                                numpy.log10(numpy.abs(array[idx])) for idx in monomials_idx}
            max_val = numpy.amax(monomials_values.values())
            rr_monomials = [n for n, i in monomials_values.items() if i > (max_val - diff_par) and max_val > -5]

            if not rr_monomials or len(rr_monomials) == mon_comb.keys()[-1]:
                largest_prod = mon_comb.values()[-1].keys()[0]
            else:
                rr_monomials.sort(key=default_sort_key)
                rr_monomials = tuple(rr_monomials)
                largest_prod = mon_comb[len(rr_monomials)].keys()[mon_comb[len(rr_monomials)].values().index(rr_monomials)]

        return largest_prod

    def _signature(self, y, pars_ready, diff_par=1):
        """

        :param y:
        :param pars_ready:
        :param diff_par:
        :return:
        """
        all_signatures = {}

        for sp in self.eqs_for_tropicalization:

            # reaction terms
            monomials = []

            for term in self.model.reactions_bidirectional:
                if sp in term[self.mon_type]:
                    monomials.append(self.sign * term['rate'])

            # Dictionary whose keys are the symbolic monomials and the values are the simulation results
            mons_dict = {}
            for mon_p in monomials:
                mon_p_values = mon_p
                # TODO Figure out a way that doesnt require an if statement here
                if mon_p_values == 0:
                    mons_dict[mon_p] = [0] * self.tspan
                else:
                    var_prod = [atom for atom in mon_p_values.atoms(sympy.Symbol)]  # Variables of monomial
                    arg_prod = [0] * len(var_prod)
                    for idx, va in enumerate(var_prod):
                        if str(va).startswith('__'):
                            arg_prod[idx] = numpy.maximum(self.mach_eps, y[str(va)])
                        else:
                            arg_prod[idx] = pars_ready[va.name]
                    # arg_prod = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_prod]
                    f_prod = sympy.lambdify(var_prod, mon_p_values)

                    prod_values = f_prod(*arg_prod)
                    mons_dict[mon_p] = prod_values
            mons_names = {}
            mons_array = numpy.zeros((len(mons_dict.keys()), len(self.tspan)))
            for idx, name in enumerate(mons_dict.keys()):
                mons_array[idx] = mons_dict[name]
                mons_names[idx] = name

            signature_species = numpy.apply_along_axis(self.choose_max_numpy, 0, mons_array,
                                                       *(mons_names, diff_par, self.all_comb[sp]))

            all_signatures[sp] = signature_species
        return all_signatures

    def signal_signature(self, param_values, diff_par=1, pre_equilibrate=False, sp_to_visualize=None):
        pars_ready = hf.check_param_values(self.model, param_values)
        if pre_equilibrate:
            eq_ic = hf.pre_equilibration(self.model, self.tspan, ligand_par_name='L_0', ligand_idx=0,
                                         ligand_value=0, parameters=pars_ready)[1]
            y = self.sim.run(initials=eq_ic, param_values=pars_ready).all
        else:
            y = self.sim.run(param_values=pars_ready).all
        all_signatures = self._signature(y, pars_ready, diff_par)

        if sp_to_visualize:
            self.visualization(y, all_signatures, pars_ready, sp_to_visualize)
        return all_signatures, y

    def qssa_signal_signature(self, param_values, diff_par=1, epsilon=1, ignore=1, plot_imposed_trace=False,
                              sp_to_visualize=None, verbose=False):
        pars_ready = hf.check_param_values(self.model, param_values)
        y = self.sim.run(param_values=pars_ready).dataframe

        self.find_passengers_qssa(y, pars_ready, epsilon, ignore=ignore, plot=plot_imposed_trace, verbose=verbose)
        self.equations_to_tropicalize()
        self.set_combinations_sm()

        all_signatures = self._signature(y, pars_ready, diff_par)

        if sp_to_visualize:
            self.visualization(y, all_signatures, pars_ready, sp_to_visualize)

        return all_signatures

    @staticmethod
    def get_global_signature(all_signatures, tspan):
        """

        :param all_signatures: All the signatures
        :param tspan: time span
        :return:
        """
        if isinstance(all_signatures, dict):
            all_signatures = [all_signatures]
        global_signature = [0]*len(all_signatures)
        global_labels = {}
        for idx, ic_signatures in enumerate(all_signatures):
            global_signature_ic = [0]*len(tspan)
            for j in range(len(tspan)):
                global_time_point = [sp_sg[j] for sp_sg in ic_signatures.values()]
                global_time_tuple = tuple(global_time_point)
                if global_time_tuple not in global_labels.keys():
                    global_labels[global_time_tuple] = 'GS{0}{1}'.format(idx, j)
                global_signature_ic[j] = global_labels[global_time_tuple]
            global_signature[idx] = global_signature_ic
        return global_signature

    def set_combinations_sm(self, max_comb=None, create_sm=True):
        """

        :param max_comb: int, the maximum number of combinations
        :param create_sm: boolean, to create a sustition matrix to use in the clustering analysis
        :return:
        """

        for sp in self.eqs_for_tropicalization:
            # reaction terms
            monomials = []

            for term in self.model.reactions_bidirectional:
                if sp in term[self.mon_type]:
                    monomials.append(self.sign * term['rate'])
                # elif sp in term[self.mon_type] and term['reversible'] is False:
                #     monomials.append(self.sign * term['rate'])

            if max_comb:
                combs = max_comb
            else:
                combs = len(monomials) + 1

            mon_comb = OrderedDict()
            comb_counter = 0
            for L in range(1, combs):

                prod_comb_names = {}

                for subset in itertools.combinations(monomials, L):
                    subset = list(subset)
                    subset.sort(key=default_sort_key)
                    subset = tuple(subset)
                    rr_label = comb_counter
                    prod_comb_names[rr_label] = subset
                    comb_counter += 1

                mon_comb[L] = prod_comb_names

            self.all_comb[sp] = mon_comb

            # merged_mon_comb = hf.merge_dicts(*mon_comb.values())
            # # Substitution matrix
            # len_ND = len(max(merged_mon_comb.values(), key=len)) + 1
            # sm = numpy.zeros((len(merged_mon_comb.keys()), len(merged_mon_comb.keys())))
            # for i, a in enumerate(merged_mon_comb):
            #     for j, b in enumerate(merged_mon_comb):
            #         if a == 'NoDominants' and b == 'NoDominants':
            #             sm[i, j] = 0
            #         elif a == 'NoDominants':
            #             sm[i, j] = 2 * len_ND - len(merged_mon_comb[b])
            #         elif b == 'NoDominants':
            #             sm[i, j] = 2 * len_ND - len(merged_mon_comb[a])
            #         else:
            #             sm[i, j] = self.sub_value(merged_mon_comb[a], merged_mon_comb[b])
            #             # max(len(self.all_comb[sp][a]), len(self.all_comb[sp][b])) - len(
            #             #             set(self.all_comb[sp][a]).intersection(self.all_comb[sp][b]))
            #
            # if create_sm:
            #     sm_df = pd.DataFrame(data=sm, index=merged_mon_comb.keys(), columns=merged_mon_comb.keys())
            #     sm_df.to_csv('/Users/dionisio/Documents/sm_{0}.{1}'.format(sp, 'csv'))
        return

    @staticmethod
    def sub_value(a, b):
        value = 2 * len(max(a, b, key=len)) - len(min(a, b, key=len)) - len(set(a).intersection(b))
        return value

    def visualization(self, y, all_signatures, param_values, sp_to_vis=None):

        species_ready = list(set(sp_to_vis).intersection(all_signatures.keys()))
        if not species_ready:
            raise Exception('None of the input species is a driver')

        for sp in species_ready:

            # Setting up figure
            plt.figure(1)
            ax3 = plt.subplot(313)

            mon_val = OrderedDict()
            signature = all_signatures[sp]

            # if not signature:
            #     continue

            # merged_mon_comb = hf.merge_dicts(*self.all_comb[sp].values())
            # merged_mon_comb.update({'ND': 'N'})

            for idx, mon in enumerate(list(set(signature))):
                mon_val[mon] = idx

            mon_rep = [0] * len(signature)
            for i, m in enumerate(signature):
                mon_rep[i] = mon_val[m]
            # mon_rep = [mon_val[self.all_comb[sp][m]] for m in signature]

            y_pos = numpy.arange(len(mon_val.keys()))
            plt.scatter(self.tspan, mon_rep)
            plt.yticks(y_pos, mon_val.keys())
            plt.ylabel('Monomials', fontsize=14)
            plt.xlabel('Time(s)', fontsize=16)
            # plt.xlim(0, self.tspan[-1])
            plt.ylim(0, max(y_pos))

            ax2 = plt.subplot(312, sharex=ax3)
            for rr in self.all_comb[sp][1].values():
                mon = rr[0].as_coefficients_dict().keys()[0]
                var_to_study = [atom for atom in mon.atoms(sympy.Symbol)]
                arg_f1 = [0] * len(var_to_study)
                for idx, va in enumerate(var_to_study):
                    if str(va).startswith('__'):
                        arg_f1[idx] = numpy.maximum(self.mach_eps, y[str(va)])
                    else:
                        arg_f1[idx] = param_values[va.name]
                f1 = sympy.lambdify(var_to_study, mon)
                mon_values = f1(*arg_f1)
                mon_name = str(rr[0]).partition('__')[2]
                plt.plot(self.tspan, mon_values, label=mon_name)
            plt.ylabel('Rate(m/sec)', fontsize=14)
            plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=2)
            plt.setp(ax2.get_xticklabels(), visible=False)

            ax1 = plt.subplot(311, sharex=ax3)
            plt.plot(self.tspan, y['__s%d' % sp], label=hf.parse_name(self.model.species[sp]))
            plt.ylabel('Molecules', fontsize=14)
            plt.legend(bbox_to_anchor=(-0.15, 0.85), loc='upper right', ncol=1)
            plt.setp(ax1.get_xticklabels(), visible=False)

            plt.suptitle('Tropicalization' + ' ' + str(self.model.species[sp]))
            plt.show()
            plt.savefig('s%d' % sp + '.png', bbox_inches='tight', dpi=400)
            # plt.clf()

    def get_passenger(self):
        """

        :return: Passenger species of the systems
        """
        return self.passengers

    def get_comb_dict(self):
        """

        :return: Combination of monomials for each species
        """
        return self.all_comb


def run_tropical(model, tspan, parameters=None, global_signature=False, diff_par=1, find_passengers_by='imp_nodes',
                 type_sign='production', pre_equilibrate=False, max_comb=None, sp_visualize=None, plot_imposed_trace=False, verbose=False):
    """

    :param plot_imposed_trace:
    :param model: PySB model of a biological system
    :param tspan: Time of the simulation
    :param parameters: Parameter values of the PySB model
    :param diff_par:
    :param find_passengers_by:
    :param type_sign:
    :param max_comb:
    :param sp_visualize: Species to visualize
    :param verbose:
    :return: A dictionary whose keys are the important species and the values are the tropical signatures
    """
    tr = Tropical(model)
    signatures = tr.tropicalize(tspan=tspan, param_values=parameters, diff_par=diff_par, type_sign=type_sign,
                                find_passengers_by=find_passengers_by, max_comb=max_comb, pre_equilibrate=pre_equilibrate,
                                sp_to_visualize=sp_visualize, plot_imposed_trace=plot_imposed_trace, verbose=verbose)
    if global_signature:
        whole_signature = tr.get_global_signature(signatures[0], tspan)
        return signatures[0], whole_signature
    else:
        return signatures[0]
    # return tr.get_species_signatures()


def run_tropical_multiprocessing(model, tspan, parameters=None, global_signature=False, diff_par=1, find_passengers_by='imp_nodes',
                                 type_sign='production', pre_equilibrate=False, to_data_frame=False, dir_path=None, verbose=False):
    """

    :param model: A PySB model
    :param tspan: time span of the simulation
    :param parameters: Parameters of the model
    :param diff_par: Magnitude difference that defines that a reaction is dominant over others
    :param find_passengers_by: str, it can be 'imp_nodes' or 'qssa' is the way to find the passenger species
    :param type_sign: Type of signature. It can be 'consumption' or 'production'
    :param to_data_frame: boolean, convert array into data frame
    :param dir_path: Path to save the data frames
    :param verbose: Verbose
    :return: A list of all the signatures and species for each parameter set.
    """
    tr = Tropical(model)
    dynamic_signatures_partial = functools.partial(dynamic_signatures, tropical_object=tr, tspan=tspan,
                                                   type_sign=type_sign, diff_par=diff_par, pre_equilibrate=pre_equilibrate,
                                                   verbose=verbose, find_passengers_by=find_passengers_by)
    p = Pool(cpu_count() - 1)
    all_drivers = p.map_async(dynamic_signatures_partial, parameters)
    while not all_drivers.ready():
        remaining = all_drivers._number_left
        print("Waiting for", remaining, "tasks to complete...")
        time.sleep(5)

    all_drivers = all_drivers.get()
    signatures = [0]*len(parameters)
    simulations = [0]*len(parameters)
    for i, j in enumerate(all_drivers):
        signatures[i] = j[0]
        simulations[i] = j[1]
    simulations = numpy.asarray(simulations)

    if to_data_frame:
        hf.sps_signature_to_df(signatures=signatures, dir_path=dir_path, col_index=tspan)
    if global_signature:
        whole_signature = tr.get_global_signature(signatures, tspan)
        hf.sps_signature_to_df(signatures=whole_signature, dir_path=dir_path, global_signature=True, col_index=tspan)
    numpy.save(dir_path + '/simulations', simulations)

    return signatures, simulations
