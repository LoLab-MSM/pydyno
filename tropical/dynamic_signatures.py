import tropical.helper_functions as hf
import h5py
from pysb.simulator import SimulationResult, ScipyOdeSimulator, CupSodaSimulator
import collections
import numpy
import operator
import sympy
import math
from pathos.multiprocessing import ProcessingPool as Pool
import itertools


class Tropical(object):
    mach_eps = 1e-11

    def __init__(self, model):
        self.all_comb = {}
        self.model = model
        self.par_name_idx = {j.name: i for i, j in enumerate(self.model.parameters)}
        self.trajectories = []
        self._is_setup = False
        self.passengers = []
        self.eqs_for_tropicalization = {}
        self.tspan = []

    def setup_tropical(self, sim_or_params, tspan=None, simulator=None, passengers_by='imp_nodes'):
        self.get_simulations(sim_or_params=sim_or_params, tspan=tspan, simulator=simulator)
        self.get_passengers(by=passengers_by)
        self.equations_to_tropicalize()
        self.set_combinations_sm()
        self._is_setup = True
        return

    def get_simulations(self, sim_or_params, tspan=None, simulator=None):
        if isinstance(sim_or_params, str):
            if h5py.is_hdf5(sim_or_params):
                sim = SimulationResult.load(sim_or_params)
                tspan = sim.tout[0]
            else:
                raise TypeError('File format not supported')
        elif isinstance(sim_or_params, SimulationResult):
            sim = sim_or_params
            tspan = sim.tout[0]
        elif isinstance(sim_or_params, collections.Iterable) or sim_or_params is None:
            # TODO check parameter length
            parameters = sim_or_params
            if simulator == 'scipy':
                sim = ScipyOdeSimulator(model=self.model, tspan=tspan, param_values=parameters).run()
            elif simulator == 'cupsoda':
                sim = CupSodaSimulator(model=self.model, tspan=tspan, param_values=parameters).run()
            else:
                raise ValueError(' A valid simulator must be provided')
        else:
            raise TypeError('format not supported')

        self.trajectories = sim.all
        self.parameters = sim.param_values
        self.tspan = tspan
        return

    def get_passengers(self, by='imp_nodes'):
        if by == 'imp_nodes':
            self.passengers = hf.find_nonimportant_nodes(self.model)
        return

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

    @staticmethod
    def choose_max_pos_neg(array, mon_names, diff_par, mon_comb):
        """

        :param array:
        :param mon_names:
        :param diff_par:
        :param mon_comb:
        :return:
        """
        mons_pos_neg = [numpy.where(array > 0)[0], numpy.where(array < 0)[0]]
        # print (mons_pos_neg)
        signs = [1, -1]
        ascending_order = [False, True]
        mons_types = ['products', 'reactants']

        pos_neg_largest = [0] * 2
        range_0_1 = range(2)
        for ii, mon_type, mons_idx, sign, ascending in zip(range_0_1, mons_types, mons_pos_neg, signs, ascending_order):
            largest_prod = 'NoDoms'
            mon_names_ready = [mon_names.keys()[mon_names.values().index(i)] for i in mons_idx]
            # print (array, mon_names_ready)
            # if mon_type == 'reactants':
            #     print(mon_names_ready, mon_names, mons_pos_neg)
            mon_comb_type = mon_comb[mon_type]

            for comb in sorted(mon_comb_type.keys()):
                # comb is an integer that represents the number of monomials in a combination
                if len(mon_comb_type[comb].keys()) == 1:
                    largest_prod = mon_comb_type[comb].keys()[0]
                    break

                monomials_values = {}
                for idx in mon_comb_type[comb].keys():
                    value = 0
                    for j in mon_comb_type[comb][idx]:
                        if j not in mon_names_ready:
                            value += sign * 1e-100 # value_to_add
                        else:
                            value += array[mon_names[j]]
                    monomials_values[idx] = value
                foo2 = sorted(monomials_values.items(), key=operator.itemgetter(1), reverse=ascending)
                # foo2 = pd.Series(monomials_values).sort_values(ascending=ascending)
                comb_largest = mon_comb_type[comb][foo2[0][0]]
                for cm in foo2:
                    # Compares the largest combination of monomials to other combinations whose monomials that are not
                    # present in comb_largest
                    if len(set(comb_largest) - set(mon_comb_type[comb][cm[0]])) == len(comb_largest):
                        value_prod_largest = math.log10(sign * foo2[0][1])
                        if abs(value_prod_largest - math.log10(sign * cm[1])) > diff_par and value_prod_largest > -5:
                            largest_prod = foo2[0][0]
                            break
                if largest_prod != 'NoDoms':
                    break
            pos_neg_largest[ii] = largest_prod
            # print(mon_type, mon_names_ready, mon_comb_type, largest_prod)
        # print (pos_neg_largest, mon_names)
        return pos_neg_largest

    def set_combinations_sm(self, max_comb=None, create_sm=False):

        for sp in self.eqs_for_tropicalization:
            # reaction terms
            pos_neg_combs = {}
            parts_reaction = ['products', 'reactants']
            parts_rev = ['reactants', 'products']
            signs = [1, -1]

            # We get the reaction rates from the bidirectional reactions in order to have reversible reactions
            # as one 'monomial'. This is helpful for visualization and other (I should think more about this)
            for mon_type, mon_sign, rev_parts in zip(parts_reaction, signs, parts_rev):
                monomials = []

                for term in self.model.reactions_bidirectional:
                    if sp in term[mon_type]:
                        # Add zero to monomials in cases like autocatalytic reactions where a species
                        # shows up both in reactants and products, and we are looking for the reactions that use a sp
                        # but the reaction produces the species overall
                        if sp in term[rev_parts]:
                            count_reac = term['reactants'].count(sp)
                            count_pro = term['products'].count(sp)
                            mon_zero = mon_sign
                            if mon_type == 'reactants':
                                if count_pro > count_reac:
                                    mon_zero = 0
                            else:
                                if count_pro < count_reac:
                                    mon_zero = 0
                            monomials.append(mon_zero * term['rate'])
                        else:
                            monomials.append(mon_sign * term['rate'])

                # remove zeros from reactions in which the species shows up both in reactants and products
                monomials = [value for value in monomials if value != 0]
                if max_comb:
                    combs = max_comb
                else:
                    combs = len(monomials) + 1

                mon_comb = {}
                prod_idx = 0

                for L in range(1, combs):
                    prod_comb_names = {}
                    if L == combs - 1:
                        prod_comb_names['NoDoms'] = 'No_Dominants'
                    else:
                        for subset in itertools.combinations(monomials, L):
                            prod_comb_names['M{0}{1}'.format(L, prod_idx)] = subset
                            prod_idx += 1
                    mon_comb[L] = prod_comb_names
                pos_neg_combs[mon_type] = mon_comb
            self.all_comb[sp] = pos_neg_combs
        return

    def signature(self, y, pars_ready, diff_par=1):
        # Dictionary that will contain the signature of each of the species to study
        if not self._is_setup:
            raise Exception('you must setup tropical first')
        all_signatures = {}

        for sp in self.eqs_for_tropicalization:
            # reaction terms for positive and negative monomials
            monomials = []
            for term in self.model.reactions_bidirectional:
                total_rate = 0
                for mon_type, mon_sign in zip(['products', 'reactants'], [1, -1]):
                    if sp in term[mon_type]:
                        count = term[mon_type].count(sp)
                        total_rate = total_rate + (mon_sign * count * term['rate'])
                if total_rate == 0:
                    continue
                monomials.append(total_rate)
            # Dictionary whose keys are the symbolic monomials and the values are the simulation results
            mons_dict = {}
            for mon_p in monomials:
                mon_p_values = mon_p
                # TODO Figure out a way that doesnt require an if statement here
                if mon_p_values == 0:
                    mons_dict[mon_p] = [0] * len(self.tspan)
                else:
                    var_prod = [atom for atom in mon_p_values.atoms(sympy.Symbol)]  # Variables of monomial
                    arg_prod = [0] * len(var_prod)
                    for idx, va in enumerate(var_prod):
                        if str(va).startswith('__'):
                            arg_prod[idx] = numpy.maximum(self.mach_eps, y[str(va)])
                        else:
                            arg_prod[idx] = pars_ready[self.par_name_idx[va.name]]
                    # arg_prod = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_prod]
                    f_prod = sympy.lambdify(var_prod, mon_p_values)

                    prod_values = f_prod(*arg_prod)
                    mons_dict[mon_p] = prod_values
            mons_names = {}
            mons_array = numpy.zeros((len(mons_dict.keys()), len(self.tspan)))
            for idx, name in enumerate(mons_dict.keys()):
                mons_array[idx] = mons_dict[name]
                mons_names[name] = idx
            signature_species = numpy.apply_along_axis(self.choose_max_pos_neg, 0, mons_array,
                                                       *(mons_names, diff_par, self.all_comb[sp]))
            # print (sp, signature_species)
            all_signatures[sp] = list(signature_species)
        return all_signatures


def run_tropical(model, sim_or_params=None, tspan=None, simulator=None, passengers_by='imp_nodes', diff_par=1, cpu_cores=1):
    tro = Tropical(model)
    tro.setup_tropical(sim_or_params, tspan=tspan, simulator=simulator, passengers_by=passengers_by)
    p = Pool(cpu_cores)
    res = p.amap(tro.signature, tro.trajectories, tro.parameters)
    return res.get()