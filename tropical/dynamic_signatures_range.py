import tropical.util as hf
from tropical.visualize_discretization import visualization_sp
import numpy
import sympy
import itertools
from collections import OrderedDict
from future.utils import iteritems, listvalues
import time
from pysb import Parameter
import pandas as pd

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None


class Tropical(object):
    """
    Obtain the dynamic signatures of species from a PySB model

    Parameters
    ----------
    model : pysb.Model
        Model to analyze.
    """
    mach_eps = 1e-11

    def __init__(self, model):

        self.all_comb = {}
        self.model = model
        self.par_name_idx = {j.name: i for i, j in enumerate(self.model.parameters)}
        self._is_setup = False
        self.passengers = []
        self.eqs_for_tropicalization = []
        self.diff_par = None
        self.tspan = None

    def setup_tropical(self, tspan, diff_par=1, passengers_by='imp_nodes', add_observables=False):
        """
        Set up parameters necessary to obtain the dynamic signatures of species signal execution

        Parameters
        ----------
        tspan : vector-like, optional
            Time values over which to do the tropical analysis. The first and last values define
            the time range.
        diff_par : float
            Magnitude difference that defines that a reaction is dominant over others.
        passengers_by : str
            It can be 'qssa' or 'imp_nodes'. It defines the method to use for finding passenger species

        Returns
        -------

        """
        self.diff_par = diff_par
        self.tspan = tspan
        self.equations_to_tropicalize(get_passengers_by=passengers_by, add_observables=add_observables)
        self._is_setup = True
        return

    def equations_to_tropicalize(self, get_passengers_by='imp_nodes', add_observables=False):
        """

        Returns
        -------
        List of dominant species indices

        """
        if get_passengers_by == 'imp_nodes':
            self.passengers = hf.find_nonimportant_nodes(self.model)
        else:
            raise ValueError('method to obtain passengers not supported')

        idx = list(set(range(len(self.model.odes))) - set(self.passengers))
        # removing source and sink species
        if self.model.has_synth_deg():
            for i, j in enumerate(self.model.species):
                if str(j) == '__sink()' or str(j) == '__source()' and i in idx:
                    idx.remove(i)

        if add_observables:
            obs_names = [ob.name for ob in self.model.observables]
            idx = idx + obs_names

        self.eqs_for_tropicalization = idx
        return

    @staticmethod
    def _choose_max_pos_neg(array, diff_par):
        """
        Get the dominant reaction(s) of a species at a specific time point

        Parameters
        ----------
        array : An array with reaction rate values
        mon_names
        diff_par
        mon_comb

        Returns
        -------

        """
        mons_pos_neg = [numpy.where(array > 0)[0], numpy.where(array < 0)[0]]
        ascending_order = [False, True]
        mons_types = ['products', 'reactants']

        pos_neg_largest = [0] * 2
        range_0_1 = range(2)
        for ii, mon_type, mons_idx, ascending in zip(range_0_1, mons_types, mons_pos_neg, ascending_order):
            # largest_prod = 'NoDoms'
            # mons_comb_type = mon_comb[mon_type]
            # mon_names_ready = [mon_names.keys()[mon_names.values().index(i)] for i in mons_idx]
            if len(mons_idx) == 0:
                largest_prod = -1
            else:
                monomials_values = {idx: numpy.log10(numpy.abs(array[idx])) for idx in mons_idx}
                max_val = numpy.amax(listvalues(monomials_values))
                rr_monomials = [n for n, i in iteritems(monomials_values) if i > (max_val - diff_par) and max_val > -5]

                if not rr_monomials:
                    mons_idx[::-1].sort()
                    largest_prod = hf.uniquifier(mons_idx, biggest=len(array))
                else:
                    rr_monomials.sort(reverse=True)
                    largest_prod = hf.uniquifier(rr_monomials, biggest=len(array))

            pos_neg_largest[ii] = largest_prod
        return pos_neg_largest

    def signature(self, y, param_values):
        """
        Dynamic signature of the dominant species

        Parameters
        ----------
        y : np.array
            Species trajectories from the model simulation
        param_values: vector-like
            Parameter values used to obtain species trajectories

        Returns
        -------

        """
        assert self._is_setup, 'you must setup tropical first'
        sp_names = ['__s{0}_p'.format(j) if i % 2 == 0 else '__s{0}_c'.format(j)
                    for i, j in enumerate(numpy.repeat(self.eqs_for_tropicalization, 2))]
        sfull_dtype = list(zip(sp_names, itertools.repeat(int)))

        # Indexed numpy array that will contain the signature of each of the species to study
        all_signatures = numpy.ndarray(len(self.tspan), sfull_dtype)
        for sp_dyn in self.eqs_for_tropicalization:
            # reaction terms of all reaction rates in which species sp is involved and assign sign to see
            # if species is being consumed or produced.

            if isinstance(sp_dyn, str):
                species = self.model.observables.get(sp_dyn).species
                sp_name = sp_dyn
            else:
                species = [sp_dyn]
                sp_name = sp_dyn

            monomials_pre = []
            for sp in species:
                for term in self.model.reactions_bidirectional:
                    total_rate = 0
                    for mon_type, mon_sign in zip(['products', 'reactants'], [1, -1]):
                        if sp in term[mon_type]:
                            count = term[mon_type].count(sp)
                            total_rate = total_rate + (mon_sign * count * term['rate'])
                    if total_rate == 0:
                        continue
                    monomials_pre.append(total_rate)

            # Removing repeated reaction rates that can occur in observables
            monomials = []
            for m in monomials_pre:
                if -1*m in monomials_pre:
                    continue
                else:
                    monomials.append(m)

            # Dictionary whose keys are the symbolic monomials and the values are the simulation results
            mons_dict = OrderedDict()
            for mon_p in monomials:
                mon_p_values = mon_p

                if mon_p_values == 0:
                    mons_dict[mon_p] = [0] * len(self.tspan)
                elif isinstance(mon_p_values, Parameter):
                    mons_dict[mon_p] = [mon_p_values.value] * len(self.tspan)
                else:
                    var_prod = [atom for atom in mon_p_values.atoms(sympy.Symbol)]  # Variables of monomial
                    arg_prod = [0] * len(var_prod)
                    for idx, va in enumerate(var_prod):
                        if str(va).startswith('__'):
                            sp_idx = int(''.join(filter(str.isdigit, str(va))))
                            arg_prod[idx] = numpy.maximum(self.mach_eps, y[:, sp_idx])
                        else:
                            arg_prod[idx] = param_values[self.par_name_idx[va.name]]
                    # arg_prod = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_prod]
                    f_prod = sympy.lambdify(var_prod, mon_p_values)
                    prod_values = f_prod(*arg_prod)
                    mons_dict[mon_p] = prod_values
            mons_names = {}
            mons_array = numpy.zeros((len(mons_dict.keys()), len(self.tspan)))
            for idx, name in enumerate(mons_dict.keys()):
                mons_array[idx] = mons_dict[name]
                mons_names[idx] = name

            # This function takes a list of the reaction rates values and calculates the largest
            # reaction rate at each time point
            sign_pro = [0] * len(self.tspan)
            sign_rea = [0] * len(self.tspan)
            for t in range(len(self.tspan)):
                rr_t = mons_array[:, t]
                sign_pro[t], sign_rea[t] = self._choose_max_pos_neg(rr_t, self.diff_par)
            # signature_species = numpy.apply_along_axis(self._choose_max_pos_neg, 0, mons_array,
            #                                            *(mons_names, self.diff_par, self.all_comb[sp_name]))
            all_signatures[['__s{0}_p'.format(sp_dyn), '__s{0}_c'.format(sp_dyn)]] = list(zip(*[sign_pro, sign_rea]))
        return all_signatures

    # def set_combinations_sm(self):
    #     """
    #     Obtain all possible combinations of the reactions in which a species is involved
    #
    #
    #     Returns
    #     -------
    #
    #     """
    #     assert self.eqs_for_tropicalization, 'you must find passenger species first'
    #
    #     all_comb = {}
    #     for sp_dyn in self.eqs_for_tropicalization:
    #         # reaction terms
    #         pos_neg_combs = {}
    #         parts_reaction = ['products', 'reactants']
    #         parts_rev = [1, 0]
    #         signs = [1, -1]
    #
    #         if isinstance(sp_dyn, str):
    #             species = self.model.observables.get(sp_dyn).species
    #             sp_name = sp_dyn
    #         else:
    #             species = [sp_dyn]
    #             sp_name = sp_dyn
    #
    #         # We get the reaction rates from the bidirectional reactions in order to have reversible reactions
    #         # as one 'monomial'. This is helpful for visualization and other (I should think more about this)
    #         for mon_type, mon_sign, rev_parts in zip(parts_reaction, signs, parts_rev):
    #             monomials = []
    #
    #             for sp in species:
    #                 for term in self.model.reactions_bidirectional:
    #                     if sp in term[mon_type]:
    #                         # Add zero to monomials in cases like autocatalytic reactions where a species
    #                         # shows up both in reactants and products, and we are looking for the reactions that use a sp
    #                         # but the reaction produces the species overall
    #                         sp_count = term[mon_type].count(sp)
    #
    #                         if sp in term[parts_reaction[rev_parts]]:
    #                             count_reac = term['reactants'].count(sp)
    #                             count_pro = term['products'].count(sp)
    #                             mon_zero = mon_sign
    #                             if mon_type == 'reactants':
    #                                 if count_pro > count_reac:
    #                                     mon_zero = 0
    #                             else:
    #                                 if count_pro < count_reac:
    #                                     mon_zero = 0
    #                             monomials.append(mon_zero * term['rate'])
    #                         else:
    #                             monomials.append(sp_count * mon_sign * term['rate'])
    #
    #                     # Add reversible reaction rates on which the species is involved but was not added
    #                     # in the previous loop because it was not in the mon_type
    #                     if sp in term[parts_reaction[rev_parts]] and term['reversible']:
    #                         sp_count = term[parts_reaction[rev_parts]].count(sp)
    #                         monomials.append(sp_count * signs[rev_parts] * term['rate'])
    #             # remove zeros from reactions in which the species shows up both in reactants and products
    #             monomials = [value for value in monomials if value != 0]
    #             combs = len(monomials) + 1
    #
    #             mon_comb = OrderedDict()
    #             comb_counter = 0
    #             for L in range(1, combs):
    #                 prod_comb_names = {}
    #                 for subset in itertools.combinations(monomials, L):
    #                     subset = list(subset)
    #                     subset.sort(key=sympy.default_sort_key)
    #                     subset = tuple(subset)
    #                     rr_label = comb_counter
    #                     prod_comb_names[rr_label] = subset
    #                     comb_counter += 1
    #
    #                 mon_comb[L] = prod_comb_names
    #             pos_neg_combs[mon_type] = mon_comb
    #         all_comb[sp_name] = pos_neg_combs
    #     self.all_comb = all_comb
    #     return


def organize_dynsign_multi(signatures):
    species = signatures[0].keys()
    nsims = [0]*len(signatures)
    organized_dynsigns = {sp: {'production': nsims[:], 'consumption': nsims[:]} for sp in species}
    for sim_idx, dyn in enumerate(signatures):
        for sp in species:
            organized_dynsigns[sp]['production'][sim_idx] = dyn[sp][0]
            organized_dynsigns[sp]['consumption'][sim_idx] = dyn[sp][1]

    return organized_dynsigns


def signatures_to_dataframe(signatures, tspan, nsims):
    sim_ids = (numpy.repeat(range(nsims), [len(tspan)]*nsims))
    times = numpy.tile(tspan, nsims)

    idx = pd.MultiIndex.from_tuples(list(zip(sim_ids, times)))
    if not isinstance(signatures, numpy.ndarray):
        signatures = numpy.concatenate(signatures)
    return pd.DataFrame(signatures, index=idx)

def run_tropical(model, simulations, add_observables=False, passengers_by='imp_nodes',
                 diff_par=1, sp_to_vis=None, plot_type=0):
    """

    Parameters
    ----------
    plot_type : 0 to plot production of a species, 1 to plot consumption
    model: pysb.model
        model to analyze
    simulations: pysb.SimulationResult, or str
        Simulation result of model or path to SimualtionResult file
    passengers_by : str
        It can be 'qssa' or 'imp_nodes'. It defines the method to use for finding passenger species
    diff_par : float
        Magnitude difference that defines that a reaction is dominant over others.
    sp_to_vis : vector-like
        Indices of species to visualize

    Returns
    -------
    Dynamic signatures of dominant species of the model
    """
    trajectories, parameters, nsims, tspan = hf.get_simulations(simulations)
    tro = Tropical(model)
    tro.setup_tropical(tspan=tspan, diff_par=diff_par, passengers_by=passengers_by, add_observables=add_observables)
    signatures = tro.signature(y=trajectories, param_values=parameters[0])
    if sp_to_vis is not None:
        visualization_sp(model=model, tspan=tspan, y=trajectories, sp_to_vis=sp_to_vis,
                         all_signatures=signatures, plot_type=plot_type, param_values=parameters[0])
    # signatures['species_combinations'] = tro.all_comb
    return signatures


def run_tropical_multi(model, simulations, add_observables=False, passengers_by='imp_nodes',
                       diff_par=1, cpu_cores=1, verbose=False):
    """

    Parameters
    ----------
    model: pysb.model
        model to analyze
    simulations: pysb.SimulationResult, or str
        Simulation result of model or path to SimualtionResult file
    passengers_by : str
        It can be 'qssa' or 'imp_nodes'. It defines the method to use for finding passenger species
    diff_par : float
        Magnitude difference that defines that a reaction is dominant over others.
    cpu_cores: int
        Number of cores to use for running the analysis

    Returns
    -------
    Dynamic signatures of dominant species of the model

    """
    if Pool is None:
        raise Exception('Please install the pathos package for this feature')

    trajectories, parameters, nsims, tspan = hf.get_simulations(simulations)
    tro = Tropical(model)

    tro.setup_tropical(tspan=tspan, diff_par=diff_par, passengers_by=passengers_by, add_observables=add_observables)
    p = Pool(cpu_cores)
    if nsims == 1:
        trajectories = [trajectories]

    res = p.amap(tro.signature, trajectories, parameters)
    if verbose:
        while not res.ready():
            print ('We\'re not done yet, %s tasks to go!' % res._number_left)
            time.sleep(60)
    signatures = res.get()
    signatures = signatures_to_dataframe(signatures, tspan, nsims)
    return signatures