import pydyno.util as hf
import numpy as np
from sympy import Symbol, lambdify
from collections import OrderedDict
from pysb import Parameter
import itertools
import time
import pandas as pd

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None


class Discretize(object):
    """
    Discretizes species trajectories from a PySB model
    model: pysb.Model
        Model to analyze
    simulation: pysb.simulator or hdf5.
        Simulations to discretize
    """
    mach_eps = 1e-11

    def __init__(self, model, simulations, diff_par):
        self._model = model
        self._trajectories, self._parameters, self._nsims, self._tspan = hf.get_simulations(simulations)
        self._par_name_idx = {j.name: i for i, j in enumerate(self.model.parameters)}
        self._diff_par = diff_par
        if self._nsims == 1:
            self._parameters = self._parameters[0]

    @property
    def model(self):
        return self._model

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def parameters(self):
        return self._parameters

    @property
    def nsims(self):
        return self._nsims

    @property
    def tspan(self):
        return self._tspan

    @property
    def par_name_idx(self):
        return self._par_name_idx

    @property
    def diff_par(self):
        return self._diff_par

    @staticmethod
    def _choose_max_pos_neg(array, diff_par):
        """
        Get the dominant reaction(s) of a species at a specific time point

        Parameters
        ----------
        array : An array with reaction rate values
        diff_par

        Returns
        -------

        """
        # Gets the indices of posive and negative reaction rates
        mons_pos_neg = [np.where(array > 0)[0], np.where(array < 0)[0]]
        ascending_order = [False, True]
        mons_types = ['products', 'reactants']

        pos_neg_largest = [0] * 2
        range_0_1 = range(2)
        for ii, mon_type, mons_idx, ascending in zip(range_0_1, mons_types, mons_pos_neg, ascending_order):
            # If there are no positive (or negative) reaction rates, it returns -1
            if len(mons_idx) == 0:
                largest_prod = -1
            else:
                # Creates dictionary whose key is the reaction rate index and the value is the log value
                # of the reaction rate
                reactions_values = {idx: np.log10(np.abs(array[idx])) for idx in mons_idx}
                max_val = np.amax(list(reactions_values.values()))
                rr_monomials = [n for n, i in reactions_values.items() if i > (max_val - diff_par) and max_val > -5]

                # If there is no a set of dominant reaction rates, then all the
                # reaction rates are equally dominant
                if not rr_monomials:
                    mons_idx[::-1].sort()
                    largest_prod = hf.uniquifier(mons_idx, biggest=len(array))
                else:
                    rr_monomials.sort(reverse=True)
                    largest_prod = hf.uniquifier(rr_monomials, biggest=len(array))

            pos_neg_largest[ii] = largest_prod
        return pos_neg_largest

    def get_important_nodes(self, get_passengers_by='imp_nodes', add_observables=False):
        """
        Function to get nodes to study
        Parameters
        ----------
        get_passengers_by: str
            options include `imp_nodes`
        add_observables: bool
            Add observables as species to study

        Returns
        -------
        Species nodes to study

        """
        if get_passengers_by == 'imp_nodes':
            passengers = hf.find_nonimportant_nodes(self.model)
        else: raise ValueError('Method to obtain passengers not supported')

        idx = list(set(range(len(self.model.species))) - set(passengers))

        if add_observables:
            obs_names = [ob.name for ob in self.model.observables]
            idx = idx + obs_names
        return idx

    def __signature(self, y, param_values):
        """
        Dynamic signature of the dominant species

        Parameters
        ----------
        diff_par: float
            Magnitude to define when a reaction rate is dominat

        Returns
        -------

        """
        imp_nodes = self.get_important_nodes()

        # Defining dtypes for the indexes of the signatures array
        sp_names = ['__s{0}_p'.format(j) if i % 2 == 0 else '__s{0}_c'.format(j)
                    for i, j in enumerate(np.repeat(imp_nodes, 2))]
        sfull_dtype = list(zip(sp_names, itertools.repeat(int)))
        # Indexed numpy array that will contain the signature of each of the species to study
        all_signatures = np.ndarray(len(self.tspan), sfull_dtype)

        for sp_dyn in imp_nodes:

            if isinstance(sp_dyn, str):
                species = self.model.observables.get(sp_dyn).species
                sp_name = sp_dyn
            else:
                species = [sp_dyn]
                sp_name = sp_dyn

            # Obtaining reaction rates in which a specific species sp is involved and it assigns
            # a sign to determine if sp is being consumed or produced
            reaction_rates_pre = []
            for sp in species:
                for term in self.model.reactions_bidirectional:
                    total_rate = 0
                    for mon_type, mon_sign in zip(['products', 'reactants'], [1, -1]):
                        if sp in term[mon_type]:
                            count = term[mon_type].count(sp)
                            total_rate = total_rate + (mon_sign * count * term['rate'])
                    if total_rate == 0:
                        continue
                    reaction_rates_pre.append(total_rate)

            # Removing repeated reaction rates that can occur in observables
            reaction_rate = []
            for m in reaction_rates_pre:
                if -1*m in reaction_rates_pre:
                    continue
                else:
                    reaction_rate.append(m)

            # Dictionary whose keys are the symbolic reaction rates and the values are the simulation results
            rr_dict = OrderedDict()
            for mon_p in reaction_rate:
                mon_p_values = mon_p

                if mon_p_values == 0:
                    rr_dict[mon_p] = [0] * len(self.tspan)
                elif isinstance(mon_p_values, Parameter):
                    rr_dict[mon_p] = [mon_p_values.value] * len(self.tspan)
                else:
                    var_prod = [atom for atom in mon_p_values.atoms(Symbol)]  # Variables of monomial
                    arg_prod = [0] * len(var_prod)
                    for idx, va in enumerate(var_prod):
                        if str(va).startswith('__'):
                            sp_idx = int(''.join(filter(str.isdigit, str(va))))
                            arg_prod[idx] = np.maximum(self.mach_eps, y[:, sp_idx])
                        else:
                            # print (param_values)
                            # print (self.par_name_idx[va.name])
                            arg_prod[idx] = param_values[self.par_name_idx[va.name]]
                    # arg_prod = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_prod]
                    f_prod = lambdify(var_prod, mon_p_values)
                    prod_values = f_prod(*arg_prod)
                    rr_dict[mon_p] = prod_values
            mons_array = np.zeros((len(rr_dict.keys()), len(self.tspan)))
            for idx, name in enumerate(rr_dict.keys()):
                mons_array[idx] = rr_dict[name]

            # This function takes a list of the reaction rates values and calculates the largest
            # reaction rate at each time point
            sign_pro = [0] * len(self.tspan)
            sign_rea = [0] * len(self.tspan)
            for t in range(len(self.tspan)):
                rr_t = mons_array[:, t]
                sign_pro[t], sign_rea[t] = self._choose_max_pos_neg(rr_t, self.diff_par)

            all_signatures[['__s{0}_p'.format(sp_dyn), '__s{0}_c'.format(sp_dyn)]] = list(zip(*[sign_pro, sign_rea]))
        return all_signatures

    def get_signatures(self, cpu_cores=1, verbose=False):
        if cpu_cores == 1:
            if self.nsims == 1:
                signatures = self.__signature(self._trajectories, self.parameters)
                signatures = signatures_to_dataframe(signatures, self.tspan, self.nsims)
                signatures = signatures.transpose().stack(0)
                return signatures
            else:
                signatures = [0] * self.nsims
                for idx in range(self.nsims):
                    signatures[idx] = self.__signature(self._trajectories[idx], self.parameters[idx])
                signatures = signatures_to_dataframe(signatures, self.tspan, self.nsims)
                signatures = signatures.transpose().stack(0)
                return signatures
        else:
            if Pool is None:
                raise Exception('Please install the pathos package for this feature')
            if self.nsims == 1:
                self._trajectories = [self._trajectories]
                self._parameters = [self._parameters]

            p = Pool(cpu_cores)
            res = p.amap(self.__signature, self._trajectories, self.parameters)
            if verbose:
                while not res.ready():
                    print ('We\'re not done yet, %s tasks to go!' % res._number_left)
                    time.sleep(60)
            signatures = res.get()
            signatures = signatures_to_dataframe(signatures, self.tspan, self.nsims)
            signatures = signatures.transpose().stack(0)
            return signatures


def signatures_to_dataframe(signatures, tspan, nsims):
    sim_ids = (np.repeat(range(nsims), [len(tspan)]*nsims))
    times = np.tile(tspan, nsims)

    idx = pd.MultiIndex.from_tuples(list(zip(sim_ids, times)))
    if not isinstance(signatures, np.ndarray):
        signatures = np.concatenate(signatures)
    return pd.DataFrame(signatures, index=idx)