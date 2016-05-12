from __future__ import print_function
import math
from collections import OrderedDict
import matplotlib.pylab as plt
import numpy
import sympy
import seaborn as sns
from pysb.integrate import odesolve



class Tropical:
    mach_eps = numpy.finfo(float).eps

    def __init__(self, model):
        """
        Constructor of tropical function
        :param model:
        """
        self.model = model
        self.tspan = None
        self.y = None
        self.param_values = None
        self.passengers = []
        self.tro_species = {}
        self.driver_signatures = None
        self.passenger_signatures = None
        self.mon_names = {}
        self.eqs_for_tropicalization = None
        self.tropical_eqs = None

    @classmethod
    def _heaviside_num(cls, x):
        """Returns result of Heavisde function

            Keyword arguments:
            x -- argument to Heaviside function
        """
        return 0.5 * (numpy.sign(x) + 1)

    def tropicalize(self, tspan=None, param_values=None, ignore=1, epsilon=1, rho=1, verbose=False):

        if verbose:
            print("Solving Simulation")

        if tspan is not None:
            self.tspan = tspan
        else:
            raise Exception("'tspan' must be defined.")

        if param_values:
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
        self.find_passengers(self.y[ignore:], epsilon)

        if verbose:
            print("equation to tropicalize")
        self.equations_to_tropicalize

        if verbose:
            print("Getting tropicalized equations")
        self.final_tropicalization()
        self.data_drivers(self.y[ignore:])
        return

    def find_passengers(self, y, epsilon=None, ptge_similar=0.9, plot=False):
        sp_imposed_trace = []
        assert not self.passengers

        # Loop through all equations (i is equation number)
        for i, eq in enumerate(self.model.odes):
            sol = sympy.solve(eq, sympy.Symbol('__s%d' % i))  # Find equation of imposed trace
            sp_imposed_trace.append(sol)
        for sp_idx, trace_soln in enumerate(sp_imposed_trace):
            distance_imposed = 999
            for idx, solu in enumerate(trace_soln):
                if solu.is_real:
                    imp_trace_values = [float(solu) + self.mach_eps] * (len(self.tspan)-1)
                else:
                    for p in self.param_values:
                        solu = solu.subs(p, self.param_values[p])
                    # @TODO CHECK THAT THIS WORKS FOR VARIOUS CASES
                    # @TODO EXPLAIN THIS
                    variables = [atom for atom in solu.atoms(sympy.Symbol)]
                    f = sympy.lambdify(variables, solu, modules=dict(sqrt=numpy.lib.scimath.sqrt))
                    args = [y[str(l)] for l in variables]  # arguments to put in the lambdify function
                    imp_trace_values = f(*args)

                if any(isinstance(n, complex) for n in imp_trace_values):
                    print("solution {0} from equation {1} is complex".format(idx, sp_idx))
                    continue
                elif any(n < 0 for n in imp_trace_values):
                    print("solution {0} from equation {1} is negative".format(idx, sp_idx))
                    continue
                diff_trace_ode = abs(numpy.log10(imp_trace_values) - numpy.log10(y['__s%d' % sp_idx]))
                if max(diff_trace_ode) < distance_imposed:
                    distance_imposed = max(diff_trace_ode)

                # @TODO move to its own function
                if plot:
                    plt.figure()
                    plt.semilogy(self.tspan[1:], imp_trace_values, 'r--', linewidth=5, label='imposed')
                    plt.semilogy(self.tspan[1:], y['__s%d' % trace_soln], label='full')
                    plt.legend(loc=0)
                    plt.xlabel('time', fontsize=20)
                    plt.ylabel('population', fontsize=20)
                    if max(diff_trace_ode) < epsilon:
                        plt.title(str(self.model.species[trace_soln]) + 'passenger', fontsize=20)
                    else:
                        plt.title(self.model.species[trace_soln], fontsize=20)
                    plt.savefig(
                        '/home/oscar/Documents/tropical_project/' + str(self.model.species[trace_soln]) + '.jpg',
                        format='jpg', dpi=400)

            if distance_imposed < epsilon:
                self.passengers.append(sp_idx)

        return self.passengers

    @property
    def equations_to_tropicalize(self):
        idx = list(set(range(len(self.model.odes))) - set(self.passengers))
        eqs = {i: self.model.odes[i] for i in idx}
        self.eqs_for_tropicalization = eqs
        return

    # @TODO document this really well
    def final_tropicalization(self):
        tropicalized = {}

        for j in sorted(self.eqs_for_tropicalization.keys()):
            coeffs = self.eqs_for_tropicalization[j].as_coefficients_dict()
            if len(coeffs.keys()) == 1:
                print('there is one or no monomials in species {0}'.format(j))
            else:
                monomials = sorted(coeffs, key=str)  # List of the terms of each equation
                trop_eq = 0
                for mon1 in monomials:
                    trop_monomial = mon1
                    for mon2 in monomials:
                        if mon1 != mon2:
                            trop_monomial *= sympy.Heaviside(sympy.log(abs(mon1)) - sympy.log(abs(mon2)))
                    trop_eq += trop_monomial
                tropicalized[j] = trop_eq
        self.tropical_eqs = tropicalized
        return

    def data_drivers(self, y):
        trop_data = OrderedDict()
        signature_sp = {}
        driver_monomials = OrderedDict()

        for i, eqn_item in self.tropical_eqs.items():
            signature = numpy.zeros(len(self.tspan) - 1, dtype=int)
            mons_data = {}
            mons = sorted(eqn_item.as_coefficients_dict().items(), key=str)
            mons_matrix = numpy.zeros((len(mons), len(self.tspan)-1), dtype=float)
            spe_monomials = OrderedDict(sorted(self.model.odes[i].as_coefficients_dict().items(), key=str))
            driver_monomials[i] = spe_monomials

            for q, m_s in enumerate(mons):
                mons_list = list(m_s)
                mdkey = str(mons_list[0]).partition('*Heaviside')[0]
                for par in self.param_values:
                    mons_list[0] = mons_list[0].subs(par, self.param_values[par])
                var_to_study = [atom for atom in mons_list[0].atoms(sympy.Symbol)]  # Variables of monomial
                arg_f1 = [numpy.maximum(self.mach_eps, y[str(va)]) for va in var_to_study]
                f1 = sympy.lambdify(var_to_study, mons_list[0],
                                    modules=dict(Heaviside=self._heaviside_num, log=numpy.log, Abs=numpy.abs))
                mon_values = f1(*arg_f1)
                mons_data[mdkey] = mon_values
                mons_matrix[q] = mon_values
            for col in range(len(self.tspan)-1):
                signature[col] = numpy.nonzero(mons_matrix[:, col])[0][0]
            signature_sp[i] = signature
            trop_data[i] = mons_data
        self.driver_signatures = signature_sp
        self.mon_names = driver_monomials
        self.tro_species = trop_data
        return

    def visualization(self, driver_species=None):
        tropical_system = self.tropical_eqs
        if driver_species:
            species_ready = list(set(driver_species).intersection(self.tro_species.keys()))

        else:
            raise Exception('list of driver species must be defined')

        if not species_ready:
            raise Exception('None of the input species is a driver')

        colors = sns.color_palette("Set2", max([len(ode.as_coeff_add()[1]) for ode in self.model.odes]))

        sep = len(self.tspan) / 1

        for sp in species_ready:
            si_flux = 0
            plt.figure()
            plt.subplot(311)
            monomials = []
            monomials_inf = self.mon_names[sp]
            for idx, name in enumerate(self.tro_species[sp].keys()):
                m_value = self.tro_species[sp][name]
                x_concentration = numpy.nonzero(m_value)[0]
                monomials.append(name)
                si_flux += 1
                x_points = [self.tspan[x] for x in x_concentration]
                prueba_y = numpy.repeat(2 * si_flux, len(x_concentration))
                if monomials_inf[sympy.sympify(name)] > 0:
                    plt.scatter(
                        x_points[::int(math.ceil(len(self.tspan) / sep))],
                        prueba_y[::int(math.ceil(len(self.tspan) / sep))],
                        color=colors[idx], marker=r'$\uparrow$',
                        s=numpy.array([m_value[k] for k in x_concentration])[
                          ::int(math.ceil(len(self.tspan) / sep))])
                if monomials_inf[sympy.sympify(name)] < 0:
                    plt.scatter(
                        x_points[::int(math.ceil(len(self.tspan) / sep))],
                        prueba_y[::int(math.ceil(len(self.tspan) / sep))],
                        color=colors[idx], marker=r'$\downarrow$',
                        s=numpy.array([m_value[k] for k in x_concentration])[
                          ::int(math.ceil(len(self.tspan) / sep))])

            y_pos = numpy.arange(2, 2 * si_flux + 4, 2)
            plt.yticks(y_pos, monomials, fontsize=12)
            plt.ylabel('Monomials', fontsize=16)
            plt.xlim(0, self.tspan[-1])
            plt.ylim(0, max(y_pos))

            plt.subplot(312)
            mons = tropical_system[sp].as_coefficients_dict().items()
            mons_matrix = numpy.zeros((len(mons), len(self.tspan[1:])), dtype=float)

            for q, name in enumerate(self.tro_species[sp].keys()):
                j = sympy.sympify(name)
                for par in self.param_values:
                    j = j.subs(par, self.param_values[par])
                var_to_study = [atom for atom in j.atoms(sympy.Symbol)]  # Variables of monomial

                arg_f1 = [numpy.maximum(self.mach_eps, self.y[str(va)][1:]) for va in var_to_study]
                f1 = sympy.lambdify(var_to_study, j,
                                    modules=dict(Heaviside=self._heaviside_num, log=numpy.log, Abs=numpy.abs))
                mon_values = f1(*arg_f1)
                mon_name = name.partition('__')[2]
                plt.plot(self.tspan[1:], mon_values, label=mon_name, color=colors[q])
                mons_matrix[q] = mon_values
            plt.legend(loc=0)

            plt.subplot(313)
            plt.plot(self.tspan[1:], self.y['__s%d' % sp][1:])
            plt.ylabel('Molecules', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.suptitle('Tropicalization' + ' ' + str(self.model.species[sp]))

            plt.savefig('/home/oscar/Desktop/' + 's%d' % sp, format='png', bbox_inches='tight', dpi=400)
            plt.show()

        # plt.ylim(0, len(monomials)+1)
        return

    def get_trop_data(self):
        return self.tro_species

    def get_passenger(self):
        return self.passengers

    def get_tropical_eqs(self):
        return self.tropical_eqs

    def get_driver_signatures(self):
        return self.driver_signatures


def run_tropical(model, tspan, parameters=None, sp_visualize=None):
    tr = Tropical(model)
    tr.tropicalize(tspan, parameters)
    if sp_visualize is not None:
        tr.visualization(driver_species=sp_visualize)
    return tr.get_driver_signatures()
