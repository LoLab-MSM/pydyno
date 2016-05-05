import copy
import itertools
import math
import re
from collections import OrderedDict
import matplotlib.pylab as plt
import numpy
import sympy
from pysb.integrate import odesolve
from stoichiometry_conservation_laws import conservation_relations


def _parse_name(spec):
    m = spec.monomer_patterns
    lis_m = []
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))
        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0], tmp_2[0]]))
    return '_'.join(lis_m)


def _heaviside_num(x):
    return 0.5 * (numpy.sign(x) + 1)


def find_nearest_zero(array):
    idx = numpy.nanargmin(numpy.abs(array))
    return array[idx]


class Tropical:
    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y = None  # ode solution, numpy array
        self.param_values = None
        self.passengers = None
        self.conservation = None
        self.value_conservation = {}
        self.tro_species = {}
        self.driver_signatures = None
        self.passenger_signatures = None
        self.mon_names = {}
        self.sol_pruned = None
        self.pruned = None
        self.eqs_for_tropicalization = None
        self.tropical_eqs = None

    def tropicalize(self, tspan=None, param_values=None, ignore=1, epsilon=1, rho=1, verbose=False):

        if verbose:
            print "Solving Simulation"

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")

        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.model.parameters):
                raise Exception("param_values must be the same length as model.parameters")
            if not isinstance(param_values, numpy.ndarray):
                param_values = numpy.array(param_values)
        else:
            # create parameter vector from the values in the model
            param_values = numpy.array([p.value for p in self.model.parameters])

        new_pars = dict((p.name, param_values[i]) for i, p in enumerate(self.model.parameters))
        self.param_values = new_pars

        self.y = odesolve(self.model, self.tspan, self.param_values)

        if verbose:
            print "Getting Passenger species"
        self.find_passengers(self.y[ignore:], verbose, epsilon)
        if verbose:
            print "Computing Conservation laws"
        self.conservation, self.value_conservation = conservation_relations(self.model)

        if verbose:
            print "equation to tropicalize"
        self.equations_to_tropicalize()
        if verbose:
            print "Getting tropicalized equations"
        self.final_tropicalization()
        self.data_drivers(self.y[ignore:])
        return

    def find_passengers(self, y, verbose=False, epsilon=None, ptge_similar=0.9, plot=False):
        sp_imposed_trace = {}
        self.passengers = []

        # Loop through all equations (i is equation number)
        for i, eq in enumerate(self.model.odes):
            sol = sympy.solve(eq, sympy.Symbol('__s%d' % i))  # Find equation of imposed trace
            sp_imposed_trace[i] = sol
        for k in sp_imposed_trace.keys():
            distance_imposed = 999
            for idx, solu in enumerate(sp_imposed_trace[k]):
                if solu.is_real:
                    print solu
                    imp_trace_values = [float(solu) + 1e-10] * len(self.tspan[1:])
                else:
                    for p in self.param_values:
                        solu = solu.subs(p, self.param_values[p])
                    variables = [atom for atom in solu.atoms(sympy.Symbol) if not re.match(r'\d', str(atom))]
                    f = sympy.lambdify(variables, solu, modules=dict(sqrt=numpy.lib.scimath.sqrt))
                    args = [y[str(l)] for l in variables] # arguments to put in the lambdify function
                    imp_trace_values = f(*args)

                if any(isinstance(n, complex) for n in imp_trace_values):
                    print 'solution' + ' ' + '%d' % idx + ' ' + 'from equation' + ' ' + str(k) + ' ' + 'is complex'
                    continue
                elif any(n < 0 for n in imp_trace_values):
                    print 'solution' + ' ' + '%d' % idx + ' ' + 'from equation' + ' ' + str(k) + ' ' + 'is negative'
                    continue
                hey = abs(numpy.log10(imp_trace_values) - numpy.log10(y['__s%d' % k]))
                if max(hey) < distance_imposed:
                    distance_imposed = max(hey)
                if plot:
                    plt.figure()
                    plt.semilogy(self.tspan[1:], imp_trace_values, 'r--', linewidth=5, label='imposed')
                    plt.semilogy(self.tspan[1:], y['__s%d' % k], label='full')
                    plt.legend(loc=0)
                    plt.xlabel('time', fontsize=20)
                    plt.ylabel('population', fontsize=20)
                    if max(hey) < epsilon:
                        plt.title(str(self.model.species[k]) + 'passenger', fontsize=20)
                    else:
                        plt.title(self.model.species[k], fontsize=20)
                    plt.savefig('/home/oscar/Documents/tropical_project/' + str(self.model.species[k]) + '.jpg',
                                format='jpg', dpi=400)

            if distance_imposed < epsilon:
                self.passengers.append(k)

        # plt.show()


        return self.passengers

    def passenger_equations(self):
        passenger_eqs = {}
        for i, j in enumerate(self.passengers):
            passenger_eqs[j] = self.model.odes[self.passengers[i]]
        return passenger_eqs

    def equations_to_tropicalize(self):
        idx = list(set(range(len(self.model.odes))) - set(self.passengers))
        eqs = {i: self.model.odes[i] for i in idx}

        for l in eqs.keys():  # Substitutes the values of the algebraic system
            #       for k in self.sol_pruned.keys(): eqs[l]=eqs[l].subs(sympy.Symbol('s%d' % k), self.sol_pruned[k])
            for q in self.value_conservation.keys():
                eqs[l] = eqs[l].subs(q, self.value_conservation[q])
        # for i in eqs.keys():
        #             for par in self.model.parameters: eqs[i] = sympy.simplify(eqs[i].subs(par.name, par.value))
        self.eqs_for_tropicalization = eqs

        return eqs

    def final_tropicalization(self):
        tropicalized = {}

        for j in sorted(self.eqs_for_tropicalization.keys()):
            if type(self.eqs_for_tropicalization[j]) == sympy.Mul:
                tropicalized[j] = self.eqs_for_tropicalization[j]  # If Mul=True there is only one monomial
            elif self.eqs_for_tropicalization[j] == 0:
                print 'there are no monomials'
            else:
                ar = sorted(self.eqs_for_tropicalization[j].as_coefficients_dict(),
                            key=str)  # List of the terms of each equation
                asd = 0
                for l, k in enumerate(ar):
                    p = k
                    for f, h in enumerate(ar):
                        if k != h:
                            p *= sympy.Heaviside(sympy.log(abs(k)) - sympy.log(abs(h)))
                    asd += p
                tropicalized[j] = asd

        self.tropical_eqs = tropicalized
        return tropicalized

    def data_drivers(self, y, plot_drivers=None):
        mach_eps = numpy.finfo(float).eps
        tropical_system = self.tropical_eqs
        trop_data = OrderedDict()
        signature_sp = {}
        driver_monomials = OrderedDict()

        for i in tropical_system.keys():
            signature = numpy.zeros(len(self.tspan[1:]), dtype=int)
            mons_data = {}
            mons = sorted(tropical_system[i].as_coefficients_dict().items(), key=str)
            mons_matrix = numpy.zeros((len(mons), len(self.tspan[1:])), dtype=float)
            spe_monomials = OrderedDict(sorted(self.model.odes[i].as_coefficients_dict().items(), key=str))
            driver_monomials[i] = spe_monomials

            for q, m_s in enumerate(mons):
                j = list(m_s)
                jj = copy.deepcopy(j[0])
                for par in self.param_values:
                    j[0] = j[0].subs(par, self.param_values[par])
                var_to_study = [atom for atom in j[0].atoms(sympy.Symbol) if
                                not re.match(r'\d', str(atom))]  # Variables of monomial
                arg_f1 = [numpy.maximum(mach_eps, y[str(va)]) for va in var_to_study]
                f1 = sympy.lambdify(var_to_study, j[0],
                                    modules=dict(Heaviside=_heaviside_num, log=numpy.log, Abs=numpy.abs))
                mon_values = f1(*arg_f1)
                mons_data[str(jj).partition('*Heaviside')[0]] = mon_values
                mons_matrix[q] = mon_values
            for col in range(len(self.tspan[1:])):
                signature[col] = numpy.nonzero(mons_matrix[:, col])[0][0]
            signature_sp[i] = signature
            trop_data[i] = mons_data
        self.driver_signatures = signature_sp
        self.mon_names = driver_monomials
        self.tro_species = trop_data
        return trop_data

    def visualization(self, driver_species=None):
        mach_eps = numpy.finfo(float).eps
        tropical_system = self.tropical_eqs
        species_ready = []
        if driver_species is not None:
            species_ready = list(set(driver_species).intersection(self.tro_species.keys()))

        elif driver_species is None:
            raise Exception('list of driver species must be defined')

        if not species_ready:
            raise Exception('None of the input species is a driver')

        colors = itertools.cycle(['#000000','#00FF00','#0000FF','#FF0000','#01FFFE','#FFA6FE','#FFDB66','#006401',
                                  '#010067','#95003A','#007DB5','#FF00F6','#FFEEE8','#774D00','#90FB92','#0076FF',
                                  '#D5FF00','#FF937E','#6A826C','#FF029D','#FE8900','#7A4782','#7E2DD2','#85A900',
                                  '#FF0056','#A42400','#00AE7E'])
        colors2 = itertools.cycle(['#000000','#00FF00','#0000FF','#FF0000','#01FFFE','#FFA6FE','#FFDB66','#006401',
                                  '#010067','#95003A','#007DB5','#FF00F6','#FFEEE8','#774D00','#90FB92','#0076FF',
                                  '#D5FF00','#FF937E','#6A826C','#FF029D','#FE8900','#7A4782','#7E2DD2','#85A900',
                                  '#FF0056','#A42400','#00AE7E'])

        sep = len(self.tspan) / 1

        for sp in species_ready:
            si_flux = 0
            plt.figure()
            plt.subplot(311)
            monomials = []
            monomials_inf = self.mon_names[sp]
            for name in self.tro_species[sp].keys():
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
                            color=next(colors), marker=r'$\uparrow$',
                            s=numpy.array([m_value[k] for k in x_concentration])[
                              ::int(math.ceil(len(self.tspan) / sep))])
                if monomials_inf[sympy.sympify(name)] < 0:
                    plt.scatter(
                            x_points[::int(math.ceil(len(self.tspan) / sep))],
                            prueba_y[::int(math.ceil(len(self.tspan) / sep))],
                            color=next(colors), marker=r'$\downarrow$',
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
                var_to_study = [atom for atom in j.atoms(sympy.Symbol) if
                                not re.match(r'\d', str(atom))]  # Variables of monomial

                arg_f1 = [numpy.maximum(mach_eps, self.y[str(va)][1:]) for va in var_to_study]
                f1 = sympy.lambdify(var_to_study, j,
                                    modules=dict(Heaviside=_heaviside_num, log=numpy.log, Abs=numpy.abs))
                mon_values = f1(*arg_f1)
                mon_name = name.partition('__')[2]
                plt.plot(self.tspan[1:], mon_values, label=mon_name, color=next(colors2))
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

    def get_pruned_equations(self):
        return self.pruned

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
