import pysb.bng
import sympy
import networkx
import itertools
from stoichiometry_conservation_laws import conservation_relations, conservation_laws_values
from sympy.core.relational import Equality
from pysb.integrate import odesolve
import matplotlib.pyplot as plt
import re
import numpy


def do(self, e, i=None):
    """do `e` to both sides of self using function given or
    model expression with a variable representing each side:
    >> eq.do(i + 2)  # add 2 to both sides of Equality
    >> eq.do(i + a, i)  # add `a` to both sides; 3rd parameter identifies `i`
    >> eq.do(lambda i: i + a)  # modification given as a function
    """
    if isinstance(e, (sympy.FunctionClass, sympy.Lambda, type(lambda: 1))):
        return self.applyfunc(e)
    e = sympy.S(e)
    i = ({i} if i else e.free_symbols) - self.free_symbols
    if len(i) != 1:
        raise ValueError('not sure what symbol is being used to represent a side')
    i = i.pop()
    f = lambda side: e.subs(i, side)
    return self.func(*[f(side) for side in self.args])


Equality.do = do


class TheoSols:
    def __init__(self, model):
        self.model = model
        self.tspan = None
        self.y = None
        self.reactions_from_rules = None
        self.new_units = None
        self.grouped_odes = None
        self.conservation_laws = None
        self.value_conservation_laws = None
        self.reverse_to_noreverse_vars = None
        self.indep_variables_odes = None
        self.indep_odes_ic = None
        self.sol_indep_odes = None
        self.new_ode_variables = None
        self.eqs_to_plot = None

    def sol_theo(self, tspan=None, verbose=True):
        if self.model.odes is None or self.model.odes == []:
            pysb.bng.generate_equations(self.model)

        if verbose: print "Solving theoretically"

        if tspan is not None:
            self.tspan = tspan
        elif self.tspan is None:
            raise Exception("'tspan' must be defined.")

        self.reactions_rules()
        self.new_ode_units(self.reactions_from_rules)
        self.group_odes(self.reactions_from_rules, self.new_units)
        self.all_conservation_laws(self.reactions_from_rules, self.new_units)
        self.indep_var_odes(self.reactions_from_rules, self.new_units, self.grouped_odes,
                            self.reverse_to_noreverse_vars)
        self.solve_indep_var_odes(self.indep_variables_odes, self.indep_odes_ic, self.value_conservation_laws)
        self.compare_to_numerical(self.eqs_to_plot, self.new_ode_variables)

    def reactions_rules(self):
        rcts_rules = {}
        for rule in self.model.rules:
            rcts_no_reverse = 0
            rcts_reverse = 0
            rule_reactants = []
            for reaction in self.model.reactions:
                if reaction['rule'][0] == rule.name and reaction['reverse'][0] == False:
                    rule_reactants.append(reaction['reactants'])

            G = networkx.Graph()
            G.add_edges_from(rule_reactants)
            sp_interactions = [G[node].keys() for node in networkx.nodes(G)]
            vars_set = set(map(tuple, sp_interactions))
            final_vars = map(list, vars_set)
            if len(final_vars) == 1:
                final_vars.append(final_vars[0])
            interactions = list(itertools.product(*final_vars))

            for interaction in interactions:
                for reaction in self.model.reactions:
                    if reaction['rule'][0] == rule.name and reaction['reverse'][0] == False and sorted(
                            reaction['reactants']) == sorted(interaction):
                        if reaction['reactants'][0] == reaction['reactants'][1]:
                            rcts_no_reverse += 2 * reaction['rate']
                        else:
                            rcts_no_reverse += reaction['rate']
                    if reaction['rule'][0] == rule.name and reaction['reverse'][0] == True and sorted(
                            reaction['products']) == sorted(interaction):
                        if reaction['products'][0] == reaction['products'][1]:
                            rcts_reverse += 2 * reaction['rate']
                        else:
                            rcts_reverse += reaction['rate']

            rcts_rules[rule.name] = {'no_reverse': sympy.factor(rcts_no_reverse), 'reverse': sympy.factor(rcts_reverse)}
        self.reactions_from_rules = rcts_rules
        return self.reactions_from_rules

    def new_ode_units(self, rcts_rules):

        # this line loops over the reactions and factor the monomials and that way  we get the new units
        new_units = {}
        for rc in rcts_rules:
            var = sympy.simplify(sympy.factor(rcts_rules[rc]['no_reverse'])).as_coeff_mul()[1]
            final_vars = [ex for ex in var if
                          (str(ex).startswith('__') or type(ex) == sympy.Pow or type(ex) == sympy.Add)]
            new_units[rc] = final_vars
        self.new_units = new_units
        return self.new_units

    def group_odes(self, rcts_rules, new_units):
        # this defines the odes for the new units
        new_units_odes = {}
        for r in new_units:
            for eq in new_units[r]:
                if type(eq) == sympy.Pow:
                    eq_name = eq.args[0]
                    new_units_odes[eq_name] = sympy.simplify(rcts_rules[r]['reverse'] - rcts_rules[r]['no_reverse'])
                else:
                    new_units_odes[eq] = rcts_rules[r]['reverse'] - rcts_rules[r]['no_reverse']
        self.grouped_odes = new_units_odes
        return self.grouped_odes

    def all_conservation_laws(self, rcts_rules, new_units):
        cl, cl_values = conservation_relations(self.model)
        for idx, nam in enumerate(new_units):
            if len(new_units[nam]) > 1:
                new_cons = new_units[nam][0] - new_units[nam][1] - sympy.symbols('b%d' % idx, real=True)
                new_cons_value = conservation_laws_values(self.model, new_cons)
                cl.append(new_cons)
                cl_values[new_cons_value.keys()[0]] = new_cons_value.values()[0]
        # this uses the conservation laws to define the new units in terms of the other variables
        equal_units = {}
        for rul in new_units:
            for un in new_units[rul]:
                for cc in cl:
                    if sympy.solve(cc, un):
                        equal_units[un] = sympy.solve(cc, un)[0]

        # this line defines the reverse reactions in term of the other variables (to have the same variable in the eq)
        equal_const = {}
        for r in rcts_rules:
            if not rcts_rules[r]['reverse'] == 0:
                for cc in cl:
                    # print sympy.solve(sympy.collect_const(cc.rhs), rcts_rules[r]['reverse'].as_two_terms()[1])
                    if sympy.solve(cc, rcts_rules[r]['reverse'].as_coeff_mul()[1][1]):
                        equal_const[rcts_rules[r]['reverse'].as_coeff_mul()[1][1]] = sympy.collect_const(
                                sympy.solve(cc, rcts_rules[r]['reverse'].as_coeff_mul()[1][1])[0])
                    if sympy.solve(sympy.collect_const(cc), rcts_rules[r]['reverse'].as_coeff_mul()[1][1]):
                        equal_const[rcts_rules[r]['reverse'].as_coeff_mul()[1][1]] = sympy.collect_const(
                                sympy.solve(sympy.collect_const(cc), rcts_rules[r]['reverse'].as_coeff_mul()[1][1])[
                                    0].evalf())

        equal_const_units = equal_units.copy()
        equal_const_units.update(equal_const)

        self.reverse_to_noreverse_vars = equal_const_units
        self.conservation_laws = cl
        self.value_conservation_laws = cl_values
        return

    def indep_var_odes(self, rcts_rules, new_units, new_units_odes, equal_const_units):
        variables_to_change = {}
        for unit in new_units:
            if len(new_units[unit]) > 1:
                for idx in range(len(new_units[unit])):
                    if rcts_rules[unit]['reverse'] != 0:
                        tmp_dict = {new_units[unit][idx]: equal_const_units[new_units[unit][idx]],
                                    rcts_rules[unit]['reverse'].as_coeff_mul()[1][1]: equal_const_units[
                                        rcts_rules[unit]['reverse'].as_coeff_mul()[1][1]]}
                    else:
                        tmp_dict = {new_units[unit][idx]: equal_const_units[new_units[unit][idx]]}
                    variables_to_change[new_units[unit][1 - idx]] = tmp_dict
            else:
                tmp_dict = {rcts_rules[unit]['reverse'].as_coeff_mul()[1][1]: equal_const_units[
                    rcts_rules[unit]['reverse'].as_coeff_mul()[1][1]]}
                variables_to_change[new_units[unit][0].as_base_exp()[0]] = tmp_dict

        # this is a simple change of variable to allow sympy to solve the differential equations.
        new_ode_vars = {}
        final_odes = {}
        for num, ode in enumerate(new_units_odes):
            new_ode_vars[ode] = sympy.symbols('U%d' % num)
            final_odes[sympy.symbols('U%d' % num)] = new_units_odes[ode].subs(variables_to_change[ode]).subs(ode,
                                                                                                             sympy.symbols(
                                                                                                                     'U%d' % num))
        new_ode_vars_ic = {}
        for idx, u in enumerate(new_ode_vars):
            new_ode_vars_ic[new_ode_vars[u](0)] = \
                conservation_laws_values(self.model, u - sympy.Symbol('d%d' % idx)).values()[0]
        self.indep_variables_odes = final_odes
        self.indep_odes_ic = new_ode_vars_ic
        self.new_ode_variables = new_ode_vars
        return self.indep_variables_odes

    def solve_indep_var_odes(self, final_odes, new_ode_vars_ic, cl_values):
        # solving the differential equations
        solutions = []
        equations = []
        solutions_eq = []
        for nom in final_odes:
            s = sympy.var('s')
            t = sympy.symbols('t')
            equation = sympy.Eq(final_odes[nom].subs(nom, nom(t)), nom(t).diff(t))
            equations.append(equation)
            sol = sympy.dsolve(sympy.expand(equation), nom(t), simplify=False)
            sol = sympy.simplify(sol)
            sol_rhs = sol.rhs
            sol_factor = sol_rhs.as_coeff_mul()[1][0]
            sol_step1 = sol.do(s * (1 / sol_factor))
            sol_step2 = sol_step1.do(sympy.exp(s))
            ode_par = sol.subs(t, 0).subs(new_ode_vars_ic)
            explicit_sol = sympy.solve(sol_step2, nom(t), dict=True)
            for s in explicit_sol:
                s[s.keys()[0]] = s.values()[0].subs({ode_par.lhs: ode_par.rhs})
            solutions.append(explicit_sol)

        for s in solutions:
            for q in s:
                solutions_eq.append(sympy.Eq(q.keys()[0], q.values()[0]))

        eqs_to_evaluate = []
        solutions_ready = []
        for idx, solu in enumerate(solutions_eq):
            sol_ic = solu.subs(cl_values)
            for p in self.model.parameters: sol_ic = sol_ic.subs(p.name, p.value)
            sol_ic_copy = sol_ic
            if sympy.simplify(sol_ic.rhs.subs(sympy.Symbol('t'), 0)) > 0:
                eqs_to_evaluate.append(sol_ic_copy)
                solutions_ready.append(solu)
        self.sol_indep_odes = solutions_ready
        self.eqs_to_plot = eqs_to_evaluate
        return self.sol_indep_odes

    def compare_to_numerical(self, eqs_to_plot, new_ode_variables):
        # for obs in new_ode_variables:
        #     species = [int(s) for s in re.findall(r'\d+', str(obs))]
        #     obs_pattern = self.model.species[species[0]]
        #     for sp in species[1:]:
        #         obs_pattern += self.model.species[sp]
        #     # self.model.add_component(pysb.Observable(str(new_ode_variables[obs]), obs_pattern))
        #     pysb.Observable(str(new_ode_variables[obs]), obs_pattern)
        #
        #     print str(new_ode_variables[obs]), obs_pattern


        plt.figure()
        colors = ["b", "g", "c", "m", "y", "k"]
        x = odesolve(self.model, self.tspan, integrator='vode', with_jacobian=True, rtol=1e-20, atol=1e-20)

        for idx, eq in enumerate(eqs_to_plot):
            print eq.rhs.subs(sympy.Symbol('t'), 0)
            f = sympy.lambdify(sympy.Symbol('t'), eq.rhs, 'numpy')
            plt.plot(self.tspan, x[str(eq.lhs).split('(')[0]], linewidth=3, label=str(eq.lhs).split('(')[0],
                     color=colors[idx])
            plt.plot(self.tspan, f(self.tspan), 'r--', linewidth=5, label=str(eq.lhs).split('(')[0], color=colors[idx])

        plt.legend(loc=0)
        plt.show()

    def get_reactions_from_rules(self):
        return self.reactions_from_rules

    def get_new_ode_units(self):
        return self.new_units

    def get_grouped_odes(self):
        return self.grouped_odes

    def get_indep_var_odes(self):
        return self.indep_variables_odes

    def get_sol_indep_odes(self):
        return self.sol_indep_odes

    def get_mass_conservations(self):
        return self.conservation_laws

    def get_new_ode_vars(self):
        return self.new_ode_variables

    def get_eqs_to_plot(self):
        return self.eqs_to_plot


def run_solve_theo(model, tspan):
    st = TheoSols(model)
    st.sol_theo(tspan)
    return st.get_indep_var_odes()

