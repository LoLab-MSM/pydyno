import mt1_mmp_model
import pysb.bng
import sympy
from stoichiometry_conservation_laws import conservation_relations

model = mt1_mmp_model.return_model('original')

pysb.bng.generate_equations(model)

# this gets the conservation laws (cl) and their constant values (cl_values)
cl, cl_values = conservation_relations(model)

# this loops over the model reactions and storage in rcts_rules the reverse and no reverse reaction
# for each rule
rcts_rules = {}
for rule in model.rules:
    rcts_no_reverse = 0
    rcts_reverse = 0
    for reaction in model.reactions:
        if reaction['rule'][0] == rule.name and reaction['reverse'][0] == False:
            rcts_no_reverse += reaction['rate']
        if reaction['rule'][0] == rule.name and reaction['reverse'][0] == True:
            rcts_reverse += reaction['rate']
    rcts_rules[rule.name] = {'no_reverse': sympy.factor(rcts_no_reverse), 'reverse': sympy.factor(rcts_reverse)}

# this line loops over the reactions and factor the monomials and that way  we get the new units
new_units = {}
for rc in rcts_rules:
    var = sympy.simplify(sympy.factor(rcts_rules[rc]['no_reverse'])).as_coeff_mul()[1]
    final_vars = [ex for ex in var if (str(ex).startswith('__') or type(ex) == sympy.Pow or type(ex) == sympy.Add)]
    new_units[rc] = final_vars

# this defines the odes for the new units
new_units_odes = {}
for r in new_units:
    for eq in new_units[r]:
        if type(eq) == sympy.Pow:
            eq_name = eq.args[0]
            new_units_odes[eq_name] = sympy.simplify(rcts_rules[r]['reverse'] + rcts_rules[r]['no_reverse'])
        else:
            new_units_odes[eq] = rcts_rules[r]['reverse'] + rcts_rules[r]['no_reverse']

# this adds more conservation laws from the reactions
for idx, nam in enumerate(new_units):
    if len(new_units[nam]) > 1:
        cl.append(new_units[nam][0] - new_units[nam][1] - sympy.symbols('b%d' % idx, real=True))

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
            if sympy.solve(cc, rcts_rules[r]['reverse'].as_two_terms()[1]):
                equal_const[rcts_rules[r]['reverse'].as_two_terms()[1]] = sympy.collect_const(
                        sympy.solve(cc, rcts_rules[r]['reverse'].as_two_terms()[1])[0])
            if sympy.solve(sympy.collect_const(cc), rcts_rules[r]['reverse'].as_two_terms()[1]):
                equal_const[rcts_rules[r]['reverse'].as_two_terms()[1]] = sympy.collect_const(
                        sympy.solve(sympy.collect_const(cc), rcts_rules[r]['reverse'].as_two_terms()[1])[0].evalf())

equal_const_units = equal_units.copy()
equal_const_units.update(equal_const)

# this is a dictionary where the keys are the new units and the values are dictionaries that contain all the
# information to define that equation in terms of una variable.
variables_to_change = {}
for unit in new_units:
    if len(new_units[unit]) > 1:
        for idx in range(len(new_units[unit])):
            if rcts_rules[unit]['reverse'] != 0:
                tmp_dict = {new_units[unit][idx]: equal_const_units[new_units[unit][idx]],
                            rcts_rules[unit]['reverse'].as_two_terms()[1]: equal_const_units[
                                rcts_rules[unit]['reverse'].as_two_terms()[1]]}
            else:
                tmp_dict = {new_units[unit][idx]: equal_const_units[new_units[unit][idx]]}
            variables_to_change[new_units[unit][1 - idx]] = tmp_dict
    else:
        tmp_dict = {rcts_rules[unit]['reverse'].as_two_terms()[1]: equal_const_units[
            rcts_rules[unit]['reverse'].as_two_terms()[1]]}
        variables_to_change[new_units[unit][0].as_base_exp()[0]] = tmp_dict

# this is a simple change of variable to allow sympy to solve the differential equations.
new_ode_vars = {}
final_odes = {}
for num, ode in enumerate(new_units_odes):
    new_ode_vars[ode] = sympy.symbols('U%d' % num)
    final_odes[sympy.symbols('U%d' % num)] = new_units_odes[ode].subs(variables_to_change[ode]).subs(ode, sympy.symbols(
            'U%d' % num))

# solving the differential equations
for nom in final_odes:
    t = sympy.symbols('t')
    equation = sympy.Eq(final_odes[nom].subs(nom, nom(t)), nom(t).diff(t))
    print sympy.dsolve(equation, nom(t))
