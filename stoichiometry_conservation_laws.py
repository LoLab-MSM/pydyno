import numpy as np
import sympy
import re
from collections import OrderedDict
from pysb.bng import generate_equations


def stoichiometry_matrix(model):
    generate_equations(model)
    sm = np.zeros((len(model.species), len(model.reactions)))
    for i_s, sp in enumerate(model.species):
        for i_r, r in enumerate(model.reactions):
            sm[i_s][i_r] = r['products'].count(i_s) - r['reactants'].count(i_s)
    return sm


def stoichimetry_matrix_passengers(model, pruned_system):
    generate_equations(model)
    sm = np.zeros((len(pruned_system.keys()), len(model.reactions)))
    for i_s, pa in enumerate(pruned_system.keys()):
        for i_r, r in enumerate(model.reactions):
            if r['rate'] in pruned_system[pa].as_coefficients_dict().keys():
                sm[i_s][i_r] = r['products'].count(pa) - r['reactants'].count(pa)
    return sm


def conservation_laws_values(model, conser_laws):
    if not isinstance(conser_laws, list):
        conser_laws = [conser_laws]

    initial_conditions_expanded = {}
    y0 = np.zeros(len(model.species))
    for cp, value_obj in model.initial_conditions:
        value = value_obj.value
        si = model.get_species_index(cp)
        y0[si] = value

    for spp in range(len(model.species)):
        initial_conditions_expanded['__s%d' % spp] = y0[spp]

    value_constants = {}
    for conser in conser_laws:
        constant_to_solve = [atom for atom in conser.atoms(sympy.Symbol) if re.match(r'[a-d]', str(atom))]
        solution = sympy.solve(conser, constant_to_solve)
        solution_ready = solution[0]
        solution_ready = solution_ready.subs(initial_conditions_expanded)
        value_constants[constant_to_solve[0]] = solution_ready

    return value_constants


def conservation_relations(model, pruned_system=None):
    if pruned_system is not None:
        stoichiometry = stoichimetry_matrix_passengers(model, pruned_system)
        model_species = [model.species[i] for i in pruned_system.keys()]
    else:
        stoichiometry = stoichiometry_matrix(model)
        model_species = model.species

    sto_rank = np.linalg.matrix_rank(stoichiometry)
    species_info = OrderedDict()
    for sp in model_species:
        species_info[str(sp)] = sympy.Symbol('__s%d' % model.get_species_index(sp))
    sto_augmented = np.concatenate((stoichiometry, np.identity(stoichiometry.shape[0])), axis=1)
    sto_augmented = sympy.Matrix(sto_augmented)
    sto_reduced = sto_augmented.rref()[0]
    conservation_matrix = sto_reduced[sto_rank:, stoichiometry.shape[1]:]
    conservation_matrix = conservation_matrix.applyfunc(sympy.Integer)
    conservation_laws = conservation_matrix.dot(species_info.values())
    if not isinstance(conservation_laws, list):
        conservation_laws = [conservation_laws]

    for ii, cl in enumerate(conservation_laws):
        conservation_laws[ii] = cl - sympy.Symbol('a%d' % ii)

    value_constants = conservation_laws_values(model, conservation_laws)

    return conservation_laws, value_constants
