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
        #             if i_s in r['reactants']: sm[i_s][i_r] = -1
        #             elif i_s in r['products']: sm[i_s][i_r] = 1
        #             else: sm[i_s][i_r] = 0
        #
    return sm


def stoichimetry_matrix_passengers(model, pruned_system):
    generate_equations(model)
    sm = np.zeros((len(pruned_system.keys()), len(model.reactions)))
    for i_s, pa in enumerate(pruned_system.keys()):
        for i_r, r in enumerate(model.reactions):
            if r['rate'] in pruned_system[pa].as_coefficients_dict().keys():
                sm[i_s][i_r] = r['products'].count(pa) - r['reactants'].count(pa)
    return sm


def conservation_relations(model, pruned_system=None):
    if pruned_system is not None:
        stoichiometry = stoichimetry_matrix_passengers(model, pruned_system)
        model_species = [model.species[i] for i in pruned_system.keys()]
    else:
        stoichiometry = stoichiometry_matrix(model)
        model_species = model.species

    species_number = {}
    for idx, spx in enumerate(model.species):
        species_number[spx] = idx

    sto_rank = np.linalg.matrix_rank(stoichiometry)
    number_conserved = stoichiometry.shape[0] - sto_rank
    species_info = OrderedDict()
    conservation_laws_sp = [0] * number_conserved
    for i, sp in enumerate(model_species):
        species_info[str(sp)] = sympy.Symbol('__s%d' % species_number[sp])
    sto_augmented = np.concatenate((stoichiometry, np.identity(stoichiometry.shape[0])), axis=1)
    sto_augmented = sympy.Matrix(sto_augmented)
    sto_reduced = sto_augmented.rref()[0]
    conservation_matrix = sto_reduced[sto_rank:, stoichiometry.shape[1]:]
    conservation_laws = conservation_matrix.dot(species_info.values())
    if not isinstance(conservation_laws, list):
        conservation_laws = [conservation_laws]
    for i, j in enumerate(conservation_laws):
        conservation_laws_sp[i] = [int(str(atom).split('__s')[1]) for atom in j.atoms(sympy.Symbol) if
                                   not re.match(r'\d', str(atom))]
    return conservation_laws_sp
