import numpy as np
import pandas as pd
import tellurium
from six import string_types

try:
    import tesbml as libsbml
except ImportError:
    import libsbml


class SbmlModel:
    """
    Class to organize a tellurium model

    Parameters
    ----------
    model : tellurium model
        Model to organize
    """
    def __init__(self, model):
        if isinstance(model, tellurium.roadrunner.extended_roadrunner.ExtendedRoadRunner):
            model_sbml = model.getSBML()
            self.doc = libsbml.readSBMLFromString(model_sbml)
            self._model = self.doc.getModel()
        elif isinstance(model, string_types):
            self.doc = libsbml.readSBMLFromString(model)
            self._model = self.doc.getModel()
        elif isinstance(model, libsbml.SBMLDocument):
            self.doc = model
            self._model = self.doc.getModel()
        elif isinstance(model, libsbml.Model):
            self._model = model
        else:
            raise Exception('SBML Input is not valid')
        self._name = 'sbml_model'
        self._sp_idx_dict = {sp.getId(): idx for idx, sp in enumerate(self.model.getListOfSpecies())}

    @property
    def model(self):
        return self._model

    @property
    def species(self):
        sps = list(self._sp_idx_dict.keys())
        return sps

    @property
    def sp_idx_dict(self):
        return self._sp_idx_dict

    @property
    def reactions_bidirectional(self):
        rxns = []
        for rxn in self.model.getListOfReactions():
            reactants = tuple(self._sp_idx_dict[r.getSpecies()] for r in rxn.getListOfReactants())
            products = tuple(self._sp_idx_dict[p.getSpecies()] for p in rxn.getListOfProducts())
            reversible = (rxn.getReversible())
            rate = rxn.getKineticLaw().getFormula()
            rxn_data = {'reactants': reactants, 'products': products, 'rate': rate, 'reversible': reversible}
            rxns.append(rxn_data)
        return rxns

    @property
    def name(self):
        return self._name


class SbmlSimulation:
    """
    Class to organize tellurium simulations
    """
    def __init__(self):
        self._trajectories = []
        self._reaction_flux_df = []
        self._parameters = []
        self._tspan = []

    @property
    def trajectories(self):
        return np.array(self._trajectories)

    @property
    def reaction_flux_df(self):
        return self._reaction_flux_df

    @property
    def parameters(self):
        return np.array(self._parameters)

    @property
    def tspan(self):
        return np.array(self._tspan)

    def add_simulation(self, sim):
        species = sim.model.getFloatingSpeciesIds()
        reactions = sim.model.getReactionIds()
        required_selections = ['time'] + [s for s in species] + \
                              [r for r in reactions]
        sim_selections = sim.selections

        # Check that the simulation selections include all species and reactions rate values
        if not all(x in required_selections for x in sim_selections):
            raise ValueError('To use this function you must use '
                             'this simulator selection \n {0}'.format(required_selections))
        simulation = sim.getSimulationData()
        trajs = np.zeros((len(simulation['time']), len(species)))

        # Concentration trajectories
        for idx, sp in enumerate(species):
            trajs[:, idx] = simulation[sp]
        self._trajectories.append(trajs)

        # Reaction flux
        rxns_names = ['r{0}'.format(rxn) for rxn in range(len(reactions))]
        rxns_df = pd.DataFrame(columns=simulation['time'], index=rxns_names)
        for idx, rxn in enumerate(reactions):
            rxns_df.loc['r{0}'.format(idx)] = simulation[rxn]
        self._reaction_flux_df.append(rxns_df)

        pars = sim.getGlobalParameterValues()
        self._parameters.append(pars)

        # Simulation time
        self._tspan.append(simulation['time'])

