import numpy as np
from pysb.bng import generate_equations
import pandas as pd
from pydyno.util import parse_name
from anytree.importer import DictImporter


class FreqAnalysis(object):
    """
    Class to perform frequency analysis on the discretize simulation results
    Parameters
    ----------
    model: PySB model
        Model to analyze
    """

    def __init__(self, model):
        self._model = model
        generate_equations(self._model)

    def convert_names(self, sp_idx):
        """
        Function to parse species names
        Parameters
        ----------
        sp_idx: Model species index

        Returns
        -------
        String with the name of the species
        """
        node_sp = self._model.species[sp_idx]
        node_name = parse_name(node_sp)

        return node_name

    @staticmethod
    def get_path_descendants(path):
        """ Get the set of descendants for a tree like path dict.
        """

        importer = DictImporter()
        root = importer.import_(path)
        descendants = set([descendant_node.name for descendant_node in root.descendants])
        return descendants

    def relative_species_frequency_signatures(self, paths, path_signatures, cluster_labels=None,
                                              accessible_species=None):
        """Compute the relative frequencies of species.

        This function computes the relative frequencies of species amongst the
        dominant paths across all simulatations and timepoints.

        Parameters
        ----------
        paths: dict
            Nested tree structure dict of paths as returned from
            DomPath.get_path_signatures()
        path_signatures: pandas.DataFrame or np.array
            The dominant path signatures for each simulation (across all
            time points).
        model: pysb.Model
            The model that is being used.
        cluster_labels: vector-like
            Array that contains the labels that result from clustering the signatures

        Returns
        -------
            A pd.DataFrame with the species codename
            (i.e. 's' + str( model.species_index)) as the index names and the fraction of dominant
            paths that species was in at each time of the simulation.

        """

        # Check path signatures data structure
        if isinstance(path_signatures, np.ndarray):
            path_signatures_np = path_signatures
        elif isinstance(path_signatures, pd.DataFrame):
            path_signatures_np = path_signatures.values
        else:
            raise TypeError('Data structure not valid for path_signatures')

        if accessible_species is None:
            species_analysis = self._model.species
        else:
            species_analysis = accessible_species

        n_species_analysis = len(species_analysis)
        spec_dict = dict()
        # Creating dictionary with species name and index information
        for i, species in enumerate(species_analysis):
            sname = "s{}".format(i)
            spec_dict[sname] = {'name': species, 'index': i}

        # Creating dictionary of descendants in each dominant path
        path_species = dict()
        for i, key in enumerate(paths.keys()):
            path = paths[key]
            descendants = self.get_path_descendants(path)
            # add the root node to the set of species for the path
            descendants.add(path['name'])
            # print(descendants)
            path_species[i] = descendants

        if cluster_labels is not None:
            # Check that length of cluster labels and simulatons are the same
            if len(cluster_labels) != path_signatures_np.shape[0]:
                raise ValueError('The length of the labels and path signatures must be the same')

            n_clusters = set(cluster_labels)
            # Getting a dataframe with species freq information for each cluster label
            clus_fractions_list = []
            for label in n_clusters:
                label_idxs = np.where(cluster_labels == label)[0]
                cluster_signatures = path_signatures_np[label_idxs]
                clus_fractions = self._obtain_fractions(cluster_signatures,
                                                   path_species, spec_dict, n_species_analysis)
                clus_fractions_list.append(clus_fractions)

            all_fractions = pd.concat(clus_fractions_list, axis=0, keys=['cluster{0}'.format(cl) for cl in n_clusters])
            # Converting species indices to readable names
            all_fractions.index.set_levels(all_fractions.index.levels[1].map(self.convert_names), level=1, inplace=True)
        else:
            all_fractions = self._obtain_fractions(path_signatures_np,
                                              path_species, spec_dict, n_species_analysis)
            all_fractions.rename(self.convert_names, axis='index', inplace=True)

        return all_fractions

    @staticmethod
    def _obtain_fractions(path_signatures_np, path_species, spec_dict, n_species_analysis):
        n_sims = path_signatures_np.shape[0]
        n_tp = path_signatures_np.shape[1]

        time_fractions = pd.DataFrame(index=range(len(spec_dict.keys())), columns=range(n_tp))
        for t in range(n_tp):
            spec_counts = np.array([0.0] * n_species_analysis)
            n_tot = 0.0
            for sim in range(n_sims):
                n_tot += 1.0
                dom_path_id = path_signatures_np[sim][t]
                for descendant in path_species[dom_path_id]:
                    d_id = spec_dict[descendant]['index']
                    spec_counts[d_id] += 1.0
            spec_fracs = spec_counts / n_tot
            time_fractions[t] = spec_fracs

        return time_fractions

    def relative_species_frequency_paths(self, paths, accessible_species=None):
        """Compute the relative frequencies of species.

        Computes the relative frequencies of species in the dominant paths.

        Parameters
        ----------
        paths: dict
            Nested tree structure dict of paths as returned from
                DomPath.get_path_signatures()

        Returns
        -------
        A list of tuples with the species codename
            (i.e. 's' + str( model.species_index)) and the fraction of dominant
            paths that species was in.

        """

        if accessible_species is None:
            species_analysis = self._model.species
        else:
            species_analysis = accessible_species

        n_species_analysis = len(species_analysis)
        spec_dict = dict()
        # Creating dictionary with species name and index information
        for i, species in enumerate(species_analysis):
            sname = "s{}".format(i)
            spec_dict[sname] = {'name': species, 'index': i}

        spec_counts = np.array([0.0] * n_species_analysis)

        path_species = dict()
        for i, key in enumerate(paths.keys()):
            path = paths[key]
            descendants = self.get_path_descendants(path)
            # add the root node to the set of species for the path
            descendants.add(path['name'])
            # print(descendants)
            path_species[i] = descendants

        # print(n_sims, n_tp)
        # quit()
        n_tot = 0.0
        for key in path_species:
            n_tot += 1.0
            for descendant in path_species[key]:
                #    print(descendant)
                d_id = spec_dict[descendant]['index']
                spec_counts[d_id] += 1.0
        # print(n_tot)
        spec_fracs = spec_counts / n_tot

        column_names = [self.convert_names(i) for i in range(n_species_analysis)]
        spec_fracs_df = pd.DataFrame(spec_fracs, index=column_names, columns=['Fractions'])

        return spec_fracs_df

    @staticmethod
    def relative_path_frequency_signatures(paths, path_signatures):
        """Compute the relative frequencies of paths.

        This function computes the relative frequencies of dominant paths across
        all simulatations and timepoints.

        Parameters
        ----------
        paths: dict
            Nested tree structure dict of paths as returned from
            DomPath.get_path_signatures()
        path_signatures: pandas.DataFrame
            The dominant path signatures for each simulation (across all
            time points).

        Returns
        -------
            A list of tuples with the species codename
            (i.e. 's' + str( model.species_index)) and the fraction of dominant
            paths that species was in.

        """

        # Check path signatures data structure
        if isinstance(path_signatures, np.ndarray):
            path_signatures_np = path_signatures
        elif isinstance(path_signatures, pd.DataFrame):
            path_signatures_np = path_signatures.values
        else:
            raise TypeError('Data structure not valid for path_signatures')

        n_sims = path_signatures_np.shape[0]
        n_tp = path_signatures_np.shape[1]

        time_paths_fractions = pd.DataFrame(index=range(len(paths)), columns=range(n_tp))
        for t in range(n_tp):
            path_counts = np.zeros(len(paths))
            n_tot = 0.0
            for sim in range(n_sims):
                n_tot += 1.0
                dom_path_id = path_signatures_np[sim][t]
                path_counts[dom_path_id] += 1.0
            path_fracs = path_counts / n_tot
            time_paths_fractions[t] = path_fracs

        return time_paths_fractions
