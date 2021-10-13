#!/usr/bin/env python

from mapel.voting.objects.OrdinalElection import OrdinalElection

try:
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.manifold import Isomap
except ImportError as error:
    MDS = None
    TSNE = None
    SpectralEmbedding = None
    LocallyLinearEmbedding = None
    Isomap = None
    print(error)

from mapel.voting.objects.ElectionExperiment import ElectionExperiment


class OrdinalElectionExperiment(ElectionExperiment):
    """Abstract set of elections."""

    def __init__(self, ignore=None, elections=None, distances=None, with_matrices=False,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 election_type='ordinal', attraction_factor=1, ):
        super().__init__(ignore=ignore, elections=elections, distances=distances,
                         with_matrices=with_matrices,
                         coordinates=coordinates, distance_name=distance_name,
                         experiment_id=experiment_id,
                         election_type=election_type, attraction_factor=attraction_factor, )

    def add_elections_to_experiment(self, with_matrices=False):
        """ Import elections from a file """

        elections = {}

        for family_id in self.families:

            ids = []
            if self.families[family_id].single_election:
                election_id = family_id
                election = OrdinalElection(self.experiment_id, election_id,
                                           with_matrix=with_matrices)
                elections[election_id] = election
                ids.append(str(election_id))
            else:
                for j in range(self.families[family_id].size):
                    election_id = family_id + '_' + str(j)
                    election = OrdinalElection(self.experiment_id, election_id,
                                               with_matrix=with_matrices)
                    elections[election_id] = election
                    ids.append(str(election_id))

            self.families[family_id].election_ids = ids

        return elections
