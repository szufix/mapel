
from mapel.marriages.cultures.impartial import generate_asymmetric_votes
import mapel.core.features.mallows as ml


def generate_mallows_votes(*args, **kwargs):
    return ml.generate_mallows_votes(*args, **kwargs)


def generate_norm_mallows_votes(num_agents=None,
                                normphi=0.5,
                                weight=0.,
                                **kwargs):

    phi = ml.phi_from_normphi(num_agents, normphi=normphi)

    return generate_mallows_votes(num_agents, num_agents, phi)


def generate_mallows_asymmetric_votes(num_agents: int = None,
                                      phi: float = 0.5,
                                      **kwargs):
    """ Mallows on top of Asymmetric instance """

    votes_left, votes_right = generate_asymmetric_votes(num_agents=num_agents)

    votes_left = mallows_votes(votes_left, phi)
    votes_right = mallows_votes(votes_right, phi)

    return [votes_left, votes_right]


def mallows_vote(vote, phi):
    num_candidates = len(vote)
    raw_vote = generate_mallows_votes(1, num_candidates, phi=phi, weight=0)[0]
    new_vote = [0] * len(vote)
    for i in range(num_candidates):
        new_vote[raw_vote[i]] = vote[i]
    return new_vote


def mallows_votes(votes, phi):
    for i in range(len(votes)):
        votes[i] = mallows_vote(votes[i], phi)
    return votes
