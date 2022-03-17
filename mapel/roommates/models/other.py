from mapel.main._utils import *
from mapel.roommates.models._utils import convert
from mapel.roommates.models.mallows import mallows_votes


def generate_roommates_malasym_votes(num_agents: int = None, params=None):
    """ Mallows on top of Asymmetric instance """

    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

    votes = mallows_votes(votes, params['phi'])

    return convert(votes)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 16.03.2022 #
# # # # # # # # # # # # # # # #


