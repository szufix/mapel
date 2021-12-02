#!/usr/bin/env python


from mapel.main.objects.Family import Family


class ElectionFamily(Family):
    """ Family of elections: a set of elections from the same election model_id """

    def __init__(self,
                 model_id: str = None,
                 family_id='none',
                 params: dict = None,
                 size: int = 1,
                 label: str = "none",
                 color: str = "black",
                 alpha: float = 1.,
                 ms: int = 20,
                 show=True,
                 marker='o',
                 starting_from: int = 0,
                 path: dict = None,
                 single_instance: bool = False,

                 num_candidates=None,
                 num_voters=None,
                 single_election: bool = False,
                 election_ids=None,
                 ballot: str = 'ordinal'):

        super().__init__(model_id=model_id,
                         family_id=family_id,
                         params=params,
                         size=size,
                         label=label,
                         color=color,
                         alpha=alpha,
                         ms=ms,
                         show=show,
                         marker=marker,
                         starting_from=starting_from,
                         path=path,
                         single_instance=single_instance)

        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.single_election = single_election
        self.election_ids = election_ids
        self.ballot = ballot

    def __getattr__(self, attr):
        if attr == 'election_ids':
            return self.instance_ids
        else:
            return getattr(self, attr)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
