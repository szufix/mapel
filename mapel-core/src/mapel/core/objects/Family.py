#!/usr/bin/env python

class Family:
    """ Family of elections: a set of elections from the same election culture_id """

    def __init__(self,
                 culture_id: str = None,
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
                 single: bool = False,
                 instance_ids=None,
                 path: dict = None):

        self.family_id = family_id
        self.culture_id = culture_id
        self.params = params
        self.printing_params = params
        self.size = size
        self.label = label
        self.color = color
        self.alpha = alpha
        self.show = show
        self.marker = marker
        self.ms = ms
        self.starting_from = starting_from
        self.single = single
        self.path = path
        self.instance_ids = instance_ids

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
