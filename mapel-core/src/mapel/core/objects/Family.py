#!/usr/bin/env python


class Family:
    """ Family of elections: a set of instances from the same culture """

    def __init__(self,
                 culture_id: str = None,
                 family_id: str = 'none',
                 params: dict = None,
                 size: int = 1,
                 label: str = None,
                 color: str = "black",
                 alpha: float = 1.,
                 ms: int = 20,
                 show: bool = True,
                 marker: str = 'o',
                 starting_from: int = 0,
                 single: bool = False,
                 instance_ids: list = None,
                 path: dict = None,
                 **kwargs):

        if path is None:
            path = {}
        if params is None:
            params = {}
        if label is None:
            label = family_id
        if instance_ids is None:
            instance_ids = {}

        self.family_id = family_id
        self.culture_id = culture_id
        self.params = params
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
# LAST CLEANUP ON: 11.07.2023 #
# # # # # # # # # # # # # # # #
