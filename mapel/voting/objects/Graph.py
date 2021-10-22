#!/usr/bin/env python


from mapel.voting.objects.Instance import Instance


class Graph(Instance):

    def __init__(self, experiment_id, election_id, model_id=None, graph=None, num_nodes=None, alpha=None):

        super().__init__(experiment_id, election_id, model_id=model_id, alpha=alpha)

        self.graph = graph
        self.num_nodes = num_nodes
