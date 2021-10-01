#!/usr/bin/env python


from mapel.voting.objects.Instance import Instance


class Graph(Instance):

    def __init__(self, experiment_id, name, model=None, graph=None, num_nodes=None, alpha=None):

        super().__init__(experiment_id, name, model=model, alpha=alpha)

        self.graph = graph
        self.num_nodes = num_nodes
