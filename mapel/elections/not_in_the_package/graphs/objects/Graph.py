
import numpy as np

from mapel.main.objects.Instance import Instance


class Graph(Instance):

    def __init__(self, experiment_id, instance_id, alpha=1, model_id=None, edges=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, model_id=model_id)

        self.edges = edges
        self.num_nodes = len(edges)


