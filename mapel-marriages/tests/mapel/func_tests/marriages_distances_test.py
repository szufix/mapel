import pytest
import numpy as np

import mapel.marriages as mapel

registered_marriages_distances_to_test = {
    'l1-mutual_attraction',
}


class TestMarriagesDistances:

    @pytest.mark.parametrize("distance_id", registered_marriages_distances_to_test)
    def test_marriages_distances(self, distance_id):
        num_agents = int(np.random.randint(5, 50) * 2)

        instance_1 = mapel.generate_marriages_instance(culture_id='ic',
                                                       num_agents=num_agents)

        instance_2 = mapel.generate_marriages_instance(culture_id='ic',
                                                       num_agents=num_agents)

        distance, mapping = mapel.compute_distance(instance_1, instance_2,
                                                   distance_id=distance_id)
        assert type(float(distance)) is float
