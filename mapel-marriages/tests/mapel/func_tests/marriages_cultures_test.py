import numpy as np

import pytest

import mapel.marriages as mapel

registered_marriages_cultures_to_test = {
    'ic',
    'id',
    'symmetric',
    'norm-mallows',
    'urn',
    'group_ic',
    'malasym',
    'asymmetric',
    'euclidean',
    'reverse_euclidean',
    'mallows_euclidean',
    'expectation',
    'attributes',
    'fame'
}


class TestCultures:

    @pytest.mark.parametrize("culture_id", registered_marriages_cultures_to_test)
    def test_marriages_cultures(self, culture_id):
        num_agents = int(np.random.randint(5, 50) * 2)

        instance = mapel.generate_marriages_instance(culture_id=culture_id,
                                                     num_agents=num_agents)

        assert instance.num_agents == num_agents
