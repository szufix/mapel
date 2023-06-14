import numpy as np

import pytest

import mapel.roommates as mapel

registered_roommates_cultures_to_test = {
    'ic',
    'id',
    'chaos',
    'symmetric',
    'asymmetric',
    'urn',
    'fame',
    'expectation',
    'attributes',
    'euclidean',
    'reverse_euclidean',
    'group_ic',
    'norm-mallows',
    'mallows_euclidean',
    'malasym'
}


class TestCultures:

    @pytest.mark.parametrize("culture_id", registered_roommates_cultures_to_test)
    def test_roommates_cultures(self, culture_id):
        num_agents = int(np.random.randint(5, 50) * 2)

        instance = mapel.generate_roommates_instance(culture_id=culture_id,
                                                     num_agents=num_agents)

        assert instance.num_agents == num_agents
