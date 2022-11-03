import pytest

from mapel.allocations.core.alloctask import AllocationTask
import mapel.allocations.essentials as ess


class TestClass:

  def test_sanity(self):
    alloc_task = AllocationTask.from_matrix([[1,0,999,0], [0,1,0,999]], "test\
    instance")
    alloc_task2 = AllocationTask.from_matrix([[1,0,999,0], [0,1,0,999]], "test\
    instance 2")
    task_family = ess.AllocationTaskFamily.from_one_alloc_task(alloc_task,
    "test_family")
    task_family2  = ess.AllocationTaskFamily.from_one_alloc_task(alloc_task2,
    "test_family 2")
    experiment = ess.AllocationExperiment(experiment_id = "test_experiment")
    experiment.add_family(family = task_family)
    experiment.add_family(family = task_family2)
    assert experiment.num_families == 2, "abc"
