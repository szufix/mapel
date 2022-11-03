import mapel.main.logs as logs
logger = logs.get_logger(__name__)
from mapel.allocations.core.alloctask import AllocationTask

def identity_alloc_task(agents_cnt, resources_cnt, total_utility = 100, instance_id = "ID"):
  logger.debug("Creating the ID allocation task")  
  alloct_matrix  = [([100] + [0]*(resources_cnt-1)) for _ in range(agents_cnt)]
  return AllocationTask.from_matrix(alloct_matrix, instance_id)

def uniformity_alloc_task(agents_cnt, resources_cnt, total_utility = 100, instance_id = "UN"):
  import random
  logger.debug("Creating the UN allocation task")  
  util_per_res_floor = total_utility / resources_cnt
  remaining_util = total_utility - (util_per_res_floor * resources_cnt)
  alloct_matrix = []
  for _ in range(agents_cnt):
    alloct_matrix.append([util_per_res_floor]*resources_cnt)
    if remaining_util > 0:
      agents_with_margin = random.sample(range(resources_cnt), remaining_util)
      for a in agents_with_margin:
        alloct_matrix[-1][a] += 1
  return AllocationTask.from_matrix(alloct_matrix, instance_id)

def separability_alloc_task(agents_cnt, resources_cnt, total_utility = 100, instance_id = "SEP"):
  logger.debug("Creating the SEP allocation task")  
  alloct_matrix = []
  for a in range(agents_cnt):
    preceeding_zeros = a % resources_cnt
    trailing_zeros = resources_cnt - preceeding_zeros - 1
    alloct_matrix.append([0]*preceeding_zeros + [total_utility] + [0]*trailing_zeros)
  return AllocationTask.from_matrix(alloct_matrix, instance_id)
