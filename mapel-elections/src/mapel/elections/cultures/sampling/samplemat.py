import logging
import numpy as np
try:
  from permanent import permanent
except:
  logging.warning("The 'permanent' module is not installed. Sampling matrices "
  "unavailable.")
import random
import math

import mapel.core.utils as utils

def _input_standarization(matrix):
  try: 
    if type(matrix) == list:
      matrix = np.asarray(matrix, dtype = 'int')
  except Exception as e:
    raise ValueError("Matrix provided for sampling cannot be made a numpy array", e)
  if matrix.shape[0] != matrix.shape[1]:
    raise ValueError("Matrix provided for sampling is not square", e)
  rows_sums = np.sum(matrix, axis = 0)
  cols_sums = np.sum(matrix, axis = 1)
  if not np.array_equal(rows_sums, cols_sums):
    raise ValueError("Matrix provided for sampling is not describing a valid \
    election---the row/column sums are not equal", e)
  return matrix

def _draw_vote(matrix):
  vote_matchings = []
  binary_matrix = np.sign(matrix)
  positions = np.arange(binary_matrix.shape[0])
  positions = positions[np.newaxis,:]
  candidates = np.arange(binary_matrix.shape[0] + 1)
  candidates = candidates[:, np.newaxis]
  binary_matrix = np.append(binary_matrix, positions, axis = 0)
  binary_matrix = np.hstack((binary_matrix, candidates))
  while len(vote_matchings) < len(matrix):
    taken_edge = None
    bin_matrix_view = binary_matrix[:-1,:-1]
    ground_perm = permanent.permanent(bin_matrix_view.astype('complex'))
    cands, poss = bin_matrix_view.nonzero()
    taken_edge = None
    while not taken_edge:
      if bin_matrix_view.shape == (1,1):
        taken_edge = (0,0)
        continue
      random_edge = random.choice(list(zip(cands, poss)))
      bin_matr_indices = range(bin_matrix_view.shape[0])
      cands_to_remain = [i for i in  bin_matr_indices if i!= random_edge[0]]
      pos_to_remain = [i for i in  bin_matr_indices if i!= random_edge[1]]
      without_taken_edge = bin_matrix_view[np.ix_(cands_to_remain,
      pos_to_remain)]
      took_edge_perm = permanent.permanent(without_taken_edge.astype('complex'))
      ratio = float(took_edge_perm.real)/float(ground_perm.real) 
      if math.isclose(ratio, 1.0):
        taken_edge = random_edge  
        continue 
      rand_num = random.random()
      if math.isclose(ratio, rand_num) or rand_num < ratio:
        taken_edge = random_edge
    orig_edge_cand = int(binary_matrix[taken_edge[0]][-1])
    orig_edge_pos = int(binary_matrix[-1][taken_edge[1]])
    vote_matchings.append((orig_edge_cand, orig_edge_pos))
    binary_matrix = np.delete(binary_matrix, taken_edge[0], axis = 0)
    binary_matrix = np.delete(binary_matrix, taken_edge[1], axis = 1)
  return vote_matchings

def sample_election_using_permanent(matrix):
  """
      Samples elections from a given position as follows:
      0. Interpret a given matrix as weighted bipartite graph (weights are
      entries)
      1. Sample uniformly at random a possible vote by sampling a perfect
      matching (that is unweighted)
      2. Subtract the generated vote from the original matrix and repeat step 0
      until the original matrix has only zero entries 

      Arguments:
        matrix: a position matrix either a list of lists or a numpy array

      Returns:
        The list of lists representing the votes realizing the given matrix
  """
  if not utils.is_module_loaded("permanent"):
    raise RuntimeError("Module 'permanent' was not loaded. Sampling elections "
    "from matrix impossible.")
  matrix = _input_standarization(matrix)
  votes_count = np.sum(matrix[0])
  curr_matrix = matrix.copy()
  votes = []
  while len(votes) < votes_count:
    matching = _draw_vote(curr_matrix)
    vote = [0] * len(matching)
    for cand, pos in matching:
      curr_matrix[cand][pos] -= 1
      vote[pos] = cand
    votes.append(vote)
  return votes
