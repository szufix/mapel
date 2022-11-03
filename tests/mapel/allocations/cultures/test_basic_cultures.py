import pytest

import mapel.allocations.cultures.basic_cultures as bcult


class TestBasiCultures:
    
  def test_square_identitity(self):
    tot_ut = 100
    res_cnt = 4
    matrix = bcult.identity_alloct_matrix(4, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, 4, res_cnt)
    self.assert_identity(matrix, tot_ut)

  def test_more_agents_identitity(self):
    tot_ut = 100
    res_cnt = 4
    ag_cnt = 10
    matrix = bcult.identity_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_identity(matrix, tot_ut)

  def test_more_resources_identitity(self):
    tot_ut = 100
    res_cnt = 10
    ag_cnt = 4
    matrix = bcult.identity_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_identity(matrix, tot_ut)
          
  def test_square_uniformity(self):
    tot_ut = 100
    res_cnt = 4
    matrix = bcult.uniformity_alloct_matrix(4, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, 4, res_cnt)
    self.assert_uniformity(matrix, res_cnt, tot_ut)

  def test_more_agents_uniformity(self):
    tot_ut = 100
    res_cnt = 4
    ag_cnt = 10
    matrix = bcult.uniformity_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_uniformity(matrix, res_cnt, tot_ut)

  def test_more_resources_uniformity(self):
    tot_ut = 100
    res_cnt = 10
    ag_cnt = 4
    matrix = bcult.uniformity_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_uniformity(matrix, res_cnt, tot_ut)

  def test_square_separability(self):
    tot_ut = 100
    res_cnt = 4
    ag_cnt = 4
    matrix = bcult.separability_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_separability(matrix, res_cnt, tot_ut)

  def test_more_agents_separability(self):
    tot_ut = 100
    res_cnt = 4
    ag_cnt = 10
    matrix = bcult.separability_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_separability(matrix, res_cnt, tot_ut)

  def test_more_resources_separability(self):
    tot_ut = 100
    res_cnt = 10
    ag_cnt = 4
    matrix = bcult.separability_alloct_matrix(ag_cnt, res_cnt, tot_ut)
    self.assert_matrix_sizes(matrix, ag_cnt, res_cnt)
    self.assert_separability(matrix, res_cnt, tot_ut)

  def assert_matrix_sizes(self, matrix, rows_cnt, cols_cnt):
    assert len(matrix) == rows_cnt, "invalid number of rows"
    for row in matrix:
      assert len(row) == cols_cnt, "invalid number of rows in column"

  def assert_identity(self, matrix, tot_ut):
    for row in matrix:
      for col, entry in enumerate(row):
        if col == 0:
          assert entry == tot_ut, "invalid utility for favorite resource"
        else:
          assert entry == 0, "invalid utility for ignored resource"

  def assert_uniformity(self, matrix, res_cnt, tot_ut):
    for row in matrix:
      for entry in row:
        assert entry == (tot_ut / res_cnt), "invalid utility for resource"

  def assert_separability(self, matrix, res_cnt, tot_ut):
    for rowi, row in enumerate(matrix):
      for col, entry in enumerate(row):
        if col == (rowi % res_cnt):
          assert entry == tot_ut, "invalid utility for favorite resource"
        else:
          assert entry == 0, "invalid utility for ignored resource"

