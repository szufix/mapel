from mapel.main.objects.Instance import Instance

class AllocationTask(Instance):
  """
  Represents a single allocation task; in other words, one instance of
  an allocation problem.
  """

  @classmethod
  def from_matrix(cls, utility_matrix, instance_id):
    """
      Constructs an instance from a utility matrix, which is a list of agent
      evaluation functions over all of the resources. Each evaluation function
      is a list of integers.
    """
    return AllocationTask(utility_matrix, None, instance_id)

  @classmethod
  def from_splidditfile(cls, fname, instance_id):
    utility_matrix = []
    with open(fname, 'r') as ffile:
      linecnt = 0
      for line in ffile:
        linecnt += 1
        if linecnt == 1:
          agnt_cnt, res_cnt = (int(val) for val in line.strip().split(" "))
          continue
        if linecnt == 2:
          continue
        if (linecnt > 2) and (linecnt <= 2 + agnt_cnt):
          utility_matrix.append([int(val) for val in line.strip().split()])
        continue
    return AllocationTask(utility_matrix, None, instance_id)

  def __init__(self, utility_matrix, experiment_id, instance_id, culture_id=None, alpha=None):
    super().__init__(experiment_id, instance_id, culture_id=culture_id, alpha=alpha)
    self._validate(utility_matrix)

    self.utility_matrix = utility_matrix
    self.agents_count = len(utility_matrix)
    self.resources_count = len(utility_matrix[0]) 

  def __getitem__(self, idx):
    return self.utility_matrix[idx]

  def _validate(self, utility_matrix):
    if type(utility_matrix) is not list:
      raise ValueError("Allocation task can only be constructed using list of "
      "lists")
    if len(utility_matrix) == 0:
      raise ValueError("An empty allocation task is '[[]]' not '[]'")
    row_size = 0
    for row in utility_matrix:
      if row_size == 0:
        row_size = len(row)
      if type(row) is not list:
        raise ValueError("Allocation task can only be constructed using list of "
        "lists")
      if row_size != len(row):
        raise ValueError("Each row in an allocation task matrix must have the "
        "same number of entries")
      for val in row:
        if type(val) not in (float, int):
          raise ValueError("Each row in an allocation task matrix must be an "
          "int or float")
