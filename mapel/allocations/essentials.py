import mapel.main.logs as logs
logger = logs.get_logger(__name__)

from mapel.main.objects.Instance import Instance
from mapel.main.objects.Family import Family
from mapel.main.objects.Experiment import Experiment
import mapel.allocations.metrics.surveying as surveying

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
    

class AllocationTaskFamily(Family):
  
  @classmethod
  def from_one_alloc_task(cls, alloc_task, family_id):
    return AllocationTaskFamily(family_id = family_id, single = True,
    ready_instances = [alloc_task])

  def __init__(self,
             culture_id: str = None,
             family_id='none',
             params: dict = None,
             size: int = 1,
             label: str = "none",
             color: str = "black",
             alpha: float = 1.,
             ms: int = 20,
             show=True,
             marker='o',
             starting_from: int = 0,
             path: dict = None,
             single: bool = False,
             election_ids=None,
             ready_instances = []):

    super().__init__(culture_id=culture_id,
                     family_id=family_id,
                     params=params,
                     size=size,
                     label=label,
                     color=color,
                     alpha=alpha,
                     ms=ms,
                     show=show,
                     marker=marker,
                     starting_from=starting_from,
                     path=path,
                     single=single,
                     instance_ids=election_ids)
    self.allocation_tasks = [] + ready_instances
    self.instance_ids = [t.instance_id for t in ready_instances]


  def __getattr__(self, attr):
      if attr == 'election_ids':
          return self.instance_ids
      else:
          return self.__dict__[attr]

  def __setattr__(self, name, value):
      if name == "election_ids":
          return setattr(self, 'instance_ids', value)
      else:
          self.__dict__[name] = value

  def prepare_family(self, experiment_id=None, store=None,
               store_points=False, aggregated=True):
    return self.allocation_tasks



class AllocationExperiment(Experiment):

    def __init__(self, **kwargs):
        super().__init__(instances = {}, **kwargs)
        self.default_num_candidates = 10
        self.default_num_voters = 100
        self.default_committee_size = 1
        self.all_winning_committees = {}

    def __getattr__(self, attr):
        if attr == 'elections':
            return self.instances
        elif attr == 'num_elections':
            return self.num_instances
        else:
            return self.__dict__[attr]

    def __setattr__(self, name, value):
        if name == "elections":
            self.instances = value
        elif name == "num_elections":
            self.num_instances = value
        else:
            self.__dict__[name] = value


#    def prepare_matrices(self):
#        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "matrices")
#        for file_name in os.listdir(path):
#            os.remove(os.path.join(path, file_name))
#
#        for election_id in self.elections:
#            matrix = self.elections[election_id].votes_to_positionwise_matrix()
#            file_name = election_id + ".csv"
#            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
#                                "matrices", file_name)
#
#            with open(path, 'w', newline='') as csv_file:
#
#                writer = csv.writer(csv_file, delimiter=';')
#                header = [str(i) for i in range(self.elections[election_id].num_candidates)]
#                writer.writerow(header)
#                for row in matrix:
#                    writer.writerow(row)

#    def add_instances_to_experiment(self):
#        instances = {}
#
#        for family_id in self.families:
#            single = self.families[family_id].single
#
#            ids = []
#            for j in range(self.families[family_id].size):
#                instance_id = get_instance_id(single, family_id, j)
#
#                if self.instance_type == 'ordinal':
#                    instance = OrdinalElection(self.experiment_id, instance_id, _import=True,
#                                               fast_import=self.fast_import)
#                elif self.instance_type == 'approval':
#                    instance = ApprovalElection(self.experiment_id, instance_id, _import=True)
#                else:
#                    instance = None
#
#                instances[instance_id] = instance
#                ids.append(str(instance_id))
#
#            self.families[family_id].election_ids = ids
#
#        return instances
#
#    def set_default_num_candidates(self, num_candidates: int) -> None:
#        """ Set default number of candidates """
#        self.default_num_candidates = num_candidates
#
#    def set_default_num_voters(self, num_voters: int) -> None:
#        """ Set default number of voters """
#        self.default_num_voters = num_voters
#
#    def set_default_committee_size(self, committee_size: int) -> None:
#        """ Set default size of the committee """
#        self.default_committee_size = committee_size

#    def add_election(self, culture_id="none", params=None, label=None,
#                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
#                     num_candidates=None, num_voters=None, election_id=None):
#        """ Add election to the experiment """
#
#        if num_candidates is None:
#            num_candidates = self.default_num_candidates
#
#        if num_voters is None:
#            num_voters = self.default_num_voters
#
#        return self.add_family(culture_id=culture_id, params=params, size=size, label=label,
#                               color=color, alpha=alpha, show=show, marker=marker,
#                               starting_from=starting_from, family_id=election_id,
#                               num_candidates=num_candidates, num_voters=num_voters,
#                               single=True)

    def add_family(self, culture_id: str = "none", params: dict = None, size: int = 1,
                   label: str = None, color: str = "black", alpha: float = 1.,
                   show: bool = True, marker: str = 'o', starting_from: int = 0,
                   num_candidates: int = None, num_voters: int = None,
                   family_id: str = None, single: bool = False,
                   path: dict = None,
                   election_id: str = None, 
                   family = None) -> list:

        if family == None:
          raise NotImplementedError

        elif label is None:
            label = family.family_id

        if self.families == None:
          self.families = {}

        self.families[family.family_id] = family
        self.num_families = len(self.families)
        self.num_elections = sum([self.families[family_id].size for family_id in self.families])
        self.main_order = [i for i in range(self.num_elections)]

        new_instances = family.allocation_tasks

        for instance in new_instances:
            self.instances[instance.instance_id] = instance

        return [alloc_task.instance_id for alloc_task in family.allocation_tasks]


    def compute_distances(self, distance_id, self_distances = True):
        matchings = {instance_id: {} for instance_id in self.instances}
        distances = {instance_id: {} for instance_id in self.instances}
        times = {instance_id: {} for instance_id in self.instances}

        ids = []
        for i, instance_1 in enumerate(self.instances):
            for j, instance_2 in enumerate(self.instances):
                if i < j or (i == j and self_distances):
                    ids.append((instance_1, instance_2))
         
        for left_task_id, right_task_id in ids:
          distance, matching = \
          surveying.get_distance(self.instances[left_task_id],
          self.instances[right_task_id], distance_id) 
          distances[left_task_id][right_task_id] = distance
          matchings[left_task_id][right_task_id] = matching

        logger.debug(f"Computed distances:\n{distances}")

        self.distances = distances
        self.times = times
        self.matchings = matchings

    def get_election_id_from_model_name(self, culture_id: str) -> str:
        for family_id in self.families:
            if self.families[family_id].culture_id == culture_id:
                return family_id

#    def compute_feature(self, feature_id: str = None, feature_params=None,
#                        printing=False, **kwargs) -> dict:
#
#        if feature_params is None:
#            feature_params = {}
#
#        if feature_id in ['priceability', 'core', 'ejr']:
#            feature_long_id = f'{feature_id}_{feature_params["rule"]}'
#        else:
#            feature_long_id = feature_id
#
#        num_iterations = 1
#        if 'num_interations' in feature_params:
#            num_iterations = feature_params['num_interations']
#
#        if feature_id == 'ejr':
#            feature_dict = {'value': {}, 'time': {}, 'ejr': {}, 'pjr': {}, 'jr': {}, 'pareto': {}}
#        elif feature_id in FEATURES_WITH_DISSAT:
#            feature_dict = {'value': {}, 'time': {}, 'dissat': {}}
#        else:
#            feature_dict = {'value': {}, 'time': {}}
#
#        if feature_id in MAIN_GLOBAL_FEATUERS or feature_id in ELECTION_GLOBAL_FEATURES:
#
#            feature = features.get_global_feature(feature_id)
#
#            values = feature(self, election_ids=list(self.instances), feature_params=feature_params)
#
#            for instance_id in self.instances:
#                feature_dict['value'][instance_id] = values[instance_id]
#                feature_dict['time'][instance_id] = 0
#
#        else:
#            feature = features.get_local_feature(feature_id)
#
#            for instance_id in self.elections:
#                if printing:
#                    print(instance_id)
#                instance = self.elections[instance_id]
#
#                start = time.time()
#
#                for _ in range(num_iterations):
#
#                    if feature_id in ['monotonicity_1', 'monotonicity_triplets']:
#                        value = feature(self, instance)
#
#                    elif feature_id in {'avg_distortion_from_guardians',
#                                        'worst_distortion_from_guardians',
#                                        'distortion_from_all',
#                                        'distortion_from_top_100'}:
#                        value = feature(self, instance_id)
#                    else:
#                        value = instance.get_feature(feature_id, feature_long_id, **kwargs)
#
#                total_time = time.time() - start
#                total_time /= num_iterations
#
#                if feature_id == 'ejr':
#                    feature_dict['ejr'][instance_id] = int(value['ejr'])
#                    feature_dict['pjr'][instance_id] = int(value['pjr'])
#                    feature_dict['jr'][instance_id] = int(value['jr'])
#                    feature_dict['pareto'][instance_id] = int(value['pareto'])
#                    feature_dict['time'][instance_id] = total_time
#
#                elif feature_id in FEATURES_WITH_DISSAT:
#                    feature_dict['value'][instance_id] = value[0]
#                    feature_dict['time'][instance_id] = total_time
#                    feature_dict['dissat'][instance_id] = value[1]
#                else:
#                    feature_dict['value'][instance_id] = value
#                    feature_dict['time'][instance_id] = total_time
#
#        if self.store:
#            self._store_election_feature(feature_id, feature_long_id, feature_dict)
#
#        self.features[feature_long_id] = feature_dict
#        return feature_dict

