""" This module contains all the objects """

import os
import math
#from dataclasses import dataclass

import csv
import numpy as np


def import_controllers_meta(experiment_id):
    info = {}
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "basic", 'meta.csv')
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            info[row['key']] = row['value']

    num_voters = int(info['num_voters'])
    num_candidates = int(info['num_candidates'])
    num_families = int(info['num_families'])
    num_elections = int(info['num_elections'])
    return num_voters, num_candidates, num_families, num_elections


class Model:
    """Abstract model of elections."""

    def __init__(self, experiment_id, ignore=None, main_order_name="default",
                 distance_name="positionwise", metric_name="emd", num_elections=None):

        self.experiment_id = experiment_id
        self.num_voters, self.num_candidates, self.num_families, self.num_elections =\
            import_controllers_meta(experiment_id)
        if num_elections:
            self.num_elections = num_elections
        self.main_order = self.import_order(main_order_name)

        self.distance_name = distance_name
        self.metric_name = metric_name

        self.families = self.import_controllers(ignore=ignore)

        self.elections = self.add_elections_to_model()

    def add_elections_to_model(self):
        elections = []
        for i in range(self.num_elections):
            election_id = 'core_' + str(i)
            election = Election(self.experiment_id, election_id)
            elections.append(election)
        return elections

    def import_controllers(self, ignore=None):
        """Import from a file all the controllers."""

        families = []

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "controllers", "basic", 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(',')]
        reader = csv.DictReader(file_, fieldnames=header)

        starting_from = 0
        for row in reader:

            election_model = None
            color = None
            label = None
            param_1 = None
            param_2 = None
            alpha = None
            size = None
            marker = None
            num_candidates = None

            if 'election_model' in row.keys():
                election_model = str(row['election_model']).strip()

            if 'color' in row.keys():
                color = str(row['color']).strip()

            if 'label' in row.keys():
                label = str(row['label'])

            if 'param_1' in row.keys():
                param_1 = float(row['param_1'])

            if 'param_2' in row.keys():
                param_2 = float(row['param_2'])

            if 'alpha' in row.keys():
                alpha = float(row['alpha'])

            if 'family_size' in row.keys():
                size = int(row['family_size'])

            if 'marker' in row.keys():
                marker = str(row['marker']).strip()

            if 'num_candidates' in row.keys():
                num_candidates = int(row['num_candidates'])

            show = True
            if row['show'].strip() != 't':
                show = False

            families.append(Family(election_model=election_model, param_1=param_1, param_2=param_2, label=label,
                                   color=color, alpha=alpha, show=show, size=size, marker=marker,
                                   starting_from=starting_from, num_candidates=num_candidates))
            starting_from += size

        if ignore is None:
            ignore = []

        ctr = 0
        for i in range(self.num_families):
            resize = 0
            for j in range(families[i].size):
                if self.main_order[ctr] >= self.num_elections or self.main_order[ctr] in ignore:
                    resize += 1
                ctr += 1
            families[i].size -= resize

        file_.close()
        return families

    def import_order(self, main_order_name):
        """ Import from a file precomputed order of all the elections """

        if main_order_name == 'default':
            main_order = [i for i in range(self.num_elections)]

        else:
            file_name = os.path.join(os.getcwd(), "experiments", self.experiment_id, "results", "orders", main_order_name + ".txt")
            file_ = open(file_name, 'r')
            file_.readline()  # skip this line
            all_elections = int(file_.readline())
            file_.readline()  # skip this line
            main_order = []

            for w in range(all_elections):
                main_order.append(int(file_.readline()))

        return main_order


class Model_xd(Model):
    """ Multi-dimensional model of elections """

    def __init__(self, experiment_id, distance_name='positionwise', metric_name='emd'):

        Model.__init__(self, experiment_id, distance_name=distance_name, metric_name=metric_name)

        #self.num_points, self.num_distances, self.distances = self.import_distances(experiment_id, metric)
        self.num_distances, self.distances = self.import_distances(experiment_id, distance_name)

    @staticmethod
    def import_distances(experiment_id, metric):
        """Import from a file precomputed distances between each pair of elections  """

        file_name = f"{metric}.txt"
        path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", file_name)
        file_ = open(path, 'r')
        num_points = int(file_.readline())
        file_.readline()  # skip this line
        num_distances = int(file_.readline())

        hist_data = [[0 for _ in range(num_points)] for _ in range(num_points)]

        for a in range(num_points):
            for b in range(a + 1, num_points):
                line = file_.readline()
                line = line.split(' ')
                hist_data[a][b] = float(line[2])
                hist_data[b][a] = hist_data[a][b]

        #return num_points, num_distances, hist_data
        return num_distances, hist_data


class Model_2d(Model):
    """ Two-dimensional model of elections """

    """
    experiment_id: str
    main_order_name: str = "default"
    metric: str = "positionwise"
    ignore: [] = None
    num_elections: int = None
    magic: int = 1
    """

    def __init__(self, experiment_id, main_order_name="default", distance_name="positionwise", ignore=None,
                 num_elections=None, attraction_factor=1):

        Model.__init__(self, experiment_id, ignore=ignore, main_order_name=main_order_name, distance_name=distance_name,
                       num_elections=num_elections)

        self.attraction_factor = int(attraction_factor)

        self.num_points, self.points, = self.import_points(ignore=ignore)
        self.points_by_families = self.compute_points_by_families()

    def import_points(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []

        points = []
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "controllers",
                            "points", self.distance_name + "_2d_a" + str(int(self.attraction_factor)) + ".csv")

        with open(path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            ctr = 0
            print(path)
            for row in reader:
                if self.main_order[ctr] < self.num_elections and self.main_order[ctr] not in ignore:
                    points.append([float(row['x']), float(row['y'])])
                ctr += 1

        return len(points), points

    def compute_points_by_families(self):
        """ Group all points by their families """

        points_by_families = [[[] for _ in range(2)] for _ in range(self.num_points)]
        ctr = 0

        for i in range(self.num_families):
            for j in range(self.families[i].size):
                points_by_families[i][0].append(self.points[ctr][0])
                points_by_families[i][1].append(self.points[ctr][1])
                ctr += 1

        return points_by_families

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.points[i][d] - self.points[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle):
        """ Rotate all the points by a given angle """

        for i in range(self.num_points):
            self.points[i][0], self.points[i][1] = self.rotate_point(0.5, 0.5, angle, self.points[i][0], self.points[i][1])

        self.points_by_families = self.compute_points_by_families()

    def reverse(self):
        """ Reverse all the points"""

        for i in range(self.num_points):
            self.points[i][0] = self.points[i][0]
            self.points[i][1] = -self.points[i][1]

        self.points_by_families = self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        if self.attraction_factor == 1:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "controllers",
                                "points", self.distance_name + "_2d.csv")
        else:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "controllers",
                                "points", self.distance_name + "_2d_p" + str(self.attraction_factor) + ".csv")

        with open(path, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "x", "y"])

            for i in range(self.num_points):
                x = round(self.points[i][0], 5)
                y = round(self.points[i][1], 5)
                writer.writerow([i, x, y])

    @staticmethod
    def rotate_point(cx, cy, angle, px, py):
        """ Rotate two-dimensional point by an angle """

        s = math.sin(angle)
        c = math.cos(angle)
        px -= cx
        py -= cy
        x_new = px * c - py * s
        y_new = px * s + py * c
        px = x_new + cx
        py = y_new + cy

        return px, py


class Model_3d(Model):
    """ Two-dimensional model of elections """

    def __init__(self, experiment_id, main_order_name="default", distance_name="positionwise", ignore=None,
                 num_elections=None, attraction_factor=1):

        Model.__init__(self, experiment_id, ignore=ignore, main_order_name=main_order_name, distance_name=distance_name,
                       num_elections=num_elections)

        self.attraction_factor = int(attraction_factor)

        self.num_points, self.points, = self.import_points(ignore=ignore)
        self.points_by_families = self.compute_points_by_families()

    def import_points(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []

        points = []
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "controllers",
                            "points", f"{self.distance_name}_3d_a" + f"{int(self.attraction_factor)}.csv")

        print(self.main_order[0])

        with open(path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            ctr = 0
            print(path)
            for row in reader:
                if self.main_order[ctr] < self.num_elections and self.main_order[ctr] not in ignore:
                    points.append([float(row['x']), float(row['y']), float(row['z'])])
                ctr += 1

        return len(points), points

    def compute_points_by_families(self):
        """ Group all points by their families """

        points_by_families = [[[] for _ in range(3)] for _ in range(self.num_points)]
        ctr = 0

        for i in range(self.num_families):
            for j in range(self.families[i].size):
                points_by_families[i][0].append(self.points[ctr][0])
                points_by_families[i][1].append(self.points[ctr][1])
                points_by_families[i][2].append(self.points[ctr][2])
                ctr += 1

        return points_by_families

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.points[i][d] - self.points[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle):
        """ Rotate all the points by a given angle """

        for i in range(self.num_points):
            self.points[i][0], self.points[i][1] = self.rotate_point(0.5, 0.5, angle, self.points[i][0], self.points[i][1])

        self.points_by_families = self.compute_points_by_families()

    def reverse(self, ):
        """ Reverse all the points"""

        for i in range(self.num_points):
            self.points[i][0] = self.points[i][0]
            self.points[i][1] = -self.points[i][1]

        self.points_by_families = self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        file_name = self.experiment_id + ".txt"
        path = os.path.join(os.getcwd(), "results", "points", file_name)
        file_ = open(path, 'w')
        file_.write(str(self.num_points) + "\n")

        for i in range(self.num_points):
            x = round(self.points[i][0], 5)
            y = round(self.points[i][1], 5)
            file_.write(str(x) + ', ' + str(y) + "\n")
        file_.close()

    @staticmethod
    def rotate_point(cx, cy, angle, px, py):
        """ Rotate two-dimensional point by angle """

        s = math.sin(angle)
        c = math.cos(angle)
        px -= cx
        py -= cy
        x_new = px * c - py * s
        y_new = px * s + py * c
        px = x_new + cx
        py = y_new + cy

        return px, py


class Family:
    """ Family of elections """

    def __init__(self, election_model="none", param_1=0., param_2=0., size=0, label="none",
                 color="black", alpha=1., show=True, marker='o', starting_from=0,
                 num_candidates=None):

        self.election_model = election_model
        self.param_1 = param_1
        self.param_2 = param_2
        self.size = size
        self.label = label
        self.color = color
        self.alpha = alpha
        self.show = show
        self.marker = marker
        self.starting_from = starting_from
        self.num_candidates = num_candidates


class Election:

    def __init__(self, experiment_id, election_id):

        self.experiment_id = experiment_id
        self.election_id = election_id

        self.fake = check_if_fake(experiment_id, election_id)

        if self.fake:
            self.fake_model_name, self.fake_param, self.num_voters, self.num_candidates = import_fake_elections(experiment_id, election_id)
        else:
            self.votes, self.num_voters, self.num_candidates = import_soc_elections(experiment_id, election_id)
            self.potes = self.votes_to_potes()

        #self.num_elections = 1  # do poprawy


    def votes_to_potes(self):
        potes = [[-1 for _ in range(self.num_candidates)] for _ in range(self.num_voters)]
        for i in range(self.num_voters):
            for j in range(self.num_candidates):
                potes[i][self.votes[i][j]] = j
        return potes

    def votes_to_positionwise_vectors(self):

        vectors = [[0. for _ in range(self.num_candidates)] for _ in range(self.num_candidates)]

        if self.fake:

            if self.fake_model_name in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                vectors = get_fake_vectors_single(self.fake_model_name, self.num_candidates)
            elif self.fake_model_name in {'unid', 'anid', 'stid', 'anun', 'stun', 'stan'}:
                vectors = get_fake_vectors_convex(self.fake_model_name, self.num_candidates, self.fake_param)
            elif self.fake_model_name == 'crate':
                vectors = get_fake_vectors_crate(self.fake_model_name, self.num_candidates, self.fake_param)

        else:
            for i in range(self.num_voters):
                pos = 0
                for j in range(self.num_candidates):
                    vote = self.votes[i][j]
                    if vote == -1:
                        continue
                    vectors[vote][pos] += 1
                    pos += 1
            for i in range(self.num_candidates):
                for j in range(self.num_candidates):
                    vectors[i][j] /= float(self.num_voters)

        return vectors

    def vector_to_interval(self, vector, precision=None):
        # discreet version for now
        interval = []
        w = int(precision / self.num_candidates)
        for i in range(self.num_candidates):
            for j in range(w):
                interval.append(vector[i]/w)
        return interval

    def votes_to_positionwise_intervals(self, precision=None):

        vectors = self.votes_to_positionwise_vectors()
        intervals = []

        for i in range(len(vectors)):
            intervals.append(self.vector_to_interval(vectors[i], precision=precision))

        return intervals

    def votes_to_bordawise_vector(self):

        if self.fake:
            print('hello')
            print(self.fake_model_name)
            if self.fake_model_name == 'antagonism':
                return [800.0, 800.0, 800.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0, 0.0], 10


        c = self.num_candidates
        v = self.num_voters
        vectors = self.votes_to_positionwise_vectors()
        borda_vector = [sum([vectors[j][i] * (c - i - 1) for i in range(c)])*v for j in range(self.num_candidates)]
        borda_vector = sorted(borda_vector, reverse=True)
        #print(borda_vector)
        return borda_vector, len(borda_vector)

    """
    def votes_to_bordawise_vector(self):

        vector = get_borda_points(self.votes, self.num_voters, self.num_candidates)
        vector = sorted(vector, reverse=True)
        return vector, len(vector)
    """

    def votes_to_bordawise_vector_long_empty(self):

        num_possible_scores = 1 + self.num_voters*(self.num_candidates - 1)

        if self.votes[0][0] == -1:

            vector = [0 for _ in range(num_possible_scores)]
            peak = sum([i for i in range(self.num_candidates)]) * float(self.num_voters) / float(self.num_candidates)
            vector[int(peak)] = self.num_candidates

        else:

            vector = [0 for _ in range(num_possible_scores)]
            points = get_borda_points(self.votes, self.num_voters, self.num_candidates)
            for i in range(self.num_candidates):
                vector[points[i]] += 1

        return vector, num_possible_scores




def import_soc_elections(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", "soc_original", file_name)
    my_file = open(path, 'r')

    first_line = my_file.readline()
    if first_line[0] != '#':
        num_candidates = int(first_line)
    else:
        num_candidates = int(my_file.readline())

    for _ in range(num_candidates):
        my_file.readline()

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])
    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    it = 0
    for j in range(num_options):
        line = my_file.readline().rstrip("\n").split(',')
        quantity = int(line[0])

        for k in range(quantity):
            for l in range(num_candidates):
                votes[it][l] = int(line[l + 1])
            it += 1

    return votes, num_voters, num_candidates


def get_borda_ranking(votes, num_voters, num_candidates):
    points = [0 for _ in range(num_candidates)]
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = (len(scoring) - float(i) - 1.) / (len(scoring) - 1.)

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    candidates = [x for x in range(num_candidates)]
    ordered_candidates = [x for _, x in sorted(zip(points, candidates), reverse=True)]

    return ordered_candidates


def get_borda_points(votes, num_voters, num_candidates):

    points = [0 for _ in range(num_candidates)]
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = len(scoring) - i - 1

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    return points

# NEW 22.11.2020
def check_if_fake(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", "soc_original", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    if line[0] == '$':
        return True
    return False


def import_fake_elections(experiment_id, election_id):

    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", "soc_original", file_name)
    my_file = open(path, 'r')
    my_file.readline()  # line with $ fake

    num_voters = int(my_file.readline().strip())
    num_candidates = int(my_file.readline().strip())
    fake_model_name = str(my_file.readline().strip())
    if fake_model_name == 'crate':
        fake_param = []
        fake_param.append(float(my_file.readline().strip()))
        fake_param.append(float(my_file.readline().strip()))
        fake_param.append(float(my_file.readline().strip()))
        fake_param.append(float(my_file.readline().strip()))
    else:
        fake_param = float(my_file.readline().strip())

    return fake_model_name, fake_param, num_voters, num_candidates


def get_fake_vectors_single(fake_model_name, num_candidates):

    vectors = np.zeros([num_candidates, num_candidates])

    if fake_model_name == 'identity':
        for i in range(num_candidates):
            vectors[i][i] = 1

    elif fake_model_name == 'uniformity':
        for i in range(num_candidates):
            for j in range(num_candidates):
                vectors[i][j] = 1. / num_candidates

    elif fake_model_name == 'stratification':
        half = int(num_candidates/2)
        for i in range(half):
            for j in range(half):
                vectors[i][j] = 1. / half
        for i in range(half, num_candidates):
            for j in range(half, num_candidates):
                vectors[i][j] = 1. / half

    elif fake_model_name == 'antagonism':
        for i in range(num_candidates):
            for _ in range(num_candidates):
                vectors[i][i] = 0.5
                vectors[i][num_candidates - i - 1] = 0.5

    return vectors


def get_fake_vectors_crate(fake_model_name, num_candidates, fake_param):

    base_1 = get_fake_vectors_single('uniformity', num_candidates)
    base_2 = get_fake_vectors_single('identity', num_candidates)
    base_3 = get_fake_vectors_single('antagonism', num_candidates)
    base_4 = get_fake_vectors_single('stratification', num_candidates)

    return crate_combination(base_1, base_2, base_3, base_4, length=num_candidates, alpha=fake_param)


def get_fake_vectors_convex(fake_model_name, num_candidates, fake_param):

    if fake_model_name == 'unid':
        base_1 = get_fake_vectors_single('uniformity', num_candidates)
        base_2 = get_fake_vectors_single('identity', num_candidates)
    elif fake_model_name == 'anid':
        base_1 = get_fake_vectors_single('antagonism', num_candidates)
        base_2 = get_fake_vectors_single('identity', num_candidates)
    elif fake_model_name == 'stid':
        base_1 = get_fake_vectors_single('stratification', num_candidates)
        base_2 = get_fake_vectors_single('identity', num_candidates)
    elif fake_model_name == 'anun':
        base_1 = get_fake_vectors_single('antagonism', num_candidates)
        base_2 = get_fake_vectors_single('uniformity', num_candidates)
    elif fake_model_name == 'stun':
        base_1 = get_fake_vectors_single('stratification', num_candidates)
        base_2 = get_fake_vectors_single('uniformity', num_candidates)
    elif fake_model_name == 'stan':
        base_1 = get_fake_vectors_single('stratification', num_candidates)
        base_2 = get_fake_vectors_single('antagonism', num_candidates)
    else:
        raise NameError('No such fake vectors!')

    return convex_combination(base_1, base_2, length=num_candidates, alpha=fake_param)


def convex_combination(base_1, base_2, length=0, alpha=0):
    vectors = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            vectors[i][j] = alpha * base_1[i][j] + (1-alpha) * base_2[i][j]
    return vectors


def crate_combination(base_1, base_2, base_3, base_4, length=0, alpha=None):
    #print(alpha)
    vectors = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            vectors[i][j] = alpha[0] * base_1[i][j] + alpha[1] * base_2[i][j] + \
                            alpha[2] * base_3[i][j] + alpha[3] * base_4[i][j]

    return vectors

