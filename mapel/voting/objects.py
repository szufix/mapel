""" This module contains all the objects """

import os
import math
import csv



class Model:
    """ Abstract model of elections """

    def __init__(self, experiment_id, ignore=None, num_elections=800, main_order_name="default"):

        self.main_order = self.import_order(experiment_id, main_order_name)
        self.experiment_id = experiment_id
        self.num_voters, self.num_candidates, self.num_families, \
            self.families = self.import_controllers(experiment_id, ignore=ignore,
                                                    num_elections=num_elections, main_order=self.main_order)

        self.num_elecitons = num_elections
        #for family in self.families:
        #    self.num_elecitons += family.size

    @staticmethod
    def import_controllers(experiment_id, ignore=None, num_elections=800, main_order=None):
        """ Import from a file all the controllers"""

        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "basic", "map.txt")
        print(file_name)
        file_ = open(file_name, 'r')
        num_voters = int(file_.readline())
        num_candidates = int(file_.readline())
        num_families = int(file_.readline())
        families = []

        for i in range(num_families):
            line = file_.readline().rstrip("\n").split(',')
            show = True
            if line[0][0] == "#":
                line[0] = line[0].replace("#", "")
                show = False

            families.append(Family(name=str(line[1]),
                                   special_1=float(line[2]),
                                   special_2=float(line[3]),
                                   size=int(line[0]), label=str(line[6]),
                                   color=str(line[4].replace(" ", "")), alpha=float(line[5]),
                                   show=show))

        if ignore is None:
            ignore = []

        ctr = 0
        for i in range(num_families):
            resize = 0
            for j in range(families[i].size):
                if main_order[ctr] >= num_elections or main_order[ctr] in ignore:
                    resize += 1
                ctr += 1
            families[i].size -= resize

        file_.close()
        return num_voters, num_candidates, num_families, families

    @staticmethod
    def import_order(experiment_id, main_order_name):
        """ Import from a file precomputed order of all the elections """

        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "results", "orders", main_order_name + ".txt")
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

    def __init__(self, experiment_id, metric):

        Model.__init__(self, experiment_id)

        self.num_points, self.num_distances, self.distances = self.import_distances(experiment_id, metric)

    @staticmethod
    def import_distances(experiment_id, metric):
        """Import from a file precomputed distances between each pair of elections  """

        file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "results", "distances", str(metric) + ".txt")
        file_ = open(file_name, 'r')
        num_points = int(file_.readline())
        file_.readline()  # skip this line
        num_distances = int(file_.readline())

        hist_data = [[0 for _ in range(num_points)] for _ in range(num_points)]

        for a in range(num_points):
            for b in range(a + 1, num_points):
                line = file_.readline()
                line = line.split(' ')
                hist_data[a][b] = float(line[2])

        return num_points, num_distances, hist_data


class Model_2d(Model):
    """ Two-dimensional model of elections """

    def __init__(self, experiment_id, num_winners=0, num_elections=800, secondary_order_name="positionwise_approx_cc",
                 main_order_name="default", metric="positionwise", ignore=None):

        Model.__init__(self, experiment_id, ignore=ignore, num_elections=num_elections, main_order_name=main_order_name)

        self.num_points, self.points, = self.import_points(experiment_id, metric, ignore=ignore,
                                                           num_elections=num_elections, main_order=self.main_order)
        self.points_by_families = self.compute_points_by_families()

        #self.secondary_order = self.import_order(exp_name, num_elections, secondary_order_name)
        #self.orders = self.secondary_order[0:num_winners]

    @staticmethod
    def import_points(experiment_id, metric, ignore=None, num_elections=800, main_order=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []
        """
        file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "results", "points", str(metric) + "_2d.txt")
        file_ = open(file_name, 'r')

        num_points = int(file_.readline())
        points = [[0, 0] for _ in range(num_points)]

        for i in range(num_points):
            line = file_.readline().replace("\n", '').split(',')
            points[i] = [float(line[0]), float(line[1])]

        file_.close()
        """

        points = []
        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "results", "points", metric + "_2d.csv")
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            ctr = 0
            for row in reader:
                if main_order[ctr] < num_elections and main_order[ctr] not in ignore:
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

    def __init__(self, experiment_id, num_winners=0, num_elections="800", winners_order="positionwise_approx_cc", order="", metric="positionwise"):
        Model.__init__(self, experiment_id)

        self.num_points, self.points, = self.import_points(experiment_id, metric)
        self.points_by_families = self.compute_points_by_families()

        #self.winners_order = self.import_order(exp_name, num_elections, winners_order)
        self.order = self.import_order(experiment_id, num_elections, order)
        #self.orders = self.winners_order[0:num_winners]

    @staticmethod
    def import_points(experiment, metric):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        """
        file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "results", "points", str(metric) + "_3d.txt")
        file_ = open(file_name, 'r')

        num_points = int(file_.readline())
        points = [[0, 0] for _ in range(num_points)]

        for i in range(num_points):
            line = file_.readline().replace("\n", '').split(',')
            points[i] = [float(line[0]), float(line[1]), float(line[2])]

        file_.close()
        """
        points = []
        file_name = os.path.join(os.getcwd(), "experiments", experiment, "results", "points", metric + "_3d.csv")
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                points.append([float(row['x']), float(row['y']), float(row['z'])])

        return len(points), points

    @staticmethod
    def import_order(experiment_id, num_elections, order_name):
        """ Import from a file precomputed order of all the elections """

        print(order_name, '?')
        if order_name == "":
            return [i for i in range(num_elections)]

        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "results", "orders", str(order_name) + ".txt")
        file_ = open(file_name, 'r')
        file_.readline()  # skip this line
        file_.readline()  # skip this line
        file_.readline()  # skip this line
        order = []

        for w in range(num_elections):
            order.append(int(file_.readline()))

        return order

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

    def __init__(self, name="none", special_1=0., special_2=0., size=0, label="none", color="black", alpha=1., show=True):

        self.name = name
        self.special_1 = special_1
        self.special_2 = special_2
        self.size = size
        self.label = label
        self.color = color
        self.alpha = alpha
        self.show = show


class Election:

    def __init__(self, experiment_id, election_id):

        self.experiment_id = experiment_id
        self.election_id = election_id

        #votes, num_voters, num_candidates = el.import_soc_elections(experiment_id, election_id)
        #self.votes = votes
        #self.num_voters = num_voters
        #self.num_candidates = num_candidates
        #self.num_elections = 1  # do poprawy

        self.votes, self.num_voters, self.num_candidates = import_soc_elections(experiment_id, election_id)
        self.num_elections = 1  # do poprawy

        self.potes = self.votes_to_potes()

    def votes_to_potes(self):
        potes = [[-1 for _ in range(self.num_candidates)] for _ in range(self.num_voters)]
        for i in range(self.num_voters):
            for j in range(self.num_candidates):
                potes[i][self.votes[i][j]] = j
        return potes

    def votes_to_positionwise_vectors(self):

        vectors = [[0 for _ in range(self.num_candidates)] for _ in range(self.num_candidates)]

        if self.votes[0][0] == -1:

            vectors = [[1. / self.num_candidates for _ in range(self.num_candidates)] for _ in
                         range(self.num_candidates)]

        elif self.votes[0][0] == -2:

            half = self.num_candidates / 2
            for i in range(half):
                vectors[i][i] = 1.
            for i in range(half, half * 2):
                for j in range(half, half * 2):
                    vectors[i][j] = 1. / half

        #  path 0.5
        elif self.votes[0][0] == -3:

            vectors = [[0.5 / self.num_candidates for _ in range(self.num_candidates)] for _ in
                         range(self.num_candidates)]

            for i in range(self.num_candidates):
                vectors[i][i] += 0.5

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

       # print(sum(vectors[0]))

        return vectors

    def votes_to_lacknerwise_vectors(self):

        vectors = [[0 for _ in range(self.num_voters)] for _ in range(self.num_voters)]

        for i in range(self.num_voters):
            #for j in range(i-1) + range(i+1, self.num_voters):
            for j in range(self.num_voters):    # now self-similarity is 1
                set_1 = set(self.votes[0][i])
                set_2 = set(self.votes[0][j])
                #vectors[i][j] = len(set_1) + len(set_2) - 2*len(set_1.intersection(set_2))
                vectors[i][j] = self.num_candidates - len(set_1) - len(set_2) + 2 * len(set_1.intersection(set_2))
                """
                #similarity = len(set_1.intersection(set_2)) / float(len(set_1))
                div = max(len(set_1), len(set_2))
                if div == 0:
                    similarity = 1.
                else:
                    similarity = len(set_1.intersection(set_2)) / float(div)
                vectors[i][j] = similarity
                """
            vectors[i] = sorted(vectors[i])

        return vectors

    def votes_to_bordawise_vector(self):

        num_possible_scores = 1 + self.num_voters*(self.num_candidates - 1)

        if self.votes[0][0][0] == -1:

            vector = [0 for _ in range(num_possible_scores)]
            peak = sum([i for i in range(self.num_candidates)]) * float(self.num_voters) / float(self.num_candidates)
            vector[int(peak)] = self.num_candidates

        else:

            vector = [0 for _ in range(num_possible_scores)]
            points = get_borda_points(self.votes[0], self.num_voters, self.num_candidates)
            for i in range(self.num_candidates):
                vector[points[i]] += 1

        print(vector)
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
    points = sorted(points, reverse=True)

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
