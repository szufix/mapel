
import os


def thread_function(experiment, distance_name, all_pairs,
                    election_models, params,
                    num_voters, num_candidates, thread_ids,
                    distances, t, precision,
                    matchings, times):
    for election_id_1, election_id_2 in thread_ids:
        # print('hello', election_id_1, election_id_2)
        result = 0
        local_ctr = 0

        total_time = 0

        for p in range(precision):
            # print('p', p)
            # print('local', t)
            if t < 5:
                print("ctr: ", local_ctr)
                local_ctr += 1

            # print(params)
            election_1 = el.generate_instances(
                experiment=experiment,
                election_model=election_models[election_id_1],
                election_id=election_id_1,
                num_candidates=num_candidates, num_voters=num_voters,
                params=params[election_id_1])
            # print('start')
            election_2 = el.generate_instances(
                experiment=experiment,
                election_model=election_models[election_id_2],
                election_id=election_id_2,
                num_candidates=num_candidates, num_voters=num_voters,
                params=params[election_id_2])

            start_time = time()
            distance, mapping = get_distance(election_1, election_2,
                                             distance_name=distance_name)
            total_time += (time() - start_time)
            # print(distance)
            # delete tmp files

            all_pairs[election_id_1][election_id_2][p] = round(distance, 5)
            result += distance

        distances[election_id_1][election_id_2] = result / precision
        distances[election_id_2][election_id_1] = \
            distances[election_id_1][election_id_2]

        # matchings[election_id_1][election_id_2] = mapping
        times[election_id_1][election_id_2] = total_time / precision
        times[election_id_2][election_id_1] = \
            times[election_id_1][election_id_2]

    print("thread " + str(t) + " is ready :)")


def compute_subelection_by_groups(
        experiment, distance_name='0-voter_subelection', t=0, num_threads=1,
        precision=10, self_distances=False, num_voters=None,
        num_candidates=None):

    if num_candidates is None:
        num_candidates = experiment.default_num_candidates
    if num_voters is None:
        num_voters = experiment.default_num_voters

    num_distances = experiment.num_families * (experiment.num_families + 1) / 2

    threads = [None for _ in range(num_threads)]

    election_models = {}
    params = {}

    for family_id in experiment.families:
        for i in range(experiment.families[family_id].size):
            election_id = family_id + '_' + str(i)
            election_models[election_id] = \
                experiment.families[family_id].model
            params[election_id] = experiment.families[family_id].params

    ids = []
    for i, election_1 in enumerate(experiment.instances):
        for j, election_2 in enumerate(experiment.instances):
            if i == j:
                if self_distances:
                    ids.append((election_1, election_2))
            elif i < j:
                ids.append((election_1, election_2))

    all_pairs = {}
    std = {}
    distances = {}
    matchings = {}
    times = {}
    for i, election_1 in enumerate(experiment.instances):
        all_pairs[election_1] = {}
        std[election_1] = {}
        distances[election_1] = {}
        matchings[election_1] = {}
        times[election_1] = {}
        for j, election_2 in enumerate(experiment.instances):
            if i == j:
                if self_distances:
                    all_pairs[election_1][election_2] = [0 for _ in range(precision)]
            elif i < j:
                all_pairs[election_1][election_2] = [0 for _ in range(precision)]

    for t in range(num_threads):
        start = int(t * num_distances / num_threads)
        stop = int((t + 1) * num_distances / num_threads)
        thread_ids = ids[start:stop]
        print('t: ', t)

        threads[t] = Thread(target=thread_function,
                            args=(experiment, distance_name, all_pairs,
                                  election_models, params,
                                  num_voters, num_candidates,
                                  thread_ids, distances, t,
                                  precision, matchings, times))
        threads[t].start()

    for t in range(num_threads):
        threads[t].join()

    # COMPUTE STD
    for i, election_1 in enumerate(experiment.instances):
        for j, election_2 in enumerate(experiment.instances):
            if i == j:
                if self_distances:
                    value = float(np.std(np.array(all_pairs[election_1][election_2])))
                    std[election_1][election_2] = round(value, 5)
            elif i < j:
                value = float(np.std(np.array(all_pairs[election_1][election_2])))
                std[election_1][election_2] = round(value, 5)

    experiment.distances = distances
    experiment.times = times

    if experiment.store:
        path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "distances",
                            str(distance_name) + ".csv")

        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(
                ["election_id_1", "election_id_2", "distance", "time", "std"])

            for i, election_1 in enumerate(experiment.instances):
                for j, election_2 in enumerate(experiment.instances):
                    if (i == j and self_distances) or i < j:
                        distance = str(distances[election_1][election_2])
                        time = str(times[election_1][election_2])
                        std_value = str(std[election_1][election_2])
                        writer.writerow(
                            [election_1, election_2, distance, time, std_value])

    if experiment.store:

        ctr = 0
        path = os.path.join(os.getcwd(), "experiments",
                            experiment.experiment_id, "distances",
                            str(distance_name) + "_all_pairs.txt")
        with open(path, 'w') as txtfile:
            for i, election_1 in enumerate(experiment.instances):
                for j, election_2 in enumerate(experiment.instances):
                    if (i == j and self_distances) or i < j:
                        for p in range(precision):
                            txtfile.write(str(i) + ' ' + str(j) + ' ' + str(p) + ' ' +
                                          str(all_pairs[election_1][election_2][p]) + '\n')
                            ctr += 1


# def _reverse(vector):
#     if vector is None:
#         return None
#     new_vector = [0 for _ in range(len(vector))]
#     for i in range(len(vector)):
#         new_vector[vector[i]] = i
#     return new_vector