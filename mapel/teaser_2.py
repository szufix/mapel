import mapel

if __name__ == "__main__":
    num_candidates = 6
    num_voters = 100
    city_size = 10

    Conitzer_city = []
    Walsh_city = []
    paths = []

    dict_with_votes = {}
    for i in range(city_size):
        votes_id = 'Conniter_' + str(i)
        Conitzer_city.append(votes_id)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='conitzer',
                                                        num_candidates=num_candidates, num_voters=num_voters)
        votes_id = 'Walsh_' + str(i)
        Walsh_city.append(votes_id)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='walsh',
                                                        num_candidates=num_candidates, num_voters=num_voters)

    for i in range(48):
        votes_id = 'UNID_' + str(i)
        paths.append(votes_id)
        param_1 = (i + 1) / (48 + 1)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='unid', num_candidates=num_candidates, num_voters=num_voters, param_1=param_1)
    for i in range(36):
        votes_id = 'ANID_' + str(i)
        paths.append(votes_id)
        param_1 = (i + 1) / (36 + 1)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='anid', num_candidates=num_candidates, num_voters=num_voters, param_1=param_1)
    for i in range(23):
        votes_id = 'STID_' + str(i)
        paths.append(votes_id)
        param_1 = (i + 1) / (23 + 1)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='stid', num_candidates=num_candidates, num_voters=num_voters, param_1=param_1)
    for i in range(23):
        votes_id = 'ANUN_' + str(i)
        paths.append(votes_id)
        param_1 = (i + 1) / (23 + 1)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='anun', num_candidates=num_candidates, num_voters=num_voters, param_1=param_1)
    for i in range(36):
        votes_id = 'STUN_' + str(i)
        paths.append(votes_id)
        param_1 = (i + 1) / (36 + 1)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='stun', num_candidates=num_candidates, num_voters=num_voters, param_1=param_1)
    for i in range(39):
        votes_id = 'STAN_' + str(i)
        paths.append(votes_id)
        param_1 = (i + 1) / (39 + 1)
        dict_with_votes[votes_id] = mapel.generate_votes(election_model='stan', num_candidates=num_candidates, num_voters=num_voters, param_1=param_1)

    dict_with_votes['ID'] = mapel.generate_votes(election_model='identity', num_candidates=num_candidates, num_voters=num_voters)
    dict_with_votes['UN'] = mapel.generate_votes(election_model='uniformity', num_candidates=num_candidates, num_voters=num_voters)
    dict_with_votes['AN'] = mapel.generate_votes(election_model='antagonism', num_candidates=num_candidates, num_voters=num_voters)
    dict_with_votes['ST'] = mapel.generate_votes(election_model='stratification', num_candidates=num_candidates, num_voters=num_voters)

    dict_with_votes['Conitzer(matrix)'] = mapel.generate_votes(election_model='conitzer_matrix',
                                                                       num_candidates=num_candidates,
                                                                       num_voters=num_voters)

    dict_with_votes['Walsh(matrix)'] = mapel.generate_votes(election_model='walsh_matrix',
                                                                    num_candidates=num_candidates,
                                                                    num_voters=num_voters)

    dict_with_votes['gs_cat'] = mapel.generate_votes(election_model='gs_caterpillar_matrix',
                                                                    num_candidates=num_candidates,
                                                                    num_voters=num_voters)
    group_by = {'cyan': Walsh_city,
                'blue': ['Walsh(matrix)'],
                'orange': Conitzer_city,
                'yellow': ['Conitzer(matrix)'],
                'grey': paths,
                'darkblue': ['ID'],
                'black': ['UN'],
                'red': ['AN'],
                'green': ['ST'],
                'lime': ['gs_cat']}

    distances = mapel.compute_distances_between_votes(dict_with_votes)
    coordinates = mapel.embed(distances, algorithm='mds', attraction_factor=1)
    mapel.print_map(coordinates, group_by=group_by, saveas='cities')
