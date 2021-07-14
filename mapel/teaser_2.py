import mapel

if __name__ == "__main__":
    num_candidates = 10
    num_voters = 100
    city_size = 10

    Conitzer_city = []
    Walsh_city = []

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

    dict_with_votes['Conitzer(matrix)'] = mapel.generate_votes(election_model='conitzer_matrix',
                                                                       num_candidates=num_candidates,
                                                                       num_voters=num_voters)

    dict_with_votes['Walsh(matrix)'] = mapel.generate_votes(election_model='walsh_matrix',
                                                                    num_candidates=num_candidates,
                                                                    num_voters=num_voters)

    group_by = {'cyan': Walsh_city,
                'blue': ['Walsh(matrix)'],
                'orange': Conitzer_city,
                'red': ['Conitzer(matrix)']}

    distances = mapel.compute_distances_between_votes(dict_with_votes)
    coordinates = mapel.embed(distances, algorithm='mds')
    mapel.print_map(coordinates, group_by=group_by, saveas='cities')
