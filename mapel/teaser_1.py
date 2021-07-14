import mapel

if __name__ == "__main__":
    num_candidates = 20
    num_voters = 1000

    dict_with_votes = {'IC': mapel.generate_votes(election_model='impartial_culture',
                                                  num_candidates=num_candidates, num_voters=num_voters),
                       'UN': mapel.generate_votes(election_model='uniformity',
                                                  num_candidates=num_candidates, num_voters=num_voters),
                       'ID': mapel.generate_votes(election_model='identity',
                                                  num_candidates=num_candidates, num_voters=num_voters),
                       'AN': mapel.generate_votes(election_model='antagonism',
                                                  num_candidates=num_candidates, num_voters=num_voters),
                       'ST': mapel.generate_votes(election_model='stratification',
                                                  num_candidates=num_candidates, num_voters=num_voters),
                       }

    distances = mapel.compute_distances_between_votes(dict_with_votes)
    coordinates = mapel.embed(distances)
    mapel.print_map(coordinates)
