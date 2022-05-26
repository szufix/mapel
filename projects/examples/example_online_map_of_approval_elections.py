import mapel


if __name__ == "__main__":

    distance_id = 'l1-approvalwise'

    experiment = mapel.prepare_experiment(instance_type='approval')

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(50)

    experiment.add_family(model_id='impartial_culture', color='black', family_id='ic', size=8)

    experiment.add_family(model_id='urn', color='orange', family_id='urn_1',
                                   params={'alpha': 0.1}, size=8)

    experiment.add_family(model_id='urn', color='red', family_id='urn_2',
                                   params={'alpha': 0.5}, size=8)

    experiment.compute_distances(distance_id=distance_id)

    # experiment.embed()
    #
    # experiment.print_map()
    #
