import mapel


if __name__ == "__main__":

    experiment = mapel.prepare_experiment()
    num_candidates = 100  # num_vertices

    experiment.add_family(election_model='erdos_renyi_graph', num_candidates=num_candidates,
                          color='blue', alpha=1, family_id='erdos_path', size=100,
                          path={'param_name': 'p'})

    experiment.add_family(election_model='erdos_renyi_graph', num_candidates=num_candidates,
                          color='black', alpha=1, family_id='complete', params={'p': 1}, size=1)

    experiment.add_family(election_model='erdos_renyi_graph',  num_candidates=num_candidates,
                          color='cyan', alpha=1, family_id='empty', params={'p': 0}, size=1)

    experiment.add_family(election_model='watts_strogatz_graph',  num_candidates=num_candidates,
                          color='green', alpha=0.4,
                          family_id='watts_4_0.2', params={'k': 4, 'p': 0.2}, size=5)
    experiment.add_family(election_model='watts_strogatz_graph',  num_candidates=num_candidates,
                          color='green',
                          family_id='watts_4_0.1', params={'k': 4, 'p': 0.1}, size=15)

    experiment.add_family(election_model='watts_strogatz_graph', num_candidates=num_candidates, color='lime', alpha=0.4,
                          family_id='watts_3_0.2', params={'k': 3, 'p': 0.2}, size=15)
    experiment.add_family(election_model='watts_strogatz_graph', num_candidates=num_candidates, color='lime',
                          family_id='watts_3_0.1', params={'k': 3, 'p': 0.1}, size=15)

    experiment.add_family(election_model='barabasi_albert_graph', num_candidates=num_candidates, color='purple', alpha=0.4,
                          family_id='barabasi_3', params={'m': 3}, size=15)
    experiment.add_family(election_model='barabasi_albert_graph', num_candidates=num_candidates, color='purple',
                          family_id='barabasi_2', params={'m': 2}, size=15)

    experiment.add_family(election_model='random_geometric_graph', num_candidates=num_candidates, color='orange',
                          family_id='geometric_path', size=15, path={'param_name': 'radius'})

    experiment.add_family(election_model='random_geometric_graph', num_candidates=num_candidates, color='red',
                          family_id='geometric_0.1', params={'radius': 0.1}, size=15)

    experiment.add_family(election_model='random_tree', num_candidates=num_candidates, color='brown', family_id='tree', size=30)

    experiment.add_election(election_model='cycle_graph',  num_candidates=num_candidates,
                            color='red', election_id='cycle')
    experiment.add_election(election_model='wheel_graph', num_candidates=num_candidates,
                            color='green', election_id='wheel')
    experiment.add_election(election_model='star_graph',  num_candidates=num_candidates,
                            color='blue', election_id='star')
    experiment.add_election(election_model='ladder_graph', num_candidates=num_candidates,
                            color='black', election_id='ladder')
    experiment.add_election(election_model='circular_ladder_graph', num_candidates=num_candidates,
                            color='purple', election_id='hamster ladder')

    distance_name = 'graph_histogram'

    experiment.compute_distances(distance_name=distance_name)
    #
    experiment.embed(algorithm='spring', num_iterations=1000)
    experiment.print_map(saveas='graphs/'+distance_name, shading=True,
                         title=distance_name+' n='+str(num_candidates))

    # experiment.compute_feature(name=distance_name)
    # experiment.compute_feature(name='graph_diameter_log')

    # experiment.print_map(feature=distance_name, saveas='graphs/'+distance_name,
    #                      title=distance_name)
    #
    # experiment.print_map(feature='graph_diameter_log', saveas='graphs/graph_diameter_log',
    #                      title='(log) graph_diameter [degree_centrality] n=100')
