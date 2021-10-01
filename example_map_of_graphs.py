import mapel


if __name__ == "__main__":

    experiment = mapel.prepare_experiment()
    num_nodes = 50

    experiment.add_family(model='erdos_renyi_graph', num_nodes=num_nodes,
                          color='blue', alpha=1, name='erdos_path', size=50,
                          path={'param_name': 'p'})

    experiment.add_family(model='erdos_renyi_graph', num_nodes=num_nodes,
                          color='black', alpha=1, name='complete', params={'p': 1}, size=1)

    experiment.add_family(model='erdos_renyi_graph',  num_nodes=num_nodes,
                          color='cyan', alpha=1, name='empty', params={'p': 0}, size=1)

    experiment.add_family(model='watts_strogatz_graph',  num_nodes=num_nodes,
                          color='green', alpha=0.4,
                          name='watts_4_0.2', params={'k': 4, 'p': 0.2}, size=5)
    experiment.add_family(model='watts_strogatz_graph',  num_nodes=num_nodes,
                          color='green',
                          name='watts_4_0.1', params={'k': 4, 'p': 0.1}, size=15)

    experiment.add_family(model='watts_strogatz_graph', num_nodes=num_nodes, color='lime', alpha=0.4,
                          name='watts_3_0.2', params={'k': 3, 'p': 0.2}, size=15)
    experiment.add_family(model='watts_strogatz_graph', num_nodes=num_nodes, color='lime',
                          name='watts_3_0.1', params={'k': 3, 'p': 0.1}, size=15)

    experiment.add_family(model='barabasi_albert_graph', num_nodes=num_nodes, color='purple', alpha=0.4,
                          name='barabasi_3', params={'m': 3}, size=15)
    experiment.add_family(model='barabasi_albert_graph', num_nodes=num_nodes, color='purple',
                          name='barabasi_2', params={'m': 2}, size=15)

    experiment.add_family(model='random_geometric_graph', num_nodes=num_nodes, color='orange',
                          name='geometric_path', size=15, path={'param_name': 'radius'})

    experiment.add_family(model='random_geometric_graph', num_nodes=num_nodes, color='red',
                          name='geometric_0.1', params={'radius': 0.1}, size=15)

    experiment.add_family(model='random_tree', num_nodes=num_nodes, color='brown', name='tree', size=30)

    experiment.add_graph(model='cycle_graph',  num_nodes=num_nodes, color='red', name='cycle')
    experiment.add_graph(model='wheel_graph', num_nodes=num_nodes, color='green', name='wheel')
    experiment.add_graph(model='star_graph',  num_nodes=num_nodes, color='blue', name='star')
    experiment.add_graph(model='ladder_graph', num_nodes=num_nodes, color='black', name='ladder')
    experiment.add_graph(model='circular_ladder_graph', num_nodes=num_nodes, color='purple', name='hamster ladder')

    distance_name = 'graph_histogram'

    experiment.compute_distances(distance_name=distance_name)

    experiment.embed(algorithm='spring', num_iterations=1000)
    experiment.print_map(saveas='graphs/'+distance_name, shading=True,
                         title=distance_name+' n='+str(num_nodes))

    # experiment.compute_feature(name=distance_name)
    # experiment.compute_feature(name='graph_diameter_log')

    # experiment.print_map(feature=distance_name, saveas='graphs/'+distance_name,
    #                      title=distance_name)
    #
    # experiment.print_map(feature='graph_diameter_log', saveas='graphs/graph_diameter_log',
    #                      title='(log) graph_diameter [degree_centrality] n=100')
