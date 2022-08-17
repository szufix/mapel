import networkx as nx


def generate_graph(model_id=None, num_nodes=None, params=None):
    non_params_graphs = {'cycle_graph': nx.cycle_graph,
                         'wheel_graph': nx.wheel_graph,
                         'star_graph': nx.star_graph,
                         'ladder_graph': nx.ladder_graph,
                         'circular_ladder_graph': nx.circular_ladder_graph,
                         'random_tree': nx.random_tree,
                         }

    if model_id in non_params_graphs:
        return non_params_graphs[model_id](num_nodes)

    elif model_id in ['erdos_renyi_graph', 'erdos_renyi_graph_path']:
        return nx.erdos_renyi_graph(num_nodes, params['p'])
    elif model_id == 'watts_strogatz_graph':
        return nx.watts_strogatz_graph(num_nodes, params['k'], params['p'])
    elif model_id == 'barabasi_albert_graph':
        return nx.barabasi_albert_graph(num_nodes, params['m'])
    elif model_id == 'random_geometric_graph':
        return nx.random_geometric_graph(num_nodes, params['radius'])
