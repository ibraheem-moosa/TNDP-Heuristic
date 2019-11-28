import sys
import networkx as nx
import numpy as np
from pathlib import Path


def read_matrix(path):
    text = path.read_text()
    numbers = list(map(int, text.split()))
    size = numbers[0]
    matrix = np.array(numbers[1:]).reshape(size, size)
    matrix = matrix.astype(np.float64)
    matrix[matrix < 0] = np.inf
    return matrix

def get_highest_demand_pair(demand_matrix):
    return np.unravel_index(np.argmax(demand_matrix), demand_matrix.shape)

def get_highest_demand_destination_from(source, demand_matrix):
    return np.argmax(demand_matrix[source])

def set_demand_satisfied_in_route(demand_matrix, route):
    demand_matrix = demand_matrix.copy()
    satisfied_demand = 0.
    for i in route:
        for j in route:
            satisfied_demand += demand_matrix[i][j]
            demand_matrix[i][j] = 0.
    return demand_matrix, satisfied_demand

def remove_nodes_in_route_from_graph(distance_matrix, route):
    distance_matrix = distance_matrix.copy()
    for i in route:
        distance_matrix[i] = np.inf
        distance_matrix[:, i] = np.inf
    return distance_matrix

def importance_of_node_in_between(source, dest, demand_matrix):
    demand_from_source = demand_matrix[source, :]
    demand_to_dest = demand_matrix[:, dest]
    return demand_from_source + demand_to_dest

def node_cost_from_importance(node_importance, weight):
    # return np.exp(- weight * node_importance)
    return weight / (1.0 + node_importance) 


def get_best_route_between(source, dest, distance_matrix, demand_matrix, weight):
    node_importance = importance_of_node_in_between(source, dest, demand_matrix)
    node_cost = node_cost_from_importance(node_importance, weight)
    edge_cost = distance_matrix + np.add.outer(node_cost, node_cost)

    edge_cost[distance_matrix == np.inf] = 0.0
    print('ratio: {}'.format(np.nanmax(edge_cost / np.add.outer(node_cost, node_cost))))

    graph = nx.convert_matrix.from_numpy_matrix(edge_cost)
    best_route = nx.algorithms.shortest_paths.weighted.dijkstra_path(graph, source, dest)

    return best_route

def get_route_satisfying_constraint(distance_matrix, demand_matrix, weight, min_hop_count, max_hop_count):
    distance_matrix = distance_matrix.copy()
    demand_matrix = demand_matrix.copy()
    source, dest = get_highest_demand_pair(demand_matrix)
    route = [source]
    while True:
        try:
            route_chunk = get_best_route_between(source, dest, distance_matrix, demand_matrix, weight)
        except nx.NetworkXNoPath as e:
            break
        route_chunk = route_chunk[1:]
        route.extend(route_chunk)
        distance_matrix = remove_nodes_in_route_from_graph(distance_matrix, route[:-1])
        demand_matrix, _ = set_demand_satisfied_in_route(demand_matrix, route)
        source, dest = dest, get_highest_demand_destination_from(dest, demand_matrix)
        if demand_matrix[source][dest] == 0.:
            break
    return route

def get_routes(distance_matrix, demand_matrix, weight, min_hop_count, max_hop_count):
    distance_matrix = distance_matrix.copy()
    demand_matrix = demand_matrix.copy()
    while np.sum(demand_matrix) > 0.:
        route = get_route_satisfying_constraint(distance_matrix, demand_matrix, weight, min_hop_count, max_hop_count)
        demand_matrix, satisfied_demand = set_demand_satisfied_in_route(demand_matrix, route)
        print(satisfied_demand)
        yield route


dist_file = Path(sys.argv[1])
distance_matrix = read_matrix(dist_file)

demand_file = Path(sys.argv[2])
demand_matrix = read_matrix(demand_file)

print(demand_matrix.sum())

dist_copy = distance_matrix.copy()
dist_copy[distance_matrix == np.inf] = np.nan
dema_copy = demand_matrix.copy()
dema_copy[demand_matrix == 0.] = np.nan
print(np.nanmean(dist_copy), np.nanmean(dema_copy))

weight = 0.5
min_hop_count = 2
max_hop_count = 8

routes = list(get_routes(distance_matrix, demand_matrix, weight, min_hop_count, max_hop_count))

print(len(routes))
for route in routes:
    print(route)
    assert(len(route) == len(set(route)))
