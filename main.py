import sys
import networkx as nx
import numpy as np
from pathlib import Path


def read_matrix(path):
    text = path.read_text()
    numbers = list(map(int, text.split()))
    size = numbers[0]
    matrix = np.array(numbers[1:]).reshape(size, size)
    return matrix

def get_highest_demand_pair(demand_matrix):
    return np.unravel_index(np.argmax(demand_matrix), demand_matrix.shape)

def highest_demand_destination(source, demand_matrix):
    return np.argmax(demand_matrix[source])

def normalize(matrix):
    matrix = matrix.astype(np.float64)
    max_value = np.max(matrix)
    matrix[matrix < 0] = np.inf
    # matrix[matrix == 0] = np.inf
    matrix /= max_value
    return matrix

def get_best_route(source, dest, distance_matrix, demand_matrix, weight):
    normalized_dist = normalize(distance_matrix.copy())
    normalized_demand = normalize(demand_matrix.copy())
    demand_from_source = normalized_demand[source, :]
    demand_to_dest = normalized_demand[:, dest]
    print(demand_to_dest)
    print(demand_from_source)
    print(normalized_demand)
    print(normalized_dist)
    node_cost = 1.0 / (demand_from_source + demand_to_dest + 1.0)
    edge_cost = (1.0 - weight) * normalized_dist + weight * 0.5 * np.add.outer(node_cost, node_cost)
    print(edge_cost)
    graph = nx.convert_matrix.from_numpy_matrix(edge_cost, create_using=nx.DiGraph)
    print(graph.edges[source, dest])
    best_route = nx.algorithms.shortest_paths.weighted.dijkstra_path(graph, source, dest)
    print(best_route)
    route_cost = sum(graph.edges[best_route[i], best_route[i+1]]['weight'] for i in range(len(best_route) - 1))
    demand_met = sum(demand_matrix[best_route[i]][best_route[i+1]] for i in range(len(best_route) - 1))
    route_dist = sum(distance_matrix[best_route[i]][best_route[i+1]] for i in range(len(best_route) - 1))
    print(demand_met)
    print(route_dist)
    print(route_cost)


dist_file = Path(sys.argv[1])
dist = read_matrix(dist_file)
print(dist.shape)
# print(dist)

demand_file = Path(sys.argv[2])
demand = read_matrix(demand_file)
print(demand.shape)
# print(demand)

index = get_highest_demand_pair(demand)
# print(index)
# print(demand[index])

# print(highest_demand_destination(index[0], demand))

# print(normalize(demand))
# print(normalize(dist))

weight = 0.99999

print(get_best_route(index[0], index[1], dist, demand, weight))
