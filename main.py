import sys
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


    
dist_file = Path(sys.argv[1])
dist = read_matrix(dist_file)
print(dist.shape)
print(dist)

demand_file = Path(sys.argv[2])
demand = read_matrix(demand_file)
print(demand.shape)
print(demand)

index = get_highest_demand_pair(demand)
print(index)
print(demand[index])
