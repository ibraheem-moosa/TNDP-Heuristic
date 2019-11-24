import sys
import numpy as np
from pathlib import Path


def read_matrix(path):
    with open(path) as f:
        first_line = next(f)
        size = int(first_line)
        matrix = []
        for line in f:
            values = list(map(int, line.split()))
            assert(len(values) == size)
            matrix.append(values)
        assert(len(matrix) == size)
        return np.array(matrix)

dist_file = Path(sys.argv[1])
dist = read_matrix(dist_file)
print(dist.shape)
print(dist)

demand_file = Path(sys.argv[2])
demand = read_matrix(demand_file)
print(demand.shape)
print(demand)
