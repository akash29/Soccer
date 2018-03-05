import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(10)
plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams["figure.figsize"] = (10, 10)

q_table_A = np.ones((112, 5, 5))
q_table_B = np.ones((112, 5, 5))

def build_ce_constraints(A):
    num_vars = int(len(A) ** (1/2))
    G = []
    # row player
    for i in range(num_vars): # action row i
        for j in range(num_vars): # action row j
            if i != j:
                constraints = [0 for i in A]
                base_idx = i * num_vars
                comp_idx = j * num_vars
                for k in range(num_vars):
                    constraints[base_idx+k] = (- A[base_idx+k][0]
                                               + A[comp_idx+k][0])
                G += [constraints]
    # col player
    for i in range(num_vars): # action column i
        for j in range(num_vars): # action column j
            if i != j:
                constraints = [0 for i in A]
                for k in range(num_vars):
                    constraints[i + (k * num_vars)] = (
                        - A[i + (k * num_vars)][1]
                        + A[j + (k * num_vars)][1])
                G += [constraints]
    return np.matrix(G, dtype="float")

result = build_ce_constraints(q_table_A)
print(result)