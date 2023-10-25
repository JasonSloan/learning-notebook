import numpy as np
from scipy.optimize import linear_sum_assignment
from lap import lapjv

# Cost matrix
cost_matrix = np.array([[10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]])

# Solve the linear assignment problem
cost, row_indices, col_indices = lapjv(cost_matrix)

print("Optimal assignment:")
print("Rows:", row_indices)
print("Columns:", col_indices)
