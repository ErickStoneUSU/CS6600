###########################################
# model: hw02_f19_data.py
# author: Erick Stone A02217762
###########################################

import numpy as np

# Input Data for AND, OR, and XOR
X1 = np.array([[0, 0],
               [1, 0],
               [0, 1],
               [1, 1]])

# Input Data for NOT
X2 = np.array([[0, 1],
               [1, 1]])

# Input Data for the boolean expression problem.
X4 = np.array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 1, 1],
               [0, 1, 0, 0],
               [0, 1, 0, 1],
               [0, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 0, 0, 0],
               [1, 0, 0, 1],
               [1, 0, 1, 0],
               [1, 0, 1, 1],
               [1, 1, 0, 0],
               [1, 1, 0, 1],
               [1, 1, 1, 0],
               [1, 1, 1, 1]])

# Ground truth for AND
y_and = np.array([[0],
                  [0],
                  [0],
                  [1]])

# Ground truth for OR
y_or = np.array([[0],
                 [1],
                 [1],
                 [1]])

# Ground truth for XOR
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# Ground truth for NOT
y_not = np.array([[1],
                  [0]])

# Ground truth for boolean expression.
bool_exp = np.array([[1],
                     [0],
                     [0],
                     [0],
                     [1],
                     [0],
                     [0],
                     [0],
                     [1],
                     [0],
                     [0],
                     [0],
                     [1],
                     [1],
                     [1],
                     [1]])
