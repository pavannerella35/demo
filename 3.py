import numpy as np

matrix_A = np.array([[1, 3, 4],
                      [3, 1, 2],
                      [4, 2, 1]])
iterations = 20

def round_value(x, decimals=4):
    """Round a number, converting to int if it's effectively zero."""
    rounded = round(x, decimals)
    return int(rounded) if rounded == 0 else rounded

# QR Iteration
for _ in range(iterations):
    Q, R = np.linalg.qr(matrix_A)
    matrix_A = R @ Q

# Print the resulting matrix A after 20 iterations
rounded_A = np.vectorize(round_value)(matrix_A)
print("Matrix A after 20 iterations:")
print(rounded_A)
