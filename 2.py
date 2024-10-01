import numpy as np

matrix_A = np.array([[8, 0, 12],
                     [1, -2, 1],
                     [0, 3, 0]])

initial_vector = np.array([1, 0, 0])  

def largest_eigenvalue(matrix, initial_vector, max_iter=100, tol=1e-4):
    """ largest eigenvalue using the Power Method."""
    vector = initial_vector / np.linalg.norm(initial_vector) 
    for _ in range(max_iter):
        product = np.dot(matrix, vector)
        vector = product / np.linalg.norm(product)  
    largest_eigenvalue = np.dot(product, vector) / np.dot(vector, vector) 
    return largest_eigenvalue, vector

def smallest_eigenvalue(matrix, initial_vector, mu_initial, max_iter=100, tol=1e-4):
    """ smallest eigenvalue using the Inverse Power Method."""
    identity = np.eye(matrix.shape[0])
    vector = initial_vector / np.linalg.norm(initial_vector)  
    for _ in range(max_iter):
        solution = np.linalg.solve(matrix - mu_initial * identity, vector)
        vector = solution / np.linalg.norm(solution)  
    smallest_eigenvalue = np.dot(vector, np.dot(matrix, vector)) / np.dot(vector, vector) 
    return smallest_eigenvalue, vector


# Calculate the largest eigenvalue
lambda_max, eigenvector_max = largest_eigenvalue(matrix_A, initial_vector)
print(f"Largest Eigenvalue (Power Method): {lambda_max:.4f}")


# Estimate for the smallest eigenvalue
mu_estimate = 0 
# Calculate the smallest eigenvalue
lambda_min, eigenvector_min = smallest_eigenvalue(matrix_A, initial_vector, mu_estimate)
print(f"Smallest Eigenvalue (Inverse Power Method): {lambda_min:.4f}")

