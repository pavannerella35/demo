import numpy as np

A = np.array([[3, -1, 0, 0, 0, 0.5],
              [-1, 3, -1, 0, 0.5, 0],
              [0, -1, 3, -1, 0, 0],
              [0, 0, -1, 3, -1, 0],
              [0, 0.5, 0, -1, 3, -1],
              [0.5, 0, 0, 0, -1, 3]])

b = np.array([5/2, 3/2, 1, 1, 3/2, 5/2])
initial_guess = np.zeros(6)
N = 100  
omega = 1.1
error_tolerance = 0.001

# Jacobi iterative method
def jacobi(A, b, x0, N, tol):
    x = x0.copy()
    for iteration in range(N):
        x_new = np.zeros_like(x)
        for i in range(len(x)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        if iteration == 19: 
            print(f"Jacobi Iteration {iteration + 1}: {x_new}")
        if np.linalg.norm(x_new - x) <= tol:
            break
        x = x_new
    return x, iteration
jacobi_result, jacobi_iterations = jacobi(A, b, initial_guess, N, error_tolerance)
print("Final Solution Vector (Jacobi):", jacobi_result)
print("Number of Iterations (Jacobi):", jacobi_iterations + 1)

# Gauss-Seidel iterative method
def gauss_seidel(A, b, x0, N, tol):
    x = x0.copy()
    for iteration in range(N):
        x_new = np.copy(x)
        for i in range(len(b)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        if iteration == 19:  
            print(f"Gauss-Seidel Iteration {iteration + 1}: {x_new}")
        if np.all(np.abs(x - x_new) <= tol):
            break
        x = x_new
    return x, iteration
gauss_result, gauss_iterations = gauss_seidel(A, b, initial_guess, N, error_tolerance)
print("Final Solution Vector (Gauss-Seidel):", gauss_result)
print("Number of Iterations (Gauss-Seidel):", gauss_iterations + 1)

# Successive Over Relaxation (SOR) method
def sor(A, b, x0, N, omega, tol):
    n = len(b)
    x = x0.copy()
    for iteration in range(N):
        x_new = x.copy()
        for i in range(n):
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:]))
        if iteration == 19:  
            print(f"SOR Iteration {iteration + 1}: {x_new}")
        if np.linalg.norm(x_new - x) <= tol:
            break
        x = x_new
    return x, iteration
sor_result, sor_iterations = sor(A, b, initial_guess, N, omega, error_tolerance)
print("Final Solution Vector (SOR):", sor_result)
print("Number of Iterations (SOR):", sor_iterations + 1)
