import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct


# Function to calculate Clenshaw-Curtis weights and nodes using FFT
def clenshaw_curtis_weights(n):
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    k = np.arange(n)
    theta = np.pi * k / (n - 1)
    x = np.cos(theta)

    c = np.zeros(n)
    c[0] = c[-1] = 2

    for i in range(1, n - 1):
        if i % 2 == 1:
            c[i] = 0
        else:
            c[i] = 2 / (1 - k[i] ** 2)

    w = idct(c, type=1) / (n - 1)

    return x, w


# Function to integrate using Clenshaw-Curtis quadrature
def integrate(f, a, b, n=10):
    x, w = clenshaw_curtis_weights(n)
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    fx = f(x)
    integral = 0.5 * (b - a) * np.dot(w, fx)
    return integral, x, fx


# Function to calculate the error and rest estimate
def error_and_rest_convergence(f, exact_value, a, b, max_n):
    ns = np.arange(2, max_n + 1)
    errors = []
    rest_estimates = []

    for n in ns:
        integral, _, _ = integrate(f, a, b, n)
        error = abs(integral - exact_value)
        errors.append(error)

        # Rest estimate calculation using a more accurate approximation
        rest_estimate = np.pi / (n ** 2)
        rest_estimates.append(rest_estimate)

    return ns, np.array(errors), np.array(rest_estimates)


if __name__ == "__main__":
    f = lambda x: 1 / (1 + x ** 2)
    exact_value = np.pi / 2  # Analytical integral of 1/(1+x^2) over [-1, 1]

    max_n = 100
    ns, errors, rest_estimates = error_and_rest_convergence(f, exact_value, -1, 1, max_n)

    # Plotting the error convergence and rest estimates
    plt.figure(figsize=(10, 5))
    plt.plot(ns, errors, marker='o', linestyle='-', color='b', label='Absolute Error')
    plt.plot(ns, rest_estimates, marker='x', linestyle='--', color='r', label='Rest Estimate')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error Convergence and Rest Estimate of Clenshaw-Curtis Quadrature')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, max_n + 1, 10))
    plt.show()

    # Plotting the function and nodes for a specific n
    n = 10
    integral, x, fx = integrate(f, -1, 1, n)
    print(f"Approximate integral with {n} nodes: {integral}")

    plt.figure(figsize=(10, 5))
    t = np.linspace(-1, 1, 1000)
    plt.plot(t, f(t), label='f(x) = 1 / (1 + x^2)')
    plt.scatter(x, fx, color='red', label='Clenshaw-Curtis Nodes')
    plt.title('Function and Clenshaw-Curtis Nodes')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
