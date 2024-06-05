import numpy as np
from scipy.fftpack import dct

def clenshaw_curtis_weights(n):
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    k = np.arange(0, n)
    theta = np.pi * k / (n - 1)
    x = np.cos(theta)
    v = np.zeros(n)
    v[0] = 1
    v[-1] = 1
    if n > 1:
        v[1:-1:2] = 2
        v[2:-1:2] = 0
    w = dct(v, type=2) / (n - 1)
    w[0] /= 2
    w[-1] /= 2
    return x, w

def adaptive_clenshaw_curtis(f, a, b, tol, max_depth=10, n=5):
    def adaptive_integrate(f, a, b, tol, n, last_integral=None, depth=0):
        if depth > max_depth:
            return last_integral  # Return the last computed integral at max depth

        x, w = clenshaw_curtis_weights(n)
        x = 0.5 * (b - a) * x + 0.5 * (b + a)  # Scale nodes to interval
        fx = f(x)
        integral = 0.5 * (b - a) * np.dot(w, fx)

        if last_integral is not None and np.abs(last_integral - integral) < tol:
            return integral

        mid = 0.5 * (a + b)
        left_integral = adaptive_integrate(f, a, mid, tol / 2, n, integral, depth + 1)
        right_integral = adaptive_integrate(f, mid, b, tol / 2, n, integral, depth + 1)
        return left_integral + right_integral

    return adaptive_integrate(f, a, b, tol, n)

def test_functions():
    functions = [
        {"func": lambda x: x**20, "name": "Polynomial x^20"},
        {"func": lambda x: np.exp(x), "name": "Exponential e^x"},
        {"func": lambda x: np.exp(-x**2), "name": "Gaussian e^(-x^2)"},
        {"func": lambda x: 1 / (1 + 16 * x**2), "name": "Rational 1/(1+16x^2)"},
        {"func": lambda x: np.abs(x)**3, "name": "Non-differentiable |x|^3"},
        {"func": lambda x: 1 / (1 + x**2), "name": "My Function 1/(1+x^2)"}
    ]

    a, b = -1, 1
    tol = 1e-5
    exact_values = [2 / 21, np.e - 1 / np.e, np.sqrt(np.pi), np.pi / 8, 2 / 5, np.pi / 2]  # Exact values for integrals over [-1, 1]

    for func, exact in zip(functions, exact_values):
        integral = adaptive_clenshaw_curtis(func["func"], a, b, tol)
        error = np.abs(exact - integral)
        print(f"{func['name']} Integral: {integral}, Error from exact value: {error}")

if __name__ == "__main__":
    test_functions()
