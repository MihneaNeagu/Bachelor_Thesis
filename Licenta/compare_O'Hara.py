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
        {"func": lambda x: 1 / (1 + x), "name": "Rational 1/(1 + x)", "exact": np.log(2)},  # Adjusted exact value
        {"func": lambda x: 1 / (1 - 0.5 * x**4), "name": "Rational 1/(1 - 0.5 * x^4)", "exact": 2.0},  # Approximate exact value
        {"func": lambda x: 1 / (1 + 100 * x**2), "name": "Rational 1/(1 + 100 * x^2)", "exact": np.arctan(10) / 10},  # Adjusted exact value
        {"func": lambda x: (x + 0.5)**(1/2), "name": "Root (x + 0.5)^(1/2)", "exact": 2.0 / 3},  # Adjusted exact value
    ]

    a, b = 0, 1
    tol = 1e-5
    n_values = [4, 8, 16, 32]  # Different values for the number of nodes

    for func in functions:
        print(f"Results for {func['name']}:")
        for n in n_values:
            integral = adaptive_clenshaw_curtis(func["func"], a, b, tol, n=n)
            error = np.abs(func["exact"] - integral)
            print(f"  Nodes: {n}, Integral: {integral}, Error from exact value: {error}")

if __name__ == "__main__":
    test_functions()
