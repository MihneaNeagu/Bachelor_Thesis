import numpy as np


    # Computes the roots and weights for Gauss-Legendre quadrature manually.
def legendre_roots_and_weights(n):

    roots, weights = np.polynomial.legendre.leggauss(n)
    return roots, weights

    # Approximate the second derivative of f at point x using finite differences.
def numerical_second_derivative(f, x, h=1e-5):
    return (f(x - h) - 2 * f(x) + f(x + h)) / h**2


    # Performs the Gauss-Legendre quadrature over the interval [a, b] and
    # estimates error using a numerical estimate of the second derivative.
def integrate_gauss_legendre(f, a, b, n=10):
    x, w = legendre_roots_and_weights(n)
    # Transform nodes to the interval [a, b]
    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    fx = f(x_mapped)
    integral = 0.5 * (b - a) * np.dot(w, fx)

    # Estimate the second derivative at the midpoint for error estimation
    midpoint = 0.5 * (a + b)
    second_deriv = numerical_second_derivative(f, midpoint)
    error_estimate = (b - a)**3 / (12 * n**2) * second_deriv

    return integral, np.abs(error_estimate)

def test_gauss_legendre_functions():
    functions = [
        {"func": lambda x: x**20, "name": "Polynomial x^20", "exact": 2 / 21},
        {"func": lambda x: np.exp(x), "name": "Exponential e^x", "exact": np.e - 1 / np.e},
        {"func": lambda x: np.exp(-x**2), "name": "Gaussian e^(-x^2)", "exact": np.sqrt(np.pi)},
        {"func": lambda x: 1 / (1 + 16 * x**2), "name": "Rational 1/(1+16x^2)", "exact": np.pi / 8},
        {"func": lambda x: np.abs(x)**3, "name": "Non-differentiable |x|^3", "exact": 2 / 5},
        {"func": lambda x: 1 / (1 + x**2), "name": "My Function 1/(1+x^2)", "exact": np.pi / 2}
    ]

    a, b = -1, 1
    n = 10  # Number of nodes (adjust as needed for accuracy)

    for func in functions:
        integral, error = integrate_gauss_legendre(func['func'], a, b, n)
        exact = func['exact']
        error_from_exact = np.abs(exact - integral)
        print(f"{func['name']} Integral: {integral}, Error from exact value: {error_from_exact}")

if __name__ == "__main__":
    test_gauss_legendre_functions()
