from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


def line_search_bisection(
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    c1: float = 1e-3,   # from the screenshot
    c2: float = 1e-2,   # from the screenshot
    alpha0: float = 0.0,
    beta0: float = 1e6,
    t_init: float = 1.0,
    max_iter: int = 1000,
    eps: float = 1e-6
) -> float:
    """
    Bisection-based line search from the provided pseudo-code:

    Initialization:
      c1 = 0.001, c2 = 0.01, alpha0 = 0, beta0 = 10^6, t = 1
    We bracket the step size t between alpha and beta, checking:
      1) f(x + t d) <= f(x) + c1 t ∇f(x)^T d    (Armijo)
      2) ∇f(x + t d)^T d >= c2 ∇f(x)^T d       (Curvature)

    If condition 1) fails => we shrink beta
    If condition 2) fails => we raise alpha
    Otherwise => we STOP
    """
    alpha = alpha0
    beta = beta0
    t = t_init
    k = 0

    # Precompute needed values
    f_x = f(x)
    grad_x = d_f(x)
    dir_deriv = np.dot(grad_x, d)  # ∇f(x)^T d

    while k < max_iter:
        # Check the Armijo condition
        lhs = f(x + t * d)
        rhs = f_x + c1 * t * dir_deriv
        if lhs > rhs:
            # Armijo not satisfied => decrease upper bracket
            beta = t
            t = 0.5 * (alpha + beta)
            k += 1
            continue

        # Check the curvature condition
        grad_xtd = d_f(x + t * d)
        dir_deriv_xtd = np.dot(grad_xtd, d)  # ∇f(x + t d)^T d
        if dir_deriv_xtd < c2 * dir_deriv:
            # Curvature not satisfied => increase lower bracket
            alpha = t
            t = 0.5 * (alpha + beta)
            k += 1
            continue

        # If both conditions satisfied => STOP
        break

    return t


def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    """
    Conjugate Gradient Descent with three variants (HS, PR, FR).
    Uses the bisection line search above.
    """
    tol = 1e-6
    max_iter = 1e3
    x = inital_point.copy()
    g = d_f(x)
    d = -g
    k = 0

    # Records for plotting
    f_vals = [f(x)]
    grad_vals = [np.linalg.norm(g)]
    path = [x.copy()]

    while np.linalg.norm(g) > tol and k < max_iter:
        # Line search
        t = line_search_bisection(f, d_f, x, d)
        x_new = x + t * d
        g_new = d_f(x_new)

        # Compute beta
        if k == 0:
            beta = 0.0
        else:
            if approach == "Hestenes-Stiefel":
                denom = np.dot(d, g_new - g)
                if abs(denom) < 1e-16:
                    beta = 0.0
                else:
                    beta = np.dot(g_new, g_new - g) / denom
            elif approach == "Polak-Ribiere":
                denom = np.dot(g, g)
                if abs(denom) < 1e-16:
                    beta = 0.0
                else:
                    beta = np.dot(g_new, g_new - g) / denom
            elif approach == "Fletcher-Reeves":
                denom = np.dot(g, g)
                if abs(denom) < 1e-16:
                    beta = 0.0
                else:
                    beta = np.dot(g_new, g_new) / denom
            else:
                raise ValueError("Unknown approach for conjugate descent")

        d = -g_new + beta * d
        x, g = x_new, g_new
        f_vals.append(f(x))
        grad_vals.append(np.linalg.norm(g))
        path.append(x.copy())
        k += 1

    path = np.array(path)
    plot_optimization_results(f, inital_point, f"CG-{approach}", f_vals, grad_vals, path)
    return x


def sr1(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Quasi-Newton method using the Symmetric Rank-One (SR1) update.
    Maintains an inverse Hessian approximation H.
    Update:
        H_{k+1} = H_k + [(s - H_k y)(s - H_k y)^T] / [(s - H_k y)^T y]
    """
    tol = 1e-6
    max_iter = 1e3
    x = inital_point.copy()
    g = d_f(x)
    n = len(x)
    H = np.eye(n)
    k = 0

    # Records for plotting
    f_vals = [f(x)]
    grad_vals = [np.linalg.norm(g)]
    path = [x.copy()]
    
    while np.linalg.norm(g) > tol and k < max_iter:
        p = -H @ g
        t = line_search_bisection(f, d_f, x, p)
        s = t * p
        x_new = x + s
        g_new = d_f(x_new)
        y = g_new - g

        diff = s - H @ y
        denom = np.dot(diff, y)
        if abs(denom) > 1e-12:
            H = H + np.outer(diff, diff) / denom

        x, g = x_new, g_new
        f_vals.append(f(x))
        grad_vals.append(np.linalg.norm(g))
        path.append(x.copy())
        k += 1

    path = np.array(path)
    plot_optimization_results(f, inital_point, "SR1", f_vals, grad_vals, path)
    
    return x


def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Quasi-Newton method using the DFP update.
    Maintains an inverse Hessian approximation H.
    Update:
        H_{k+1} = H_k + (s s^T) / (s^T y) - (H_k y y^T H_k) / (y^T H_k y)
    """
    tol = 1e-6
    max_iter = 1e3
    x = inital_point.copy()
    g = d_f(x)
    n = len(x)
    H = np.eye(n)
    k = 0


    # Records for plotting
    f_vals = [f(x)]
    grad_vals = [np.linalg.norm(g)]
    path = [x.copy()]
    
    while np.linalg.norm(g) > tol and k < max_iter:
        p = -H @ g
        t = line_search_bisection(f, d_f, x, p)
        s = t * p
        x_new = x + s
        g_new = d_f(x_new)
        y = g_new - g

        sTy = np.dot(s, y)
        if abs(sTy) > 1e-12:
            Hy = H @ y
            yTHy = np.dot(y, Hy)
            if abs(yTHy) > 1e-12:
                H = H + np.outer(s, s) / sTy - np.outer(Hy, Hy) / yTHy

        x, g = x_new, g_new
        f_vals.append(f(x))
        grad_vals.append(np.linalg.norm(g))
        path.append(x.copy())
        k += 1

    path = np.array(path)
    plot_optimization_results(f, inital_point, "DFP", f_vals, grad_vals, path)

    return x


def bfgs(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Quasi-Newton method using the BFGS update (inverse Hessian form).
    Update:
        H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
      where ρ = 1 / (y^T s).
    """
    tol = 1e-6
    max_iter = 1e3
    x = inital_point.copy()
    g = d_f(x)
    n = len(x)
    H = np.eye(n)
    k = 0

    # Records for plotting
    f_vals = [f(x)]
    grad_vals = [np.linalg.norm(g)]
    path = [x.copy()]

    while np.linalg.norm(g) > tol and k < max_iter:
        p = -H @ g
        t = line_search_bisection(f, d_f, x, p)
        s = t * p
        x_new = x + s
        g_new = d_f(x_new)
        y = g_new - g

        sTy = np.dot(s, y)
        if abs(sTy) > 1e-12:
            rho = 1.0 / sTy
            I = np.eye(n)
            # BFGS update (inverse Hessian):
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x, g = x_new, g_new
        f_vals.append(f(x))
        grad_vals.append(np.linalg.norm(g))
        path.append(x.copy())
        k += 1
        
    path = np.array(path)
    plot_optimization_results(f, inital_point, "BFGS", f_vals, grad_vals, path)
    
    return x


def plot_optimization_results(
    f: Callable[[NDArray[np.float64]], float],
    inital_point: NDArray[np.float64],
    condition: str,
    f_vals: list,
    grad_vals: list,
    path: NDArray[np.float64],
):
    """
    Plots the function values, gradient norms, and contour plots (if applicable).

    Parameters:
      f: The objective function.
      inital_point: Starting point of the optimization.
      condition: String label for the method (e.g., "SR1").
      f_vals: List of function values over iterations.
      grad_vals: List of gradient norms over iterations.
      path: Array of points visited during optimization.
    """
    # Generate a formatted name from the function's name
    fname_parts = f.__name__.split("_")
    formatted_fname = " ".join(word.capitalize() for word in fname_parts[:-1])
    formatted_point = "(" + ", ".join(map(str, inital_point)) + ")"

    # Plot Function Value vs. Iterations
    plt.figure()
    plt.plot(f_vals, label="Function Value", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title(f"{formatted_fname}; {condition}; $x_0={formatted_point}$")
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png")
    plt.close()

    # Plot Gradient Norm vs. Iterations
    plt.figure()
    plt.plot(grad_vals, label="Gradient Norm", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("||∇f(x)||")
    plt.legend()
    plt.title(f"{formatted_fname}; {condition}; $x_0={formatted_point}$")
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png")
    plt.close()

    # Create a Contour Plot for 2D problems
    if path.shape[1] == 2:
        # Set up a grid that covers the path with a margin of 1 unit
        x1_vals = np.linspace(np.min(path[:, 0]) - 1, np.max(path[:, 0]) + 1, 100)
        x2_vals = np.linspace(np.min(path[:, 1]) - 1, np.max(path[:, 1]) + 1, 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[f(np.array([x1, x2])) for x1 in x1_vals] for x2 in x2_vals])

        plt.figure()
        cp = plt.contour(X1, X2, Z, levels=30, cmap="viridis")
        plt.clabel(cp, inline=True, fontsize=8)
        plt.scatter(inital_point[0], inital_point[1], color="blue", label="Start", marker="o")
        plt.scatter(path[:, 0], path[:, 1], color="blue", marker=".")
        plt.scatter(path[-1, 0], path[-1, 1], color="blue", label="End", marker="x")

        # Draw arrows showing the optimization steps
        for i in range(len(path) - 1):
            plt.quiver(
                path[i, 0],
                path[i, 1],
                path[i+1, 0] - path[i, 0],
                path[i+1, 1] - path[i, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="red",
                width=0.005
            )

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.title(f"Contour — {formatted_fname}; {condition}; \n $x_0={formatted_point}$")
        plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
        plt.close()


