from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

def backtracking_armijo(f, d_f, x, p, alpha_0=10, rho=0.75, c=1e-3):
    alpha = alpha_0
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(d_f(x), p):
        alpha *= rho
    return alpha
    
def backtracking_goldstein(f, d_f, x, p, alpha_0=10, rho=0.75, c1=0.001):
    alpha = alpha_0

    while alpha > 1e-6: # and num_steps < MAX_STEPS:
        new_x = x + alpha * p
        lhs = f(new_x)
        armijo_rhs = f(x) + c1 * alpha * np.dot(d_f(x), p)
        goldstein_rhs = f(x) + (1-c1) * alpha * np.dot(d_f(x), p)

        if lhs > armijo_rhs or lhs < goldstein_rhs:  # If Armijo condition fails (step is too large), reduce alpha
            alpha *= rho
        else:
            break  # Both conditions satisfied

    return alpha

def bisection_wolfe(f, d_f, x, p, c1=1e-3, c2=0.1, alpha_0=0, t=1, beta_0=1e6, eps=1e-6):
    alpha, beta = alpha_0, beta_0
    while True:
        new_x = x + t * p
        if f(new_x) > f(x) + c1 * t * np.dot(d_f(x), p):
            beta = t
        elif np.dot(d_f(new_x), p) < c2 * np.dot(d_f(x), p):
            alpha = t
        else:
            break
        t = 0.5 * (alpha + beta)
    return t

def plot_optimization_results(f, inital_point, condition, f_vals, grad_vals, path):
    """
    Plots the function values, gradient norms, and contour plots (if applicable).
    
    Parameters:
    - f: Objective function
    - inital_point: Initial starting point of the optimization
    - condition: Optimization condition used (for naming plots)
    - f_vals: List of function values per iteration
    - grad_vals: List of gradient norms per iteration
    - path: List/array of points traversed during optimization
    """
    fname_parts = f.__name__.split("_")  # Split on "_"
    formatted_fname = " ".join(word.capitalize() for word in fname_parts[:-1])  # Capitalize and exclude last word
    formatted_point = "(" + ", ".join(map(str, inital_point)) + ")"

    # Plot Function Value vs Iterations
    plt.figure()
    plt.plot(f_vals, label="Function Value")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.legend()
    plt.title(f"{formatted_fname}; {condition}; $x_0={formatted_point}$")
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png")
    plt.close()

    # Plot Gradient Norm vs Iterations
    plt.figure()
    plt.plot(grad_vals, label="Gradient Norm")
    plt.xlabel("Iteration")
    plt.ylabel("||∇f(x)||")
    plt.legend()
    plt.title(f"{formatted_fname}; {condition}; $x_0={formatted_point}$")
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png")
    plt.close()

    # Contour Plot (Only for 2D functions)
    if path.shape[1] == 2:
        x1_vals = np.linspace(min(path[:, 0]) - 1, max(path[:, 0]) + 1, 100)
        x2_vals = np.linspace(min(path[:, 1]) - 1, max(path[:, 1]) + 1, 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[f(np.array([x1, x2])) for x1 in x1_vals] for x2 in x2_vals])

        plt.figure()
        plt.contour(X1, X2, Z, levels=30, cmap="viridis")
        plt.scatter(inital_point[0], inital_point[1], color="blue", label="Start", marker="o")
        plt.scatter(path[:, 0], path[:, 1], color="blue", marker=".")
        plt.scatter(path[-1, 0], path[-1, 1], color="blue", label="End", marker="x")

        # Adding arrows to show the update directions
        for i in range(len(path) - 1):
            plt.quiver(
                path[i, 0], path[i, 1], 
                path[i+1, 0] - path[i, 0], path[i+1, 1] - path[i, 1], 
                angles="xy", scale_units="xy", scale=1, color="red", width=0.005
            )

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.title(f"Contour — {formatted_fname}; {condition}; $x_0={formatted_point}$")
        plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
        plt.close()


# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking-Armijo", "Backtracking-Goldstein", "Bisection"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    
    x = inital_point.copy()
    max_iters = int(1e4)
    eps = 1e-6
    
    f_vals, grad_vals, path = [], [], [x.copy()]
    # print(condition)
    # print('Grad at Start', np.linalg.norm(d_f(inital_point)))
    # alpha_arr = []

    for k in range(max_iters):
        grad = d_f(x)
        grad_norm = np.linalg.norm(grad)
        f_vals.append(f(x))
        grad_vals.append(grad_norm)
        
        if grad_norm < eps:
            break
        
        p = -grad # gradient function
        if condition == "Backtracking-Armijo":
            alpha = backtracking_armijo(f, d_f, x, p)
        elif condition == "Backtracking-Goldstein":
            alpha = backtracking_goldstein(f, d_f, x, p)
        elif condition == "Bisection":
            alpha = bisection_wolfe(f, d_f, x, p)
        else:
            raise ValueError("Invalid condition")
        
        x = x + alpha * p
        path.append(x.copy())  
        # alpha_arr.append(alpha)
        
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
    # print(alpha_arr[: 3], alpha_arr[-3: ])
    # print('Grad at End', np.linalg.norm(d_f(path[-1])))
    
    path = np.array(path)
    plot_optimization_results(f, inital_point, condition, f_vals, grad_vals, np.array(path))

    return x


def clip_grad(d):
    max_step_norm = 1.5  # Define a threshold for step size
    step_norm = np.linalg.norm(d)
    
    if step_norm > max_step_norm:
        d = (max_step_norm / step_norm) * d  # Scale down while maintaining direction
    return d
    
# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    
    x = inital_point.copy()
    max_iters = int(1e4)
    eps = 1e-6
    
    alpha = 0.001
    beta = 0.75
    f_vals, grad_vals, path = [], [], [x.copy()]

    # print(condition)
    # print('Grad at Start', np.linalg.norm(d_f(inital_point)))
    # tk_arr = []
    
    for k in range(max_iters):
        grad = d_f(x)
        grad_norm = np.linalg.norm(grad)
        f_vals.append(f(x))
        grad_vals.append(grad_norm)
        
        if grad_norm < eps:
            break
        
        hessian = d2_f(x)
        
        if condition == "Pure":
            # Standard Newton's step
            try:
                d_k = -np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                # print(f"Singular Hessian encountered at iteration {k}. Using gradient descent step.")
                d_k = -grad

            # print(f'Gradient at iter {k+1}', np.linalg.norm(d_k))
            d_k = clip_grad(d_k) # clip gradient to prevent exploding
            x = x + d_k
            # print(d_k)
    
        
        elif condition == "Damped":
            # Damped Newton's Method
            try:
                d_k = -np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                d_k = -grad  # Fall back to gradient descent if Hessian is singular

            t_k = 1.0
            while f(x) - f(x + t_k * d_k) < -alpha * t_k * np.dot(grad, d_k) and t_k > 1e-6:  
                t_k *= beta  # Reduce step size

            # tk_arr.append(t_k)
            x = x + t_k * d_k
        
        elif condition == "Levenberg-Marquardt":
            # Levenberg-Marquardt Algorithm
            eigvals = np.linalg.eigvalsh(hessian)
            eigval_min = np.min(eigvals)
            
            if eigval_min <= 0:
                u_k = -eigval_min + 0.1  # Adjust Hessian to be positive definite
                d_k = -np.linalg.solve(hessian + u_k * np.eye(len(x)), grad)
            else:
                d_k = -np.linalg.solve(hessian, grad)

            # print(f'Gradient at iter {k+1}', np.linalg.norm(d_k))
            d_k = clip_grad(d_k) # clip gradient to prevent exploding
            x = x + d_k
        
        elif condition == "Combined":
            # Combined Algorithm
            eigvals = np.linalg.eigvalsh(hessian)
            eigval_min = np.min(eigvals)
            
            if eigval_min <= 0:
                u_k = -eigval_min + 0.1
                d_k = -np.linalg.solve(hessian + u_k * np.eye(len(x)), grad)
            else:
                d_k = -np.linalg.solve(hessian, grad)

            # Use Backtracking with Armijo condition to find alpha_k
            alpha_k = backtracking_armijo(f, d_f, x, d_k)
            x = x + alpha_k * d_k
        
        else:
            raise ValueError("Invalid condition")

        path.append(x.copy())
    
    # print(tk_arr[: 3], tk_arr[-3: ])
    # print("First 5 Function values:", len(fvals_arr), fvals_arr[: 5])
    # print("Last 3 Function values:", fvals_arr[-3:])
    # print("First 5 Path values:", len(path), path[: 5])
    # print("Last 3 Path values:", path[-3: ])
    # print('Grad at End', np.linalg.norm(d_f(path[-1])))
    
    path = np.array(path)
    plot_optimization_results(f, inital_point, condition, f_vals, grad_vals, np.array(path))
    
    return x
