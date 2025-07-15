#!/usr/bin/env python
# Created by "Thieu" at 15:08, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class IGWO:
    """
    Incremental Grey Wolf Optimizer (I-GWO)

    Based on the paper: "I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer
    to solve global optimization problems" by Amir Seyyedabbasi and Farzad Kiani
    """

    def __init__(self, objective_function: Callable, dim: int, lb: float, ub: float,
                 max_iter: int = 500, population_size: int = 30, j: float = 1.5):
        """
        Initialize I-GWO algorithm

        Parameters:
        - objective_function: Function to minimize
        - dim: Dimension of the problem
        - lb: Lower bound
        - ub: Upper bound
        - max_iter: Maximum number of iterations
        - population_size: Number of wolves in the pack
        - j: Parameter for exploration enhancement (default 1.5)
        """
        self.objective_function = objective_function
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter
        self.population_size = population_size
        self.j = j

        # Initialize population
        self.positions = np.random.uniform(lb, ub, (population_size, dim))
        self.fitness = np.array([objective_function(pos) for pos in self.positions])

        # Best positions tracking
        self.best_positions = np.copy(self.positions)
        self.convergence_curve = []

    def update_parameters(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Update control parameters a, A, and C"""
        # Modified a parameter with j for enhanced exploration (Equation 16)
        a = 2 * (1 - (t / self.max_iter) ** self.j)

        # Generate random vectors for each wolf
        r1 = np.random.random((self.population_size, self.dim))
        r2 = np.random.random((self.population_size, self.dim))

        # Calculate A and C vectors (Equations 3, 4)
        A = 2 * a * r1 - a
        C = 2 * r2

        return A, C

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the I-GWO optimization algorithm

        Returns:
        - best_position: Best solution found
        - best_fitness: Best fitness value
        - convergence_curve: Fitness values over iterations
        """

        for t in range(self.max_iter):
            # Update control parameters
            A, C = self.update_parameters(t)

            # Sort wolves by fitness (ascending order for minimization)
            sorted_indices = np.argsort(self.fitness)
            sorted_positions = self.positions[sorted_indices]

            # Update alpha wolf (best wolf) - Equations 17, 18
            alpha_idx = sorted_indices[0]
            alpha_pos = sorted_positions[0]

            # Update positions for each wolf
            new_positions = np.zeros_like(self.positions)

            for n in range(self.population_size):
                if n == 0:
                    # Alpha wolf updates based on hunting mechanism
                    D_alpha = np.abs(C[n] * alpha_pos - self.positions[n])
                    new_positions[n] = alpha_pos - A[n] * D_alpha
                else:
                    # Other wolves update based on all previous wolves (Equation 19)
                    # Average position of all previous wolves (n-1 wolves)
                    avg_prev_positions = np.mean(sorted_positions[:n], axis=0)
                    new_positions[n] = avg_prev_positions

            # Boundary handling
            new_positions = np.clip(new_positions, self.lb, self.ub)

            # Update positions and fitness
            for i in range(self.population_size):
                new_fitness = self.objective_function(new_positions[i])
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_positions[i]
                    self.fitness[i] = new_fitness

            # Record best fitness
            best_fitness = np.min(self.fitness)
            self.convergence_curve.append(best_fitness)

            # Print progress
            if t % 50 == 0:
                print(f"I-GWO Iteration {t}: Best fitness = {best_fitness:.6f}")

        # Return best solution
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.convergence_curve


class ExGWO:
    """
    Expanded Grey Wolf Optimizer (Ex-GWO)

    Based on the paper: "I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer
    to solve global optimization problems" by Amir Seyyedabbasi and Farzad Kiani
    """

    def __init__(self, objective_function: Callable, dim: int, lb: float, ub: float,
                 max_iter: int = 500, population_size: int = 30):
        """
        Initialize Ex-GWO algorithm

        Parameters:
        - objective_function: Function to minimize
        - dim: Dimension of the problem
        - lb: Lower bound
        - ub: Upper bound
        - max_iter: Maximum number of iterations
        - population_size: Number of wolves in the pack
        """
        self.objective_function = objective_function
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter
        self.population_size = population_size

        # Initialize population
        self.positions = np.random.uniform(lb, ub, (population_size, dim))
        self.fitness = np.array([objective_function(pos) for pos in self.positions])

        # Best positions tracking
        self.convergence_curve = []

    def update_parameters(self, t: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """Update control parameters a, A, and C"""
        # Linear decrease of a from 2 to 0 (Equation 5)
        a = 2 * (1 - t / self.max_iter)

        # Generate random vectors for each wolf
        r1 = np.random.random((self.population_size, self.dim))
        r2 = np.random.random((self.population_size, self.dim))

        # Calculate A and C vectors (Equations 3, 4)
        A = 2 * a * r1 - a
        C = 2 * r2

        return a, A, C

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the Ex-GWO optimization algorithm

        Returns:
        - best_position: Best solution found
        - best_fitness: Best fitness value
        - convergence_curve: Fitness values over iterations
        """

        for t in range(self.max_iter):
            # Update control parameters
            a, A, C = self.update_parameters(t)

            # Sort wolves by fitness (ascending order for minimization)
            sorted_indices = np.argsort(self.fitness)
            sorted_positions = self.positions[sorted_indices]

            # Get alpha, beta, delta positions (first three best)
            alpha_pos = sorted_positions[0]
            beta_pos = sorted_positions[1]
            delta_pos = sorted_positions[2]

            # Update positions for each wolf
            new_positions = np.zeros_like(self.positions)

            for n in range(self.population_size):
                if n < 3:
                    # First three wolves (alpha, beta, delta) update using standard GWO
                    if n == 0:  # Alpha
                        D = np.abs(C[n] * alpha_pos - self.positions[n])
                        new_positions[n] = alpha_pos - A[n] * D
                    elif n == 1:  # Beta
                        D = np.abs(C[n] * beta_pos - self.positions[n])
                        new_positions[n] = beta_pos - A[n] * D
                    else:  # Delta
                        D = np.abs(C[n] * delta_pos - self.positions[n])
                        new_positions[n] = delta_pos - A[n] * D
                else:
                    # Other wolves update based on first three + previous wolves (Equation 15)
                    # Average of first three wolves + previously updated wolves
                    prev_wolves = sorted_positions[:n]
                    avg_position = np.mean(prev_wolves, axis=0)
                    new_positions[n] = avg_position

            # Boundary handling
            new_positions = np.clip(new_positions, self.lb, self.ub)

            # Update positions and fitness
            for i in range(self.population_size):
                new_fitness = self.objective_function(new_positions[i])
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_positions[i]
                    self.fitness[i] = new_fitness

            # Record best fitness
            best_fitness = np.min(self.fitness)
            self.convergence_curve.append(best_fitness)

            # Print progress
            if t % 50 == 0:
                print(f"Ex-GWO Iteration {t}: Best fitness = {best_fitness:.6f}")

        # Return best solution
        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.convergence_curve


# Standard GWO for comparison
class GWO:
    """Standard Grey Wolf Optimizer for comparison"""

    def __init__(self, objective_function: Callable, dim: int, lb: float, ub: float,
                 max_iter: int = 500, population_size: int = 30):
        self.objective_function = objective_function
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter
        self.population_size = population_size

        # Initialize population
        self.positions = np.random.uniform(lb, ub, (population_size, dim))
        self.fitness = np.array([objective_function(pos) for pos in self.positions])
        self.convergence_curve = []

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        for t in range(self.max_iter):
            # Update a
            a = 2 * (1 - t / self.max_iter)

            # Sort wolves
            sorted_indices = np.argsort(self.fitness)
            alpha_pos = self.positions[sorted_indices[0]]
            beta_pos = self.positions[sorted_indices[1]]
            delta_pos = self.positions[sorted_indices[2]]

            # Update positions
            for i in range(self.population_size):
                r1, r2 = np.random.random((2, self.dim))
                A1, A2, A3 = 2 * a * r1 - a, 2 * a * r1 - a, 2 * a * r1 - a
                C1, C2, C3 = 2 * r2, 2 * r2, 2 * r2

                D_alpha = np.abs(C1 * alpha_pos - self.positions[i])
                D_beta = np.abs(C2 * beta_pos - self.positions[i])
                D_delta = np.abs(C3 * delta_pos - self.positions[i])

                X1 = alpha_pos - A1 * D_alpha
                X2 = beta_pos - A2 * D_beta
                X3 = delta_pos - A3 * D_delta

                new_pos = (X1 + X2 + X3) / 3
                new_pos = np.clip(new_pos, self.lb, self.ub)

                new_fitness = self.objective_function(new_pos)
                if new_fitness < self.fitness[i]:
                    self.positions[i] = new_pos
                    self.fitness[i] = new_fitness

            best_fitness = np.min(self.fitness)
            self.convergence_curve.append(best_fitness)

            if t % 50 == 0:
                print(f"GWO Iteration {t}: Best fitness = {best_fitness:.6f}")

        best_idx = np.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx], self.convergence_curve


# Test functions
def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x ** 2)


def rosenbrock_function(x):
    """Rosenbrock function"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin_function(x):
    """Rastrigin function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def ackley_function(x):
    """Ackley function"""
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e


# Example usage and comparison
def compare_algorithms():
    """Compare I-GWO, Ex-GWO, and standard GWO"""
    print("=" * 60)
    print("COMPARISON OF GWO ALGORITHMS")
    print("=" * 60)

    # Test parameters
    dim = 10
    lb = -10
    ub = 10
    max_iter = 300
    population_size = 30

    # Test functions
    test_functions = [
        ("Sphere", sphere_function, 0),
        ("Rosenbrock", rosenbrock_function, 0),
        ("Rastrigin", rastrigin_function, 0),
        ("Ackley", ackley_function, 0)
    ]

    results = {}

    for func_name, func, optimal in test_functions:
        print(f"\n{func_name} Function (Optimal = {optimal}):")
        print("-" * 40)

        # I-GWO
        print("Running I-GWO...")
        igwo = IGWO(func, dim, lb, ub, max_iter, population_size)
        igwo_best_pos, igwo_best_fitness, igwo_curve = igwo.optimize()

        # Ex-GWO
        print("Running Ex-GWO...")
        exgwo = ExGWO(func, dim, lb, ub, max_iter, population_size)
        exgwo_best_pos, exgwo_best_fitness, exgwo_curve = exgwo.optimize()

        # Standard GWO
        print("Running Standard GWO...")
        gwo = GWO(func, dim, lb, ub, max_iter, population_size)
        gwo_best_pos, gwo_best_fitness, gwo_curve = gwo.optimize()

        # Store results
        results[func_name] = {
            'I-GWO': igwo_best_fitness,
            'Ex-GWO': exgwo_best_fitness,
            'GWO': gwo_best_fitness,
            'curves': {
                'I-GWO': igwo_curve,
                'Ex-GWO': exgwo_curve,
                'GWO': gwo_curve
            }
        }

        # Print results
        print(f"\nResults for {func_name}:")
        print(f"I-GWO:  Best fitness = {igwo_best_fitness:.6f}")
        print(f"Ex-GWO: Best fitness = {exgwo_best_fitness:.6f}")
        print(f"GWO:    Best fitness = {gwo_best_fitness:.6f}")

        # Plot convergence curves
        plt.figure(figsize=(10, 6))
        plt.plot(igwo_curve, label='I-GWO', linewidth=2)
        plt.plot(exgwo_curve, label='Ex-GWO', linewidth=2)
        plt.plot(gwo_curve, label='Standard GWO', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title(f'Convergence Curves - {func_name} Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"{'Function':<12} {'I-GWO':<12} {'Ex-GWO':<12} {'GWO':<12}")
    print("-" * 50)
    for func_name in results:
        print(f"{func_name:<12} {results[func_name]['I-GWO']:<12.6f} "
              f"{results[func_name]['Ex-GWO']:<12.6f} {results[func_name]['GWO']:<12.6f}")


if __name__ == "__main__":
    # Run comparison
    compare_algorithms()

    # Example of using individual algorithm
    print("\n" + "=" * 60)
    print("INDIVIDUAL ALGORITHM EXAMPLE")
    print("=" * 60)

    # Example with Sphere function
    print("\nExample: Minimizing Sphere function with I-GWO")
    igwo = IGWO(sphere_function, dim=5, lb=-5, ub=5, max_iter=100, population_size=20)
    best_pos, best_fitness, curve = igwo.optimize()

    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Function evaluations: {100 * 20}")