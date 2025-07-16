#!/usr/bin/env python
# Created by "Thieu" at 00:53, 16/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional


class ERGWO:
    """
    Efficient and Robust Grey Wolf Optimizer (ERGWO)

    An improved version of GWO with:
    1. Nonlinear variation of control parameter α
    2. Modified position-updating equation with dynamic weights
    """

    def __init__(self,
                 population_size: int = 30,
                 max_iterations: int = 500,
                 a_initial: float = 2.0,
                 a_final: float = 0.0,
                 lambda_param: float = 1.001):
        """
        Initialize ERGWO parameters

        Args:
            population_size: Number of wolves in the pack
            max_iterations: Maximum number of iterations
            a_initial: Initial value of parameter α
            a_final: Final value of parameter α
            lambda_param: Nonlinear modulation index (between 1.0001 and 1.005)
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.a_initial = a_initial
        self.a_final = a_final
        self.lambda_param = lambda_param

        # Initialize positions and fitness
        self.positions = None
        self.fitness = None
        self.dimension = None
        self.bounds = None

        # Best solutions (Alpha, Beta, Delta)
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        self.alpha_fitness = float('inf')
        self.beta_fitness = float('inf')
        self.delta_fitness = float('inf')

        # Convergence history
        self.convergence_history = []

    def initialize_population(self, dimension: int, bounds: Tuple[float, float]):
        """Initialize the wolf population randomly within bounds"""
        self.dimension = dimension
        self.bounds = bounds

        # Initialize positions randomly
        self.positions = np.random.uniform(
            bounds[0], bounds[1],
            (self.population_size, dimension)
        )

        # Initialize fitness array
        self.fitness = np.full(self.population_size, float('inf'))

        # Initialize best positions
        self.alpha_pos = np.zeros(dimension)
        self.beta_pos = np.zeros(dimension)
        self.delta_pos = np.zeros(dimension)

    def evaluate_fitness(self, objective_function: Callable):
        """Evaluate fitness for all wolves"""
        for i in range(self.population_size):
            self.fitness[i] = objective_function(self.positions[i])

    def update_leaders(self):
        """Update Alpha, Beta, and Delta wolves"""
        # Sort wolves by fitness
        sorted_indices = np.argsort(self.fitness)

        # Update Alpha (best)
        if self.fitness[sorted_indices[0]] < self.alpha_fitness:
            self.alpha_fitness = self.fitness[sorted_indices[0]]
            self.alpha_pos = self.positions[sorted_indices[0]].copy()

        # Update Beta (second best)
        if self.fitness[sorted_indices[1]] < self.beta_fitness:
            self.beta_fitness = self.fitness[sorted_indices[1]]
            self.beta_pos = self.positions[sorted_indices[1]].copy()

        # Update Delta (third best)
        if self.fitness[sorted_indices[2]] < self.delta_fitness:
            self.delta_fitness = self.fitness[sorted_indices[2]]
            self.delta_pos = self.positions[sorted_indices[2]].copy()

    def calculate_a_parameter(self, iteration: int) -> float:
        """
        Calculate nonlinear variation of control parameter α

        Equation (8): α(t) = a_initial - (a_initial - a_final) * λ^t
        """
        return self.a_initial - (self.a_initial - self.a_final) * (self.lambda_param ** iteration)

    def calculate_weights(self, X1: np.ndarray, X2: np.ndarray, X3: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate dynamic weights based on Euclidean distance

        Equations (9)-(11): Proportional weighting method
        """
        norm_X1 = np.linalg.norm(X1)
        norm_X2 = np.linalg.norm(X2)
        norm_X3 = np.linalg.norm(X3)

        total_norm = norm_X1 + norm_X2 + norm_X3

        if total_norm == 0:
            return 1 / 3, 1 / 3, 1 / 3

        w1 = norm_X1 / total_norm
        w2 = norm_X2 / total_norm
        w3 = norm_X3 / total_norm

        return w1, w2, w3

    def update_positions(self, iteration: int):
        """Update positions of all wolves using modified position-updating equation"""
        a = self.calculate_a_parameter(iteration)

        for i in range(self.population_size):
            # Generate random vectors
            r1 = np.random.random(self.dimension)
            r2 = np.random.random(self.dimension)

            # Calculate A and C vectors
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            r1 = np.random.random(self.dimension)
            r2 = np.random.random(self.dimension)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            r1 = np.random.random(self.dimension)
            r2 = np.random.random(self.dimension)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            # Calculate positions according to Alpha, Beta, Delta
            X1 = self.alpha_pos - A1 * np.abs(C1 * self.alpha_pos - self.positions[i])
            X2 = self.beta_pos - A2 * np.abs(C2 * self.beta_pos - self.positions[i])
            X3 = self.delta_pos - A3 * np.abs(C3 * self.delta_pos - self.positions[i])

            # Calculate dynamic weights
            w1, w2, w3 = self.calculate_weights(X1, X2, X3)

            # Modified position-updating equation (12)
            total_weight = w1 + w2 + w3
            if total_weight > 0:
                self.positions[i] = (w1 * X1 + w2 * X2 + w3 * X3) / total_weight
            else:
                self.positions[i] = (X1 + X2 + X3) / 3

            # Boundary handling
            self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

    def optimize(self, objective_function: Callable,
                 dimension: int,
                 bounds: Tuple[float, float],
                 verbose: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """
        Main optimization loop

        Args:
            objective_function: Function to minimize
            dimension: Problem dimension
            bounds: Search space bounds (min, max)
            verbose: Print progress information

        Returns:
            best_position: Best solution found
            best_fitness: Best fitness value
            convergence_history: Fitness evolution over iterations
        """
        # Initialize population
        self.initialize_population(dimension, bounds)

        # Evaluate initial fitness
        self.evaluate_fitness(objective_function)
        self.update_leaders()

        self.convergence_history = [self.alpha_fitness]

        if verbose:
            print(f"Initial best fitness: {self.alpha_fitness:.6f}")

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update positions
            self.update_positions(iteration)

            # Evaluate fitness
            self.evaluate_fitness(objective_function)

            # Update leaders
            self.update_leaders()

            # Record convergence
            self.convergence_history.append(self.alpha_fitness)

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.alpha_fitness:.6f}")

        if verbose:
            print(f"Final best fitness: {self.alpha_fitness:.6f}")

        return self.alpha_pos.copy(), self.alpha_fitness, self.convergence_history

    def plot_convergence(self):
        """Plot convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('ERGWO Convergence Curve')
        plt.grid(True)
        plt.show()


# Test functions for benchmarking
class BenchmarkFunctions:
    """Collection of benchmark functions for testing ERGWO"""

    @staticmethod
    def sphere(x):
        """Sphere function: f(x) = sum(x_i^2)"""
        return np.sum(x ** 2)

    @staticmethod
    def rosenbrock(x):
        """Rosenbrock function"""
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    @staticmethod
    def rastrigin(x):
        """Rastrigin function"""
        A = 10
        n = len(x)
        return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def ackley(x):
        """Ackley function"""
        n = len(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) -
                np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

    @staticmethod
    def griewank(x):
        """Griewank function"""
        return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))


# Example usage and testing
def main():
    """Example usage of ERGWO"""
    print("ERGWO (Efficient and Robust Grey Wolf Optimizer) Implementation")
    print("=" * 60)

    # Test on different benchmark functions
    functions = {
        'Sphere': (BenchmarkFunctions.sphere, (-100, 100), 0),
        'Rosenbrock': (BenchmarkFunctions.rosenbrock, (-30, 30), 0),
        'Rastrigin': (BenchmarkFunctions.rastrigin, (-5.12, 5.12), 0),
        'Ackley': (BenchmarkFunctions.ackley, (-32.768, 32.768), 0),
        'Griewank': (BenchmarkFunctions.griewank, (-600, 600), 0)
    }

    # Test parameters
    dimension = 30
    population_size = 30
    max_iterations = 500

    results = {}

    for func_name, (func, bounds, optimal) in functions.items():
        print(f"\nTesting {func_name} function (D={dimension})")
        print("-" * 40)

        # Initialize ERGWO
        ergwo = ERGWO(
            population_size=population_size,
            max_iterations=max_iterations,
            a_initial=2.0,
            a_final=0.0,
            lambda_param=1.001
        )

        # Optimize
        best_pos, best_fitness, convergence = ergwo.optimize(
            func, dimension, bounds, verbose=False
        )

        error = abs(best_fitness - optimal)
        results[func_name] = {
            'best_fitness': best_fitness,
            'error': error,
            'convergence': convergence
        }

        print(f"Best fitness: {best_fitness:.6e}")
        print(f"Error: {error:.6e}")
        print(f"Convergence in {len(convergence)} iterations")

    # Plot convergence for all functions
    plt.figure(figsize=(15, 10))
    for i, (func_name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(result['convergence'])
        plt.title(f'{func_name} Function')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.yscale('log')

    plt.tight_layout()
    plt.show()

    # Summary table
    print("\nSummary Results:")
    print("=" * 60)
    print(f"{'Function':<15} {'Best Fitness':<15} {'Error':<15}")
    print("-" * 60)
    for func_name, result in results.items():
        print(f"{func_name:<15} {result['best_fitness']:<15.6e} {result['error']:<15.6e}")


if __name__ == "__main__":
    main()
