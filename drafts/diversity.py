#!/usr/bin/env python
# Created by "Thieu" at 17:09, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class DSGWO:
    """
    Diversity enhanced Strategy based Grey Wolf Optimizer (DSGWO)

    This implementation includes:
    1. Group-stage competition mechanism
    2. Exploration-exploitation balance mechanism
    """

    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]],
                 population_size: int = 30, max_iterations: int = 500,
                 exploration_phase_ratio: float = 0.4):
        """
        Initialize DSGWO algorithm

        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) tuples for each dimension
            population_size: Size of wolf population
            max_iterations: Maximum number of iterations
            exploration_phase_ratio: Ratio of iterations for exploration phase
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.exploration_phase_ratio = exploration_phase_ratio
        self.exploration_iterations = int(max_iterations * exploration_phase_ratio)

        # Problem dimensions
        self.dim = len(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]

        # Initialize population
        self.population = None
        self.fitness = None

        # Leading wolves
        self.alpha_pos = None
        self.alpha_fitness = float('inf')
        self.beta_pos = None
        self.beta_fitness = float('inf')
        self.delta_candidates = None
        self.delta_fitness = None

        # Convergence history
        self.convergence_curve = []
        self.best_fitness_history = []

    def initialize_population(self):
        """Initialize wolf population randomly within bounds"""
        self.population = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.population_size, self.dim)
        )

        # Evaluate initial fitness
        self.fitness = np.array([self.objective_function(wolf) for wolf in self.population])

    def group_stage_competition(self):
        """
        Group-stage competition mechanism:
        1. Divide population into 6 subgroups
        2. Select best wolf from each subgroup as delta candidates
        3. Set best overall as alpha
        4. Set delta candidate farthest from alpha as beta
        """
        # Divide population into 6 groups
        num_groups = 6
        group_size = self.population_size // num_groups

        self.delta_candidates = np.zeros((num_groups, self.dim))
        self.delta_fitness = np.zeros(num_groups)

        for i in range(num_groups):
            start_idx = i * group_size
            if i == num_groups - 1:  # Last group takes remaining wolves
                end_idx = self.population_size
            else:
                end_idx = (i + 1) * group_size

            # Get group members
            group_population = self.population[start_idx:end_idx]
            group_fitness = self.fitness[start_idx:end_idx]

            # Find best wolf in group
            best_idx = np.argmin(group_fitness)
            self.delta_candidates[i] = group_population[best_idx]
            self.delta_fitness[i] = group_fitness[best_idx]

        # Set alpha wolf (best among all delta candidates)
        alpha_idx = np.argmin(self.delta_fitness)
        self.alpha_pos = self.delta_candidates[alpha_idx].copy()
        self.alpha_fitness = self.delta_fitness[alpha_idx]

        # Set beta wolf (delta candidate farthest from alpha)
        distances = np.linalg.norm(self.delta_candidates - self.alpha_pos, axis=1)
        beta_idx = np.argmax(distances)
        self.beta_pos = self.delta_candidates[beta_idx].copy()
        self.beta_fitness = self.delta_fitness[beta_idx]

    def update_coefficients(self, iteration: int):
        """Update coefficient vectors A and C"""
        # Linear decrease from 2 to 0
        a = 2 - iteration * (2 / self.max_iterations)

        # Random vectors
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)

        A = 2 * a * r1 - a
        C = 2 * r2

        return A, C

    def exploration_phase_update(self, wolf_pos: np.ndarray, iteration: int):
        """
        Exploration phase: Update position using two randomly selected delta candidates
        """
        # Randomly select two delta candidates
        selected_indices = np.random.choice(len(self.delta_candidates), 2, replace=False)
        delta1 = self.delta_candidates[selected_indices[0]]
        delta2 = self.delta_candidates[selected_indices[1]]

        # Update coefficients
        A1, C1 = self.update_coefficients(iteration)
        A2, C2 = self.update_coefficients(iteration)

        # Calculate distances and new positions
        D1 = np.abs(C1 * delta1 - wolf_pos)
        X1 = delta1 - A1 * D1

        D2 = np.abs(C2 * delta2 - wolf_pos)
        X2 = delta2 - A2 * D2

        # Final position (average of two positions)
        new_pos = (X1 + X2) / 2

        return new_pos

    def exploitation_phase_update(self, wolf_pos: np.ndarray, iteration: int):
        """
        Exploitation phase: Update position using classical GWO approach
        """
        # Get third best delta candidate as delta wolf
        sorted_indices = np.argsort(self.delta_fitness)
        delta_pos = self.delta_candidates[sorted_indices[2]] if len(sorted_indices) >= 3 else self.delta_candidates[
            sorted_indices[-1]]

        # Update coefficients
        A1, C1 = self.update_coefficients(iteration)
        A2, C2 = self.update_coefficients(iteration)
        A3, C3 = self.update_coefficients(iteration)

        # Calculate distances
        D_alpha = np.abs(C1 * self.alpha_pos - wolf_pos)
        D_beta = np.abs(C2 * self.beta_pos - wolf_pos)
        D_delta = np.abs(C3 * delta_pos - wolf_pos)

        # Calculate candidate positions
        X1 = self.alpha_pos - A1 * D_alpha
        X2 = self.beta_pos - A2 * D_beta
        X3 = delta_pos - A3 * D_delta

        # Final position (average of three positions)
        new_pos = (X1 + X2 + X3) / 3

        return new_pos

    def bound_position(self, position: np.ndarray):
        """Ensure position is within bounds"""
        return np.clip(position, self.lower_bounds, self.upper_bounds)

    def optimize(self, verbose: bool = True):
        """
        Main optimization loop

        Args:
            verbose: Whether to print progress

        Returns:
            Tuple of (best_position, best_fitness, convergence_curve)
        """
        # Initialize population
        self.initialize_population()

        # Initial group-stage competition
        self.group_stage_competition()

        if verbose:
            print(f"Initial best fitness: {self.alpha_fitness:.6f}")

        # Main optimization loop
        for iteration in range(self.max_iterations):

            # Update each wolf
            new_population = np.zeros_like(self.population)

            for i in range(self.population_size):
                if iteration < self.exploration_iterations:
                    # Exploration phase
                    new_pos = self.exploration_phase_update(self.population[i], iteration)
                else:
                    # Exploitation phase
                    new_pos = self.exploitation_phase_update(self.population[i], iteration)

                # Apply bounds
                new_population[i] = self.bound_position(new_pos)

            # Update population
            self.population = new_population

            # Evaluate fitness
            self.fitness = np.array([self.objective_function(wolf) for wolf in self.population])

            # Update leading wolves using group-stage competition
            self.group_stage_competition()

            # Store convergence data
            self.convergence_curve.append(self.alpha_fitness)
            self.best_fitness_history.append(self.alpha_fitness)

            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.alpha_fitness:.6f}")

        if verbose:
            print(f"Final best fitness: {self.alpha_fitness:.6f}")
            print(f"Best position: {self.alpha_pos}")

        return self.alpha_pos, self.alpha_fitness, self.convergence_curve


# Test functions for benchmarking
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
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)

    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))

    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)


# Example usage and testing
if __name__ == "__main__":
    # Test on Sphere function
    print("Testing DSGWO on Sphere Function")
    print("=" * 50)

    # Define problem
    dimensions = 10
    bounds = [(-100, 100)] * dimensions

    # Create DSGWO instance
    dsgwo = DSGWO(
        objective_function=sphere_function,
        bounds=bounds,
        population_size=30,
        max_iterations=500,
        exploration_phase_ratio=0.4
    )

    # Optimize
    best_pos, best_fitness, convergence = dsgwo.optimize(verbose=True)

    print(f"\nOptimization Results:")
    print(f"Best fitness: {best_fitness:.10f}")
    print(f"Best position: {best_pos}")

    # Plot convergence curve
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(convergence)
    plt.title('Convergence Curve - Sphere Function')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.grid(True)

    # Test on other functions
    test_functions = [
        (rosenbrock_function, "Rosenbrock", [(-30, 30)] * dimensions),
        (rastrigin_function, "Rastrigin", [(-5.12, 5.12)] * dimensions),
        (ackley_function, "Ackley", [(-32.768, 32.768)] * dimensions)
    ]

    for i, (func, name, bounds_func) in enumerate(test_functions, 2):
        print(f"\nTesting DSGWO on {name} Function")
        print("=" * 50)

        dsgwo_func = DSGWO(
            objective_function=func,
            bounds=bounds_func,
            population_size=30,
            max_iterations=500,
            exploration_phase_ratio=0.4
        )

        best_pos, best_fitness, convergence = dsgwo_func.optimize(verbose=False)

        print(f"Best fitness: {best_fitness:.10f}")

        plt.subplot(2, 2, i)
        plt.plot(convergence)
        plt.title(f'Convergence Curve - {name} Function')
        plt.xlabel('Iterations')
        plt.ylabel('Best Fitness')
        plt.yscale('log')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Performance comparison example
    print("\nPerformance Analysis:")
    print("=" * 50)

    # Run multiple trials
    num_trials = 10
    results = []

    for trial in range(num_trials):
        dsgwo_trial = DSGWO(
            objective_function=sphere_function,
            bounds=[(-100, 100)] * 10,
            population_size=30,
            max_iterations=500
        )

        _, fitness, _ = dsgwo_trial.optimize(verbose=False)
        results.append(fitness)

    results = np.array(results)
    print(f"Mean fitness over {num_trials} trials: {np.mean(results):.2e}")
    print(f"Std deviation: {np.std(results):.2e}")
    print(f"Best fitness: {np.min(results):.2e}")
    print(f"Worst fitness: {np.max(results):.2e}")