#!/usr/bin/env python
# Created by "Thieu" at 23:36, 15/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import random


class IGWO:
    """
    Improved Grey Wolf Optimizer (IGWO) implementation
    Based on: "A better exploration strategy in Grey Wolf Optimizer"
    by Jagdish Chand Bansal and Shitu Singh
    """

    def __init__(self,
                 objective_function: Callable[[np.ndarray], float],
                 dimension: int,
                 lower_bound: float,
                 upper_bound: float,
                 population_size: int = 30,
                 max_iterations: int = 500):
        """
        Initialize IGWO optimizer

        Args:
            objective_function: Function to minimize
            dimension: Problem dimension
            lower_bound: Lower bound of search space
            upper_bound: Upper bound of search space
            population_size: Number of wolves in population
            max_iterations: Maximum number of iterations
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.lb = lower_bound
        self.ub = upper_bound
        self.population_size = population_size
        self.max_iterations = max_iterations

        # Initialize population
        self.population = np.random.uniform(self.lb, self.ub,
                                            (self.population_size, self.dimension))

        # Initialize fitness values
        self.fitness = np.array([self.objective_function(wolf) for wolf in self.population])

        # Initialize alpha, beta, delta wolves
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None
        self.alpha_fitness = float('inf')
        self.beta_fitness = float('inf')
        self.delta_fitness = float('inf')

        # Convergence tracking
        self.convergence_curve = []

    def update_alpha_beta_delta(self):
        """Update alpha, beta, and delta wolves based on fitness"""
        # Sort indices by fitness
        sorted_indices = np.argsort(self.fitness)

        # Update alpha wolf (best)
        self.alpha_wolf = self.population[sorted_indices[0]].copy()
        self.alpha_fitness = self.fitness[sorted_indices[0]]

        # Update beta wolf (second best)
        self.beta_wolf = self.population[sorted_indices[1]].copy()
        self.beta_fitness = self.fitness[sorted_indices[1]]

        # Update delta wolf (third best)
        self.delta_wolf = self.population[sorted_indices[2]].copy()
        self.delta_fitness = self.fitness[sorted_indices[2]]

    def opposition_based_learning(self, wolf: np.ndarray) -> np.ndarray:
        """
        Apply Opposition-Based Learning (OBL)

        Args:
            wolf: Wolf position

        Returns:
            Opposite wolf position
        """
        return (self.lb + self.ub) - wolf

    def explorative_equation(self, wolf: np.ndarray, t: int) -> np.ndarray:
        """
        Apply explorative equation for better exploration

        Args:
            wolf: Current wolf position
            t: Current iteration

        Returns:
            New wolf position
        """
        r1, r2, r3, r4, r5 = np.random.rand(5)

        # Select random wolf from population
        rand_index = np.random.randint(0, self.population_size)
        x_rand = self.population[rand_index]

        # Calculate average position of all wolves
        x_avg = np.mean(self.population, axis=0)

        if r5 >= 0.5:
            # Exploration around random wolf
            new_position = x_rand - r1 * np.abs(x_rand - 2 * r2 * wolf)
        else:
            # Exploration around alpha wolf
            new_position = (self.alpha_wolf - x_avg) - r3 * (self.lb + r4 * (self.ub - self.lb))

        # Apply boundary constraints
        new_position = np.clip(new_position, self.lb, self.ub)

        return new_position

    def classical_gwo_update(self, wolf: np.ndarray, a: float) -> np.ndarray:
        """
        Classical GWO position update

        Args:
            wolf: Current wolf position
            a: Parameter a (linearly decreasing from 2 to 0)

        Returns:
            Updated wolf position
        """
        # Random vectors
        r1, r2 = np.random.rand(2, self.dimension)
        r1_prime, r2_prime = np.random.rand(2, self.dimension)
        r1_double_prime, r2_double_prime = np.random.rand(2, self.dimension)

        # Calculate A and C vectors for each leader
        A1 = 2 * a * r1 - a
        A2 = 2 * a * r1_prime - a
        A3 = 2 * a * r1_double_prime - a

        C1 = 2 * r2
        C2 = 2 * r2_prime
        C3 = 2 * r2_double_prime

        # Calculate distances
        D_alpha = np.abs(C1 * self.alpha_wolf - wolf)
        D_beta = np.abs(C2 * self.beta_wolf - wolf)
        D_delta = np.abs(C3 * self.delta_wolf - wolf)

        # Calculate new positions based on alpha, beta, delta
        X1 = self.alpha_wolf - A1 * D_alpha
        X2 = self.beta_wolf - A2 * D_beta
        X3 = self.delta_wolf - A3 * D_delta

        # Update position (average of three positions)
        new_position = (X1 + X2 + X3) / 3

        # Apply boundary constraints
        new_position = np.clip(new_position, self.lb, self.ub)

        return new_position

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run IGWO optimization

        Returns:
            best_solution: Best solution found
            best_fitness: Best fitness value
            convergence_curve: Convergence history
        """

        for t in range(self.max_iterations):
            # Update alpha, beta, delta wolves
            self.update_alpha_beta_delta()

            # Calculate parameter a (linearly decreasing from 2 to 0)
            a = 2 - 2 * t / self.max_iterations

            # Update each wolf position
            for i in range(self.population_size):
                # Try explorative equation first
                new_position_explorative = self.explorative_equation(self.population[i], t)
                new_fitness_explorative = self.objective_function(new_position_explorative)

                # Greedy selection for explorative equation
                if new_fitness_explorative <= self.fitness[i]:
                    self.population[i] = new_position_explorative
                    self.fitness[i] = new_fitness_explorative
                else:
                    # Use classical GWO update if explorative equation fails
                    new_position_classical = self.classical_gwo_update(self.population[i], a)
                    new_fitness_classical = self.objective_function(new_position_classical)

                    # Greedy selection for classical update
                    if new_fitness_classical <= self.fitness[i]:
                        self.population[i] = new_position_classical
                        self.fitness[i] = new_fitness_classical

            # Apply Opposition-Based Learning (OBL) for leading wolves
            self.update_alpha_beta_delta()

            # Generate opposite solutions for alpha, beta, delta
            opposite_alpha = self.opposition_based_learning(self.alpha_wolf)
            opposite_beta = self.opposition_based_learning(self.beta_wolf)
            opposite_delta = self.opposition_based_learning(self.delta_wolf)

            # Evaluate opposite solutions
            opposite_fitness = [
                self.objective_function(opposite_alpha),
                self.objective_function(opposite_beta),
                self.objective_function(opposite_delta)
            ]

            # Replace worst 3 wolves with opposite solutions if they are better
            worst_indices = np.argsort(self.fitness)[-3:]
            opposite_solutions = [opposite_alpha, opposite_beta, opposite_delta]

            for j, idx in enumerate(worst_indices):
                if opposite_fitness[j] < self.fitness[idx]:
                    self.population[idx] = opposite_solutions[j]
                    self.fitness[idx] = opposite_fitness[j]

            # Update convergence curve
            self.convergence_curve.append(self.alpha_fitness)

            # Print progress
            if t % 50 == 0:
                print(f"Iteration {t}: Best fitness = {self.alpha_fitness:.6f}")

        return self.alpha_wolf, self.alpha_fitness, self.convergence_curve


# Test functions from the paper
def sphere_function(x):
    """Sphere function f1"""
    return np.sum(x ** 2)


def schwefel_2_22(x):
    """Schwefel's problem 2.22 f2"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def rastrigin_function(x):
    """Rastrigin function f9"""
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def ackley_function(x):
    """Ackley function f10"""
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e


def griewank_function(x):
    """Griewank function f11"""
    sum_term = np.sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1


# Example usage
if __name__ == "__main__":
    # Test on Sphere function
    print("Testing IGWO on Sphere function...")
    igwo = IGWO(
        objective_function=sphere_function,
        dimension=30,
        lower_bound=-100,
        upper_bound=100,
        population_size=30,
        max_iterations=500
    )

    best_solution, best_fitness, convergence = igwo.optimize()

    print(f"\nFinal Results:")
    print(f"Best fitness: {best_fitness:.10f}")
    print(f"Best solution (first 5 dimensions): {best_solution[:5]}")

    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(convergence)
    plt.title('IGWO Convergence Curve - Sphere Function')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Test on other functions
    test_functions = [
        ("Rastrigin", rastrigin_function, -5.12, 5.12),
        ("Ackley", ackley_function, -32, 32),
        ("Griewank", griewank_function, -600, 600)
    ]

    print("\n" + "=" * 50)
    print("Testing on multiple benchmark functions:")
    print("=" * 50)

    for name, func, lb, ub in test_functions:
        print(f"\nTesting {name} function...")
        igwo_test = IGWO(
            objective_function=func,
            dimension=30,
            lower_bound=lb,
            upper_bound=ub,
            population_size=30,
            max_iterations=300
        )

        best_sol, best_fit, conv = igwo_test.optimize()
        print(f"{name} - Best fitness: {best_fit:.10f}")
