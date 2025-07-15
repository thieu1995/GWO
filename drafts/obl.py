#!/usr/bin/env python
# Created by "Thieu" at 00:27, 16/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class OGWO:
    """
    Opposition-based Learning Grey Wolf Optimizer (OGWO)

    Based on the paper: "Opposition-based learning grey wolf optimizer for global optimization"
    by Xiaobing Yu, WangYing Xu, ChenLiang Li (2021)
    """

    def __init__(self,
                 objective_function: Callable,
                 dimensions: int,
                 lower_bound: float,
                 upper_bound: float,
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 u: float = 2.0,  # nonlinear coefficient
                 jr: float = 0.05):  # jumping rate

        self.objective_function = objective_function
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.u = u  # nonlinear coefficient for equation (11)
        self.jr = jr  # jumping rate for OBL

        # Initialize population
        self.population = None
        self.fitness = None
        self.alpha_pos = None
        self.alpha_score = float('inf')
        self.beta_pos = None
        self.beta_score = float('inf')
        self.delta_pos = None
        self.delta_score = float('inf')

        # For tracking convergence
        self.convergence_curve = []

    def initialize_population(self):
        """Initialize population with opposition-based learning"""
        # Generate random initial population
        self.population = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dimensions)
        )

        # Generate opposition population using equation (12)
        opposition_population = self.lower_bound + self.upper_bound - self.population

        # Combine original and opposition populations
        combined_population = np.vstack([self.population, opposition_population])

        # Evaluate fitness for combined population
        combined_fitness = np.array([self.objective_function(ind) for ind in combined_population])

        # Select the best NP individuals
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        self.fitness = combined_fitness[best_indices]

        # Initialize alpha, beta, delta
        self.update_leaders()

    def update_leaders(self):
        """Update alpha, beta, delta wolves"""
        sorted_indices = np.argsort(self.fitness)

        self.alpha_pos = self.population[sorted_indices[0]].copy()
        self.alpha_score = self.fitness[sorted_indices[0]]

        self.beta_pos = self.population[sorted_indices[1]].copy()
        self.beta_score = self.fitness[sorted_indices[1]]

        self.delta_pos = self.population[sorted_indices[2]].copy()
        self.delta_score = self.fitness[sorted_indices[2]]

    def calculate_a(self, iteration: int) -> float:
        """Calculate coefficient 'a' using nonlinear function (equation 11)"""
        return 2 - 2 * (iteration / self.max_iterations) ** self.u

    def opposition_based_learning(self):
        """Apply opposition-based learning with jumping rate"""
        if np.random.random() < self.jr:
            # Generate opposition population
            opposition_population = self.lower_bound + self.upper_bound - self.population

            # Combine current and opposition populations
            combined_population = np.vstack([self.population, opposition_population])

            # Evaluate fitness for combined population
            combined_fitness = np.array([self.objective_function(ind) for ind in combined_population])

            # Select the best NP individuals
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            self.population = combined_population[best_indices]
            self.fitness = combined_fitness[best_indices]

            # Update leaders
            self.update_leaders()

    def update_position(self, wolf_pos: np.ndarray, a: float) -> np.ndarray:
        """Update wolf position according to GWO equations (6-8)"""
        # Generate random vectors
        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)

        # Calculate A and C vectors for each leader
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1 = np.random.random(self.dimensions)
        r2 = np.random.random(self.dimensions)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2

        # Calculate distances to leaders (equation 6)
        D_alpha = np.abs(C1 * self.alpha_pos - wolf_pos)
        D_beta = np.abs(C2 * self.beta_pos - wolf_pos)
        D_delta = np.abs(C3 * self.delta_pos - wolf_pos)

        # Calculate new positions according to leaders (equation 7)
        X1 = self.alpha_pos - A1 * D_alpha
        X2 = self.beta_pos - A2 * D_beta
        X3 = self.delta_pos - A3 * D_delta

        # Final position (equation 8)
        new_pos = (X1 + X2 + X3) / 3

        # Boundary handling
        new_pos = np.clip(new_pos, self.lower_bound, self.upper_bound)

        return new_pos

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, List[float]]:
        """Main optimization loop"""
        # Initialize population
        self.initialize_population()

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Calculate coefficient 'a' using nonlinear function
            a = self.calculate_a(iteration)

            # Update each wolf's position
            for i in range(self.population_size):
                new_pos = self.update_position(self.population[i], a)

                # Evaluate new position
                new_fitness = self.objective_function(new_pos)

                # Update position if better
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness

            # Apply opposition-based learning
            self.opposition_based_learning()

            # Update leaders
            self.update_leaders()

            # Store convergence information
            self.convergence_curve.append(self.alpha_score)

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.alpha_score:.6e}")

        return self.alpha_pos, self.alpha_score, self.convergence_curve

    def plot_convergence(self):
        """Plot convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve)
        plt.title('OGWO Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.yscale('log')
        plt.grid(True)
        plt.show()


# Test functions from the paper
def sphere_function(x):
    """Sphere function f1"""
    return np.sum(x ** 2)


def schwefel_2_22(x):
    """Schwefel 2.22 function f2"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def schwefel_1_2(x):
    """Schwefel 1.2 function f3"""
    return np.sum([(np.sum(x[:i + 1])) ** 2 for i in range(len(x))])


def schwefel_2_21(x):
    """Schwefel 2.21 function f4"""
    return np.max(np.abs(x))


def rosenbrock(x):
    """Rosenbrock function f5"""
    return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)])


def step_function(x):
    """Step function f6"""
    return np.sum(np.floor(x + 0.5) ** 2)


def quartic_noise(x):
    """Quartic function with noise f7"""
    return np.sum([i * x[i - 1] ** 4 for i in range(1, len(x) + 1)]) + np.random.random()


def schwefel_function(x):
    """Schwefel function f8"""
    return np.sum([-x[i] * np.sin(np.sqrt(np.abs(x[i]))) for i in range(len(x))])


def rastrigin(x):
    """Rastrigin function f9"""
    A = 10
    n = len(x)
    return A * n + np.sum([x[i] ** 2 - A * np.cos(2 * np.pi * x[i]) for i in range(n)])


def ackley(x):
    """Ackley function f10"""
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e


def griewank(x):
    """Griewank function f11"""
    return 1 + np.sum(x ** 2) / 4000 - np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])


# Example usage
if __name__ == "__main__":
    # Test with Sphere function
    print("Testing OGWO with Sphere function (f1)")
    print("=" * 50)

    # Create OGWO instance
    ogwo = OGWO(
        objective_function=sphere_function,
        dimensions=30,
        lower_bound=-100,
        upper_bound=100,
        population_size=50,
        max_iterations=1000,
        u=2.0,
        jr=0.05
    )

    # Run optimization
    best_position, best_fitness, convergence = ogwo.optimize(verbose=True)

    print(f"\nOptimization completed!")
    print(f"Best fitness: {best_fitness:.6e}")
    print(f"Best position: {best_position[:5]}...")  # Show first 5 dimensions

    # Plot convergence
    ogwo.plot_convergence()

    # Test with Rastrigin function
    print("\n" + "=" * 50)
    print("Testing OGWO with Rastrigin function (f9)")
    print("=" * 50)

    ogwo_rastrigin = OGWO(
        objective_function=rastrigin,
        dimensions=30,
        lower_bound=-5.12,
        upper_bound=5.12,
        population_size=50,
        max_iterations=1000,
        u=2.0,
        jr=0.05
    )

    best_position_r, best_fitness_r, convergence_r = ogwo_rastrigin.optimize(verbose=True)

    print(f"\nOptimization completed!")
    print(f"Best fitness: {best_fitness_r:.6e}")
    print(f"Best position: {best_position_r[:5]}...")

    # Plot convergence
    ogwo_rastrigin.plot_convergence()

    # Test with Ackley function
    print("\n" + "=" * 50)
    print("Testing OGWO with Ackley function (f10)")
    print("=" * 50)

    ogwo_ackley = OGWO(
        objective_function=ackley,
        dimensions=30,
        lower_bound=-32,
        upper_bound=32,
        population_size=50,
        max_iterations=1000,
        u=2.0,
        jr=0.05
    )

    best_position_a, best_fitness_a, convergence_a = ogwo_ackley.optimize(verbose=True)

    print(f"\nOptimization completed!")
    print(f"Best fitness: {best_fitness_a:.6e}")
    print(f"Best position: {best_position_a[:5]}...")

    # Plot convergence
    ogwo_ackley.plot_convergence()
