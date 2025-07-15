#!/usr/bin/env python
# Created by "Thieu" at 13:41, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class ChaoticMaps:
    """
    Implementation of 10 chaotic maps used in CGWO algorithm
    """

    @staticmethod
    def bernoulli_map(x: float, a: float = 0.5) -> float:
        """Bernoulli map"""
        if 0 <= x <= a:
            return x / (1 - a)
        else:
            return (x - a) / a

    @staticmethod
    def logistic_map(x: float, a: float = 4.0) -> float:
        """Logistic map"""
        return a * x * (1 - x)

    @staticmethod
    def chebyshev_map(x: float, a: float = 4.0) -> float:
        """Chebyshev map"""
        return np.cos(a * np.arccos(x))

    @staticmethod
    def circle_map(x: float, a: float = 0.5, b: float = 0.2) -> float:
        """Circle map"""
        return (x + b - (a / (2 * np.pi)) * np.sin(2 * np.pi * x)) % 1

    @staticmethod
    def cubic_map(x: float, q: float = 2.59) -> float:
        """Cubic map"""
        return q * x * (1 - x * x)

    @staticmethod
    def icmic_map(x: float, a: float = 0.7) -> float:
        """Iterative chaotic map with infinite collapses"""
        if x == 0:
            return 0.1  # Avoid division by zero
        return np.abs(np.sin(a / x))

    @staticmethod
    def piecewise_map(x: float, a: float = 0.4) -> float:
        """Piecewise map"""
        if 0 <= x < a:
            return x / a
        elif a <= x < 0.5:
            return (x - a) / (0.5 - a)
        elif 0.5 <= x < 1 - a:
            return (1 - a - x) / (0.5 - a)
        else:
            return (1 - x) / a

    @staticmethod
    def singer_map(x: float, a: float = 1.07) -> float:
        """Singer map"""
        return a * (7.86 * x - 23.31 * x ** 2 + 28.75 * x ** 3 - 13.302875 * x ** 4)

    @staticmethod
    def sinusoidal_map(x: float, a: float = 2.3) -> float:
        """Sinusoidal map"""
        return a * x * x * np.sin(np.pi * x)

    @staticmethod
    def tent_map(x: float) -> float:
        """Tent map"""
        if x < 0.7:
            return x / 0.7
        else:
            return (10 / 3) * (1 - x)


class CGWO:
    """
    Chaotic Grey Wolf Optimization Algorithm
    """

    def __init__(self,
                 objective_function: Callable,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 30,
                 max_iterations: int = 100,
                 chaotic_map: str = 'chebyshev',
                 initial_chaotic_value: float = 0.7,
                 constraint_function: Optional[Callable] = None,
                 penalty_factor: float = 1000.0):
        """
        Initialize CGWO algorithm

        Parameters:
        -----------
        objective_function : Callable
            Function to optimize
        bounds : List[Tuple[float, float]]
            Search space bounds for each dimension
        population_size : int
            Number of wolves in the population
        max_iterations : int
            Maximum number of iterations
        chaotic_map : str
            Type of chaotic map to use
        initial_chaotic_value : float
            Initial value for chaotic map
        constraint_function : Optional[Callable]
            Constraint function (returns penalty if violated)
        penalty_factor : float
            Penalty factor for constraint violations
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.dim = len(bounds)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.chaotic_map = chaotic_map
        self.chaotic_value = initial_chaotic_value
        self.constraint_function = constraint_function
        self.penalty_factor = penalty_factor

        # Initialize chaotic map function
        self.chaotic_maps = {
            'bernoulli': ChaoticMaps.bernoulli_map,
            'logistic': ChaoticMaps.logistic_map,
            'chebyshev': ChaoticMaps.chebyshev_map,
            'circle': ChaoticMaps.circle_map,
            'cubic': ChaoticMaps.cubic_map,
            'icmic': ChaoticMaps.icmic_map,
            'piecewise': ChaoticMaps.piecewise_map,
            'singer': ChaoticMaps.singer_map,
            'sinusoidal': ChaoticMaps.sinusoidal_map,
            'tent': ChaoticMaps.tent_map
        }

        # Initialize population
        self.population = self._initialize_population()
        self.fitness = np.zeros(self.population_size)

        # Best positions
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)

        # Best fitness values
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')

        # History for plotting
        self.convergence_history = []

    def _initialize_population(self) -> np.ndarray:
        """Initialize population randomly within bounds"""
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.dim):
                population[i, j] = np.random.uniform(self.bounds[j][0], self.bounds[j][1])
        return population

    def _update_chaotic_value(self):
        """Update chaotic value using selected chaotic map"""
        if self.chaotic_map in self.chaotic_maps:
            self.chaotic_value = self.chaotic_maps[self.chaotic_map](self.chaotic_value)
            # Ensure chaotic value stays in [0, 1]
            self.chaotic_value = np.clip(self.chaotic_value, 0, 1)

    def _evaluate_fitness(self, position: np.ndarray) -> float:
        """Evaluate fitness with constraint handling"""
        # Ensure position is within bounds
        position = np.clip(position, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

        # Calculate objective function value
        fitness = self.objective_function(position)

        # Add penalty for constraint violations
        if self.constraint_function is not None:
            penalty = self.constraint_function(position)
            fitness += self.penalty_factor * penalty

        return fitness

    def _update_alpha_beta_delta(self):
        """Update alpha, beta, and delta wolves"""
        for i in range(self.population_size):
            fitness = self._evaluate_fitness(self.population[i])

            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness
                self.alpha_pos = self.population[i].copy()
            elif fitness < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness
                self.beta_pos = self.population[i].copy()
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = self.population[i].copy()

    def _update_position(self, iteration: int):
        """Update positions of all wolves"""
        # Calculate 'a' parameter (linearly decreasing from 2 to 0)
        a = 2 - iteration * (2 / self.max_iterations)

        for i in range(self.population_size):
            # Update chaotic value
            self._update_chaotic_value()

            # Use chaotic value to influence random vectors
            r1 = np.random.random(self.dim) * self.chaotic_value
            r2 = np.random.random(self.dim) * self.chaotic_value

            # Calculate A and C vectors
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            r1 = np.random.random(self.dim) * self.chaotic_value
            r2 = np.random.random(self.dim) * self.chaotic_value
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            r1 = np.random.random(self.dim) * self.chaotic_value
            r2 = np.random.random(self.dim) * self.chaotic_value
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            # Calculate distances
            D_alpha = np.abs(C1 * self.alpha_pos - self.population[i])
            D_beta = np.abs(C2 * self.beta_pos - self.population[i])
            D_delta = np.abs(C3 * self.delta_pos - self.population[i])

            # Calculate new positions
            X1 = self.alpha_pos - A1 * D_alpha
            X2 = self.beta_pos - A2 * D_beta
            X3 = self.delta_pos - A3 * D_delta

            # Update position
            self.population[i] = (X1 + X2 + X3) / 3

            # Apply bounds
            for j in range(self.dim):
                self.population[i, j] = np.clip(self.population[i, j],
                                                self.bounds[j][0],
                                                self.bounds[j][1])

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run the CGWO optimization algorithm

        Parameters:
        -----------
        verbose : bool
            Whether to print progress

        Returns:
        --------
        Tuple[np.ndarray, float]
            Best position and best fitness value
        """
        # Initialize alpha, beta, delta
        self._update_alpha_beta_delta()

        for iteration in range(self.max_iterations):
            # Update positions
            self._update_position(iteration)

            # Update alpha, beta, delta
            self._update_alpha_beta_delta()

            # Store convergence history
            self.convergence_history.append(self.alpha_score)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.alpha_score:.6f}")

        if verbose:
            print(f"Final result: Best fitness = {self.alpha_score:.6f}")
            print(f"Best position: {self.alpha_pos}")

        return self.alpha_pos, self.alpha_score

    def plot_convergence(self):
        """Plot convergence history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history)
        plt.title('CGWO Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()


# Example usage and test functions
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


def constraint_g1(x):
    """Example constraint function for G1 problem"""
    # G1 constraint: sum(2*x_i) - 10 <= 0
    constraint = np.sum(2 * x) - 10
    return max(0, constraint)


# Example test cases
if __name__ == "__main__":
    print("Testing CGWO Algorithm")
    print("=" * 50)

    # Test 1: Sphere function
    print("\n1. Testing Sphere function (2D)")
    bounds = [(-5, 5), (-5, 5)]
    cgwo = CGWO(sphere_function, bounds, population_size=30, max_iterations=100)
    best_pos, best_fitness = cgwo.optimize(verbose=False)
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fitness}")

    # Test 2: Rosenbrock function
    print("\n2. Testing Rosenbrock function (2D)")
    bounds = [(-2, 2), (-2, 2)]
    cgwo = CGWO(rosenbrock_function, bounds, population_size=30, max_iterations=200)
    best_pos, best_fitness = cgwo.optimize(verbose=False)
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fitness}")

    # Test 3: Constrained optimization
    print("\n3. Testing constrained optimization")
    bounds = [(-5, 5), (-5, 5)]
    cgwo = CGWO(sphere_function, bounds, population_size=30, max_iterations=100,
                constraint_function=constraint_g1, penalty_factor=1000)
    best_pos, best_fitness = cgwo.optimize(verbose=False)
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fitness}")

    # Test different chaotic maps
    print("\n4. Testing different chaotic maps on Sphere function")
    chaotic_maps = ['chebyshev', 'logistic', 'tent', 'sinusoidal']

    for map_name in chaotic_maps:
        cgwo = CGWO(sphere_function, [(-5, 5), (-5, 5)],
                    population_size=30, max_iterations=50,
                    chaotic_map=map_name)
        best_pos, best_fitness = cgwo.optimize(verbose=False)
        print(f"{map_name.capitalize()} map: Best fitness = {best_fitness:.6f}")

    # Plot convergence for Chebyshev map
    print("\n5. Plotting convergence for Chebyshev map")
    cgwo = CGWO(sphere_function, [(-5, 5), (-5, 5)],
                population_size=30, max_iterations=100,
                chaotic_map='chebyshev')
    best_pos, best_fitness = cgwo.optimize(verbose=False)
    cgwo.plot_convergence()
