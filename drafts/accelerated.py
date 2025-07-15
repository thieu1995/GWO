#!/usr/bin/env python
# Created by "Thieu" at 16:53, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import special
import warnings

warnings.filterwarnings('ignore')


class AGWO:
    """
    Accelerated Grey Wolf Optimization (AGWO) Algorithm
    Based on the paper: "Accelerated grey wolf optimization for global optimization problems"
    """

    def __init__(self, population_size=100, max_iterations=100, dim=30,
                 lower_bound=-100, upper_bound=100, levy_beta=1.5):
        """
        Initialize AGWO parameters

        Parameters:
        - population_size: Number of wolves in the pack
        - max_iterations: Maximum number of iterations
        - dim: Problem dimension
        - lower_bound: Lower bound of search space
        - upper_bound: Upper bound of search space
        - levy_beta: Levy flight parameter (0 < beta <= 2)
        """
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.levy_beta = levy_beta

        # Initialize population
        self.population = np.random.uniform(lower_bound, upper_bound,
                                            (population_size, dim))
        self.fitness = np.zeros(population_size)

        # Best solutions (Alpha, Beta, Delta)
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')

        # Convergence tracking
        self.convergence_curve = []

    def levy_flight(self, dim):
        """
        Generate Levy flight step using Mantegna's algorithm
        """
        # Calculate sigma_u using Gamma function
        numerator = special.gamma(1 + self.levy_beta) * np.sin(np.pi * self.levy_beta / 2)
        denominator = special.gamma((1 + self.levy_beta) / 2) * self.levy_beta * (2 ** ((self.levy_beta - 1) / 2))
        sigma_u = (numerator / denominator) ** (1 / self.levy_beta)

        # Generate random samples
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)

        # Calculate step size
        step = u / (np.abs(v) ** (1 / self.levy_beta))

        return step

    def bfgs_local_search(self, x0, objective_func):
        """
        BFGS local search for intensification
        """
        try:
            # Use scipy's BFGS implementation
            bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
            result = minimize(objective_func, x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 10})
            return result.x if result.success else x0
        except:
            return x0

    def global_diversity_measure(self, best_solutions):
        """
        Calculate Global Diversity Measure (GDM)
        """
        if len(best_solutions) < 2:
            return 0.0

        diversity_sum = 0.0
        best_pos = best_solutions[0]

        for i in range(1, min(3, len(best_solutions))):
            diversity_sum += np.sum((best_pos - best_solutions[i]) ** 2)

        # Normalize by search space diagonal
        diagonal = np.sum((self.upper_bound - self.lower_bound) ** 2)
        gdm = diversity_sum / np.sqrt(diagonal) if diagonal > 0 else 0.0

        return gdm

    def local_diversity_measure(self, current_pos, best_pos, worst_pos):
        """
        Calculate Local Diversity Measure (LDM)
        """
        diversity_sum = np.sum((current_pos - best_pos) ** 2) + np.sum((current_pos - worst_pos) ** 2)

        # Normalize by search space diagonal
        diagonal = np.sum((self.upper_bound - self.lower_bound) ** 2)
        ldm = diversity_sum / np.sqrt(diagonal) if diagonal > 0 else 0.0

        return ldm

    def update_parameter_a(self, iteration):
        """
        Parameter tuning strategy for acceleration
        """
        # Tuned parameter 'a' decreases from 1.5 to 0
        a = 1.5 - (iteration ** 2 * 1.5) / (self.max_iterations ** 2)
        return max(0, a)

    def boundary_check(self, position):
        """
        Ensure wolves stay within search boundaries
        """
        position = np.clip(position, self.lower_bound, self.upper_bound)
        return position

    def optimize(self, objective_func):
        """
        Main optimization loop
        """
        # Initialize fitness values
        for i in range(self.population_size):
            self.fitness[i] = objective_func(self.population[i])

        # Find initial best solutions
        sorted_indices = np.argsort(self.fitness)
        self.alpha_pos = self.population[sorted_indices[0]].copy()
        self.alpha_score = self.fitness[sorted_indices[0]]

        if self.population_size > 1:
            self.beta_pos = self.population[sorted_indices[1]].copy()
            self.beta_score = self.fitness[sorted_indices[1]]

        if self.population_size > 2:
            self.delta_pos = self.population[sorted_indices[2]].copy()
            self.delta_score = self.fitness[sorted_indices[2]]

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update parameter a
            a = self.update_parameter_a(iteration)

            # Calculate diversity measures
            best_solutions = [self.alpha_pos, self.beta_pos, self.delta_pos]
            worst_pos = self.population[sorted_indices[-1]]
            gdm = self.global_diversity_measure(best_solutions)

            # Update each wolf
            for i in range(self.population_size):
                # Calculate coefficients
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                A1 = 2 * a * r1 - a
                A2 = 2 * a * r1 - a
                A3 = 2 * a * r1 - a

                C1 = 2 * r2
                C2 = 2 * r2
                C3 = 2 * r2

                # Calculate distances and positions
                D_alpha = np.abs(C1 * self.alpha_pos - self.population[i])
                D_beta = np.abs(C2 * self.beta_pos - self.population[i])
                D_delta = np.abs(C3 * self.delta_pos - self.population[i])

                X1 = self.alpha_pos - A1 * D_alpha
                X2 = self.beta_pos - A2 * D_beta
                X3 = self.delta_pos - A3 * D_delta

                # Standard GWO position update
                new_position = (X1 + X2 + X3) / 3

                # Calculate local diversity measure
                ldm = self.local_diversity_measure(self.population[i],
                                                   self.alpha_pos, worst_pos)

                # Diversity-based search strategy
                if ldm < gdm:
                    # Global search using Levy flight
                    levy_step = self.levy_flight(self.dim)
                    random_wolf = self.population[np.random.randint(self.population_size)]
                    step_size = 0.01 * levy_step * (self.population[i] - random_wolf)
                    new_position = self.population[i] + step_size
                else:
                    # Local search decision
                    psi = np.random.random()
                    if psi < 0.6:
                        # Apply BFGS local search
                        new_position = self.bfgs_local_search(self.alpha_pos, objective_func)
                        # Blend with current position
                        theta = 0.5  # Blending factor
                        new_position = self.population[i] - theta * new_position

                # Boundary checking
                new_position = self.boundary_check(new_position)

                # Update position if better
                new_fitness = objective_func(new_position)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness

            # Update alpha, beta, delta
            sorted_indices = np.argsort(self.fitness)

            if self.fitness[sorted_indices[0]] < self.alpha_score:
                self.alpha_score = self.fitness[sorted_indices[0]]
                self.alpha_pos = self.population[sorted_indices[0]].copy()

            if self.population_size > 1 and self.fitness[sorted_indices[1]] < self.beta_score:
                self.beta_score = self.fitness[sorted_indices[1]]
                self.beta_pos = self.population[sorted_indices[1]].copy()

            if self.population_size > 2 and self.fitness[sorted_indices[2]] < self.delta_score:
                self.delta_score = self.fitness[sorted_indices[2]]
                self.delta_pos = self.population[sorted_indices[2]].copy()

            # Store convergence data
            self.convergence_curve.append(self.alpha_score)

            # Print progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.alpha_score:.6e}")

        return self.alpha_pos, self.alpha_score


# Test functions from the paper
def sphere_function(x):
    """F1: Sphere function"""
    return np.sum(x ** 2)


def rosenbrock_function(x):
    """F5: Rosenbrock function"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin_function(x):
    """F9: Rastrigin function"""
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def ackley_function(x):
    """F10: Ackley function"""
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank_function(x):
    """F11: Griewank function"""
    sum_part = np.sum(x ** 2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1


# Example usage and testing
if __name__ == "__main__":
    # Test with Sphere function
    print("Testing AGWO with Sphere Function (F1)")
    print("=" * 50)

    # Initialize optimizer
    agwo = AGWO(population_size=50, max_iterations=100, dim=30,
                lower_bound=-100, upper_bound=100)

    # Optimize
    best_pos, best_fitness = agwo.optimize(sphere_function)

    print(f"\nOptimization Results:")
    print(f"Best fitness: {best_fitness:.6e}")
    print(f"Best position (first 5 dims): {best_pos[:5]}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(agwo.convergence_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('AGWO Convergence Curve - Sphere Function')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Test with other functions
    test_functions = [
        (sphere_function, "Sphere", -100, 100),
        (rastrigin_function, "Rastrigin", -5.12, 5.12),
        (ackley_function, "Ackley", -32, 32),
        (griewank_function, "Griewank", -600, 600)
    ]

    print("\n" + "=" * 60)
    print("Testing AGWO on multiple benchmark functions")
    print("=" * 60)

    results = []
    for func, name, lb, ub in test_functions:
        print(f"\nTesting {name} function...")
        agwo = AGWO(population_size=50, max_iterations=100, dim=30,
                    lower_bound=lb, upper_bound=ub)
        best_pos, best_fitness = agwo.optimize(func)
        results.append((name, best_fitness))
        print(f"{name}: Best fitness = {best_fitness:.6e}")

    print("\n" + "=" * 40)
    print("SUMMARY OF RESULTS")
    print("=" * 40)
    for name, fitness in results:
        print(f"{name:15}: {fitness:.6e}")