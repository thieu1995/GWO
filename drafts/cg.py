#!/usr/bin/env python
# Created by "Thieu" at 01:03, 16/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm
import warnings

warnings.filterwarnings('ignore')


class CGGWO:
    def __init__(self, obj_func, dim, bounds, pop_size=30, max_iter=200):
        """
        Cauchy-Gaussian Grey Wolf Optimization Algorithm

        Parameters:
        - obj_func: Objective function to minimize
        - dim: Problem dimension
        - bounds: List of [lower_bound, upper_bound] for each dimension
        - pop_size: Population size (default: 30)
        - max_iter: Maximum number of iterations (default: 200)
        """
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter

        # Initialize population
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.pop_size, self.dim)
        )

        # Initialize fitness values
        self.fitness = np.array([self.obj_func(ind) for ind in self.population])

        # Initialize alpha, beta, delta wolves
        self.alpha_pos = np.zeros(self.dim)
        self.beta_pos = np.zeros(self.dim)
        self.delta_pos = np.zeros(self.dim)

        self.alpha_fitness = float('inf')
        self.beta_fitness = float('inf')
        self.delta_fitness = float('inf')

        # History for convergence analysis
        self.convergence_history = []

    def update_leaders(self):
        """Update alpha, beta, delta wolves based on fitness"""
        # Sort indices by fitness
        sorted_indices = np.argsort(self.fitness)

        # Update alpha wolf (best)
        if self.fitness[sorted_indices[0]] < self.alpha_fitness:
            self.alpha_fitness = self.fitness[sorted_indices[0]]
            self.alpha_pos = self.population[sorted_indices[0]].copy()

        # Update beta wolf (second best)
        if self.fitness[sorted_indices[1]] < self.beta_fitness:
            self.beta_fitness = self.fitness[sorted_indices[1]]
            self.beta_pos = self.population[sorted_indices[1]].copy()

        # Update delta wolf (third best)
        if self.fitness[sorted_indices[2]] < self.delta_fitness:
            self.delta_fitness = self.fitness[sorted_indices[2]]
            self.delta_pos = self.population[sorted_indices[2]].copy()

    def cauchy_gaussian_mutation(self, leader_pos, leader_fitness, t):
        """
        Apply Cauchy-Gaussian mutation to leader wolves

        Parameters:
        - leader_pos: Position of leader wolf
        - leader_fitness: Fitness of leader wolf
        - t: Current iteration
        """
        # Calculate dynamic parameters (equations 11 and 12)
        epsilon1 = 1 - (t ** 2) / (self.max_iter ** 2)
        epsilon2 = (t ** 2) / (self.max_iter ** 2)

        # Calculate sigma (equation 9)
        if abs(self.alpha_fitness) > 1e-10:
            sigma = np.exp((leader_fitness - self.alpha_fitness) / abs(self.alpha_fitness))
        else:
            sigma = 1.0

        # Generate Cauchy and Gaussian random variables
        cauchy_random = cauchy.rvs(loc=0, scale=sigma, size=self.dim)
        gaussian_random = norm.rvs(loc=0, scale=sigma, size=self.dim)

        # Apply mutation (equation 8)
        mutated_pos = leader_pos * (1 + epsilon1 * cauchy_random + epsilon2 * gaussian_random)

        # Ensure bounds
        mutated_pos = np.clip(mutated_pos, self.bounds[:, 0], self.bounds[:, 1])

        return mutated_pos

    def greedy_selection(self, original_pos, mutated_pos):
        """Greedy selection mechanism"""
        original_fitness = self.obj_func(original_pos)
        mutated_fitness = self.obj_func(mutated_pos)

        if mutated_fitness <= original_fitness:
            return mutated_pos, mutated_fitness
        else:
            return original_pos, original_fitness

    def improved_search_strategy(self, wolf_pos, wolf_fitness, t):
        """
        Improved search strategy for all wolves

        Parameters:
        - wolf_pos: Current position of wolf
        - wolf_fitness: Current fitness of wolf
        - t: Current iteration
        """
        # Calculate average position of population
        avg_pos = np.mean(self.population, axis=0)

        # Random parameters
        r1, r2, r3, r4, r5 = np.random.random(5)

        # Select random wolf from population
        rand_idx = np.random.randint(0, self.pop_size)
        rand_pos = self.population[rand_idx]

        # Apply improved search strategy (equation 13)
        if r5 >= 0.5:
            # First strategy
            new_pos = rand_pos - r1 * np.abs(rand_pos - 2 * r2 * wolf_pos)
        else:
            # Second strategy
            new_pos = (self.alpha_pos - avg_pos -
                       r3 * (self.bounds[:, 0] + r4 * (self.bounds[:, 1] - self.bounds[:, 0])))

        # Ensure bounds
        new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

        # Greedy selection (equation 14)
        new_fitness = self.obj_func(new_pos)
        if new_fitness <= wolf_fitness:
            return new_pos, new_fitness
        else:
            return wolf_pos, wolf_fitness

    def traditional_gwo_update(self, wolf_pos, a):
        """Traditional GWO position update"""
        # Random parameters
        r1, r2 = np.random.random(2)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        r1, r2 = np.random.random(2)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2

        r1, r2 = np.random.random(2)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2

        # Calculate distances (equation 5)
        D_alpha = np.abs(C1 * self.alpha_pos - wolf_pos)
        D_beta = np.abs(C2 * self.beta_pos - wolf_pos)
        D_delta = np.abs(C3 * self.delta_pos - wolf_pos)

        # Update positions (equation 6)
        X1 = self.alpha_pos - A1 * D_alpha
        X2 = self.beta_pos - A2 * D_beta
        X3 = self.delta_pos - A3 * D_delta

        # Final position (equation 7)
        new_pos = (X1 + X2 + X3) / 3

        # Ensure bounds
        new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

        return new_pos

    def optimize(self):
        """Main optimization loop"""
        # Initialize leaders
        self.update_leaders()

        for t in range(self.max_iter):
            # Parameter a decreases linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)

            # Apply Cauchy-Gaussian mutation to leaders
            # Alpha wolf mutation
            mutated_alpha = self.cauchy_gaussian_mutation(
                self.alpha_pos, self.alpha_fitness, t
            )
            self.alpha_pos, self.alpha_fitness = self.greedy_selection(
                self.alpha_pos, mutated_alpha
            )

            # Beta wolf mutation
            mutated_beta = self.cauchy_gaussian_mutation(
                self.beta_pos, self.beta_fitness, t
            )
            self.beta_pos, self.beta_fitness = self.greedy_selection(
                self.beta_pos, mutated_beta
            )

            # Delta wolf mutation
            mutated_delta = self.cauchy_gaussian_mutation(
                self.delta_pos, self.delta_fitness, t
            )
            self.delta_pos, self.delta_fitness = self.greedy_selection(
                self.delta_pos, mutated_delta
            )

            # Update each wolf in population
            for i in range(self.pop_size):
                # Apply improved search strategy
                new_pos, new_fitness = self.improved_search_strategy(
                    self.population[i], self.fitness[i], t
                )

                # If improved search doesn't help, use traditional GWO
                if new_fitness >= self.fitness[i]:
                    new_pos = self.traditional_gwo_update(self.population[i], a)
                    new_fitness = self.obj_func(new_pos)

                # Update if better
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_pos
                    self.fitness[i] = new_fitness

            # Update leaders
            self.update_leaders()

            # Store convergence history
            self.convergence_history.append(self.alpha_fitness)

            # Print progress
            if t % 50 == 0:
                print(f"Iteration {t}: Best fitness = {self.alpha_fitness:.6f}")

        return self.alpha_pos, self.alpha_fitness, self.convergence_history


# Test functions from the paper
def sphere_function(x):
    """F1: Sphere function"""
    return np.sum(x ** 2)


def schwefel_function(x):
    """F2: Schwefel's function"""
    return np.sum([np.sum(x[:i + 1]) ** 2 for i in range(len(x))])


def max_function(x):
    """F3: Max function"""
    return np.max(np.abs(x))


def step_function(x):
    """F4: Step function"""
    return np.sum(np.floor(x + 0.5) ** 2)


def quartic_function(x):
    """F5: Quartic function with noise"""
    return np.sum([i * x[i - 1] ** 4 for i in range(1, len(x) + 1)]) + np.random.random()


def rastrigin_function(x):
    """F6: Rastrigin function"""
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley_function(x):
    """F7: Ackley function"""
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) -
            np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def griewank_function(x):
    """F8: Griewank function"""
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))


# Example usage
if __name__ == "__main__":
    # Test with Sphere function
    print("Testing CG-GWO on Sphere Function (F1)")
    print("=" * 50)

    # Define problem
    dim = 30
    bounds = [[-100, 100]] * dim

    # Create optimizer
    optimizer = CGGWO(
        obj_func=sphere_function,
        dim=dim,
        bounds=bounds,
        pop_size=30,
        max_iter=200
    )

    # Run optimization
    best_pos, best_fitness, history = optimizer.optimize()

    print(f"\nFinal Results:")
    print(f"Best fitness: {best_fitness:.10f}")
    print(f"Best position (first 5 dimensions): {best_pos[:5]}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(history, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('CG-GWO Convergence on Sphere Function')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Test with Rastrigin function
    print("\n" + "=" * 50)
    print("Testing CG-GWO on Rastrigin Function (F6)")
    print("=" * 50)

    bounds_rastrigin = [[-5.12, 5.12]] * dim
    optimizer_rastrigin = CGGWO(
        obj_func=rastrigin_function,
        dim=dim,
        bounds=bounds_rastrigin,
        pop_size=30,
        max_iter=200
    )

    best_pos_r, best_fitness_r, history_r = optimizer_rastrigin.optimize()

    print(f"\nFinal Results:")
    print(f"Best fitness: {best_fitness_r:.10f}")
    print(f"Best position (first 5 dimensions): {best_pos_r[:5]}")

    # Plot convergence comparison
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.semilogy(history, linewidth=2, label='Sphere Function')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('CG-GWO Convergence on Sphere Function')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.semilogy(history_r, linewidth=2, label='Rastrigin Function', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('CG-GWO Convergence on Rastrigin Function')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()
    