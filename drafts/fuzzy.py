#!/usr/bin/env python
# Created by "Thieu" at 14:36, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class FuzzySystem:
    """Fuzzy System cho hierarchical pyramid weights"""

    def __init__(self, pyramid_type='increase'):
        """
        Args:
            pyramid_type: 'increase' hoặc 'decrease'
        """
        self.pyramid_type = pyramid_type

    def triangular_membership(self, x, a, b, c):
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    def fuzzify_iterations(self, iteration_percent):
        """Fuzzify input iterations (0-1)"""
        low = self.triangular_membership(iteration_percent, 0, 0, 0.5)
        medium = self.triangular_membership(iteration_percent, 0, 0.5, 1)
        high = self.triangular_membership(iteration_percent, 0.5, 1, 1)
        return {'low': low, 'medium': medium, 'high': high}

    def defuzzify_centroid(self, membership_values):
        """Defuzzification using centroid method"""
        # Triangular membership functions for output (0-100)
        low_center = 25
        medium_center = 50
        high_center = 75

        numerator = (membership_values['low'] * low_center +
                     membership_values['medium'] * medium_center +
                     membership_values['high'] * high_center)
        denominator = sum(membership_values.values())

        if denominator == 0:
            return 50  # Default value

        return numerator / denominator

    def get_fuzzy_weights(self, current_iteration, max_iterations):
        """Get fuzzy weights for alpha, beta, delta"""
        iteration_percent = current_iteration / max_iterations
        input_membership = self.fuzzify_iterations(iteration_percent)

        if self.pyramid_type == 'increase':
            # Rules for increase pyramid
            alpha_membership = {
                'low': input_membership['low'],
                'medium': max(input_membership['medium'], input_membership['low']),
                'high': input_membership['high']
            }
            beta_membership = {
                'low': input_membership['high'],
                'medium': max(input_membership['low'], input_membership['medium'], input_membership['high']),
                'high': 0
            }
            delta_membership = {
                'low': max(input_membership['medium'], input_membership['high']),
                'medium': input_membership['low'],
                'high': 0
            }
        else:  # decrease
            # Rules for decrease pyramid
            alpha_membership = {
                'low': input_membership['high'],
                'medium': max(input_membership['medium'], input_membership['high']),
                'high': input_membership['low']
            }
            beta_membership = {
                'low': input_membership['high'],
                'medium': max(input_membership['low'], input_membership['medium'], input_membership['high']),
                'high': 0
            }
            delta_membership = {
                'low': max(input_membership['low'], input_membership['medium']),
                'medium': input_membership['high'],
                'high': 0
            }

        alpha_weight = self.defuzzify_centroid(alpha_membership)
        beta_weight = self.defuzzify_centroid(beta_membership)
        delta_weight = self.defuzzify_centroid(delta_membership)

        return alpha_weight, beta_weight, delta_weight


class FuzzyGWO:
    """Fuzzy Grey Wolf Optimizer"""

    def __init__(self, objective_function: Callable,
                 dimension: int,
                 search_space: Tuple[float, float],
                 population_size: int = 30,
                 max_iterations: int = 500,
                 hierarchical_operator: str = 'fuzzy_increase'):
        """
        Args:
            objective_function: Hàm mục tiêu cần tối ưu
            dimension: Số chiều của bài toán
            search_space: Không gian tìm kiếm (min, max)
            population_size: Kích thước quần thể
            max_iterations: Số lần lặp tối đa
            hierarchical_operator: Loại operator ('centroid', 'weighted', 'fitness', 'fuzzy_increase', 'fuzzy_decrease')
        """
        self.objective_function = objective_function
        self.dimension = dimension
        self.search_space = search_space
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.hierarchical_operator = hierarchical_operator

        # Khởi tạo fuzzy system
        if 'fuzzy' in hierarchical_operator:
            pyramid_type = 'increase' if 'increase' in hierarchical_operator else 'decrease'
            self.fuzzy_system = FuzzySystem(pyramid_type)

        # Lịch sử tối ưu
        self.best_fitness_history = []
        self.alpha_history = []

    def initialize_population(self):
        """Khởi tạo quần thể"""
        return np.random.uniform(self.search_space[0], self.search_space[1],
                                 (self.population_size, self.dimension))

    def evaluate_fitness(self, population):
        """Đánh giá fitness cho toàn bộ quần thể"""
        fitness = np.array([self.objective_function(individual) for individual in population])
        return fitness

    def get_leaders(self, population, fitness):
        """Lấy alpha, beta, delta wolves"""
        sorted_indices = np.argsort(fitness)
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1]
        delta_idx = sorted_indices[2]

        return (population[alpha_idx], population[beta_idx], population[delta_idx],
                fitness[alpha_idx], fitness[beta_idx], fitness[delta_idx])

    def update_position_centroid(self, alpha, beta, delta, current_wolf):
        """Centroid operator (Equation 7)"""
        # Tính toán X1, X2, X3
        A1 = 2 * self.a * np.random.random(self.dimension) - self.a
        A2 = 2 * self.a * np.random.random(self.dimension) - self.a
        A3 = 2 * self.a * np.random.random(self.dimension) - self.a

        C1 = 2 * np.random.random(self.dimension)
        C2 = 2 * np.random.random(self.dimension)
        C3 = 2 * np.random.random(self.dimension)

        D_alpha = np.abs(C1 * alpha - current_wolf)
        D_beta = np.abs(C2 * beta - current_wolf)
        D_delta = np.abs(C3 * delta - current_wolf)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3

    def update_position_weighted(self, alpha, beta, delta, current_wolf):
        """Weighted average operator (Equation 8)"""
        # Tính toán X1, X2, X3
        A1 = 2 * self.a * np.random.random(self.dimension) - self.a
        A2 = 2 * self.a * np.random.random(self.dimension) - self.a
        A3 = 2 * self.a * np.random.random(self.dimension) - self.a

        C1 = 2 * np.random.random(self.dimension)
        C2 = 2 * np.random.random(self.dimension)
        C3 = 2 * np.random.random(self.dimension)

        D_alpha = np.abs(C1 * alpha - current_wolf)
        D_beta = np.abs(C2 * beta - current_wolf)
        D_delta = np.abs(C3 * delta - current_wolf)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        return (5 * X1 + 3 * X2 + 2 * X3) / 10

    def update_position_fitness(self, alpha, beta, delta, current_wolf,
                                alpha_fitness, beta_fitness, delta_fitness):
        """Fitness-based weighted operator (Equations 9-10)"""
        # Tính toán X1, X2, X3
        A1 = 2 * self.a * np.random.random(self.dimension) - self.a
        A2 = 2 * self.a * np.random.random(self.dimension) - self.a
        A3 = 2 * self.a * np.random.random(self.dimension) - self.a

        C1 = 2 * np.random.random(self.dimension)
        C2 = 2 * np.random.random(self.dimension)
        C3 = 2 * np.random.random(self.dimension)

        D_alpha = np.abs(C1 * alpha - current_wolf)
        D_beta = np.abs(C2 * beta - current_wolf)
        D_delta = np.abs(C3 * delta - current_wolf)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        # Tính weights dựa trên fitness
        total_fitness = alpha_fitness + beta_fitness + delta_fitness
        if total_fitness == 0:
            # Nếu tất cả fitness = 0, sử dụng equal weights
            W_alpha = W_beta = W_delta = 1 / 3
        else:
            W_alpha = total_fitness / alpha_fitness if alpha_fitness != 0 else 1e6
            W_beta = total_fitness / beta_fitness if beta_fitness != 0 else 1e6
            W_delta = total_fitness / delta_fitness if delta_fitness != 0 else 1e6

        total_weight = W_alpha + W_beta + W_delta

        return (X1 * W_alpha + X2 * W_beta + X3 * W_delta) / total_weight

    def update_position_fuzzy(self, alpha, beta, delta, current_wolf, iteration):
        """Fuzzy weights operator (Equation 11)"""
        # Tính toán X1, X2, X3
        A1 = 2 * self.a * np.random.random(self.dimension) - self.a
        A2 = 2 * self.a * np.random.random(self.dimension) - self.a
        A3 = 2 * self.a * np.random.random(self.dimension) - self.a

        C1 = 2 * np.random.random(self.dimension)
        C2 = 2 * np.random.random(self.dimension)
        C3 = 2 * np.random.random(self.dimension)

        D_alpha = np.abs(C1 * alpha - current_wolf)
        D_beta = np.abs(C2 * beta - current_wolf)
        D_delta = np.abs(C3 * delta - current_wolf)

        X1 = alpha - A1 * D_alpha
        X2 = beta - A2 * D_beta
        X3 = delta - A3 * D_delta

        # Lấy fuzzy weights
        FW_alpha, FW_beta, FW_delta = self.fuzzy_system.get_fuzzy_weights(iteration, self.max_iterations)

        total_weight = FW_alpha + FW_beta + FW_delta

        return (X1 * FW_alpha + X2 * FW_beta + X3 * FW_delta) / total_weight

    def bound_position(self, position):
        """Giới hạn vị trí trong không gian tìm kiếm"""
        return np.clip(position, self.search_space[0], self.search_space[1])

    def optimize(self):
        """Thuật toán tối ưu chính"""
        # Khởi tạo quần thể
        population = self.initialize_population()

        # Đánh giá fitness ban đầu
        fitness = self.evaluate_fitness(population)

        # Lấy leaders ban đầu
        alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness = self.get_leaders(population, fitness)

        # Lưu lịch sử
        self.best_fitness_history.append(alpha_fitness)
        self.alpha_history.append(alpha.copy())

        # Vòng lặp chính
        for iteration in range(self.max_iterations):
            # Cập nhật tham số a
            self.a = 2 - 2 * iteration / self.max_iterations

            # Cập nhật vị trí cho từng wolf
            for i in range(self.population_size):
                if self.hierarchical_operator == 'centroid':
                    new_position = self.update_position_centroid(alpha, beta, delta, population[i])
                elif self.hierarchical_operator == 'weighted':
                    new_position = self.update_position_weighted(alpha, beta, delta, population[i])
                elif self.hierarchical_operator == 'fitness':
                    new_position = self.update_position_fitness(alpha, beta, delta, population[i],
                                                                alpha_fitness, beta_fitness, delta_fitness)
                elif 'fuzzy' in self.hierarchical_operator:
                    new_position = self.update_position_fuzzy(alpha, beta, delta, population[i], iteration)

                # Giới hạn vị trí
                population[i] = self.bound_position(new_position)

            # Đánh giá fitness mới
            fitness = self.evaluate_fitness(population)

            # Cập nhật leaders
            alpha, beta, delta, alpha_fitness, beta_fitness, delta_fitness = self.get_leaders(population, fitness)

            # Lưu lịch sử
            self.best_fitness_history.append(alpha_fitness)
            self.alpha_history.append(alpha.copy())

            # In tiến trình
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Best fitness = {alpha_fitness:.6f}")

        return alpha, alpha_fitness, self.best_fitness_history


# Hàm test functions
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


# Demo và so sánh các operators
if __name__ == "__main__":
    # Thiết lập bài toán
    dimension = 30
    search_space = (-10, 10)
    population_size = 30
    max_iterations = 300

    # Các operators để test
    operators = ['centroid', 'weighted', 'fitness', 'fuzzy_increase', 'fuzzy_decrease']

    # Test function
    test_function = sphere_function

    print("=== FuzzyGWO Algorithm Test ===")
    print(f"Test function: Sphere function")
    print(f"Dimension: {dimension}")
    print(f"Search space: {search_space}")
    print(f"Population size: {population_size}")
    print(f"Max iterations: {max_iterations}")
    print()

    results = {}

    for operator in operators:
        print(f"Testing {operator} operator...")

        # Khởi tạo algorithm
        optimizer = FuzzyGWO(
            objective_function=test_function,
            dimension=dimension,
            search_space=search_space,
            population_size=population_size,
            max_iterations=max_iterations,
            hierarchical_operator=operator
        )

        # Chạy optimization
        best_position, best_fitness, history = optimizer.optimize()

        results[operator] = {
            'best_position': best_position,
            'best_fitness': best_fitness,
            'history': history
        }

        print(f"Final best fitness: {best_fitness:.6f}")
        print(f"Best position: {best_position[:5]}...")  # Show first 5 dimensions
        print()

    # Vẽ đồ thị so sánh
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (operator, result) in enumerate(results.items()):
        plt.plot(result['history'], label=f'{operator} (final: {result["best_fitness"]:.2e})',
                 color=colors[i], linewidth=2)

    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.title('FuzzyGWO Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # In kết quả cuối cùng
    print("=== Final Results Summary ===")
    for operator, result in results.items():
        print(f"{operator}: {result['best_fitness']:.6f}")
