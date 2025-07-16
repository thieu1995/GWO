#!/usr/bin/env python
# Created by "Thieu" at 13:51, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Callable, Tuple, List, Optional
import numpy as np


class ChaoticMaps:
    """
    Implementation of 10 chaotic maps
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


class FuzzySystem:
    """Fuzzy System for hierarchical pyramid weights"""

    def __init__(self, pyramid_type='increase'):
        """
        Args:
            pyramid_type: 'increase' or 'decrease'
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
