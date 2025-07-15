#!/usr/bin/env python
# Created by "Thieu" at 14:19, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import Problem, FloatVar
from GWO import ChaoticGWO, FuzzyGWO, IncrementalGWO, ExGWO, DS_GWO, IOBL_GWO, OGWO


def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x ** 2)


prob = {
    "obj_func": sphere_function,
    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    "minmax": "min",
}

# model = ChaoticGWO(epoch=100, pop_size=20, chaotic_name='chebyshev', initial_chaotic_value=0.7)
# model.solve(problem=prob)

# model = FuzzyGWO(epoch=100, pop_size=20, fuzzy_name="increase")
# model.solve(problem=prob)

# model = IncrementalGWO(epoch=100, pop_size=20, explore_factor=1.5)
# model.solve(problem=prob, mode="swarm")

# model = ExGWO(epoch=100, pop_size=20)
# model.solve(problem=prob, mode="swarm")

# model = DS_GWO(epoch=100, pop_size=20, explore_ratio=0.4, n_groups=6)
# model.solve(problem=prob, mode="swarm")

# model = IOBL_GWO(epoch=100, pop_size=20)
# model.solve(problem=prob)

model = OGWO(epoch=100, pop_size=20, miu_factor=2.0, jumping_rate=0.6)
model.solve(problem=prob)
