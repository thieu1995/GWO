#!/usr/bin/env python
# Created by "Thieu" at 13:35, 15/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.swarm_based.GWO import OriginalGWO, RW_GWO, GWO_WOA, IGWO
from utils.helper import ChaoticMaps as CM
from utils.helper import FuzzySystem as FS


class ChaoticGWO(Optimizer):
    """
    The original version of: Chaotic-based Grey Wolf Optimizer (C-GWO)

    Links:
        1. https://doi.org/10.1016/j.jcde.2017.02.005

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kohli, M., & Arora, S. (2018). Chaotic grey wolf optimization algorithm for constrained optimization problems. Journal of computational design and engineering, 5(4), 458-472.
    """

    CHAOTIC_MAPS = {
        "bernoulli": CM.bernoulli_map,
        "logistic": CM.logistic_map,
        "chebyshev": CM.chebyshev_map,
        "circle": CM.circle_map,
        "cubic": CM.cubic_map,
        "icmic": CM.icmic_map,
        "piecewise": CM.piecewise_map,
        "singer": CM.singer_map,
        "sinusoidal": CM.sinusoidal_map,
        "tent": CM.tent_map
    }

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 chaotic_name: str = "chebyshev", initial_chaotic_value: float = 0.7, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            chaotic_name (str): name of chaotic map to use, default = "chebyshev"
            initial_chaotic_value (float): initial value for chaotic map, default = 0.7

        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.chaotic_name = self.validator.check_str("chaotic_name", chaotic_name, ChaoticGWO.CHAOTIC_MAPS.keys())
        self.initial_chaotic_value = self.validator.check_float("initial_chaotic_value", initial_chaotic_value, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "chaotic_name", "initial_chaotic_value"])
        self.sort_flag = False

    def initialize_variables(self) -> None:
        self.chao_value = self.initial_chaotic_value
        self.chao_func = ChaoticGWO.CHAOTIC_MAPS[self.chaotic_name]

    def _update_chao_value(self):
        """Update chaotic value using selected chaotic map"""
        chao_value = self.chao_func(self.chao_value)
        # Ensure chaotic value stays in [0, 1]
        self.chao_value = np.clip(chao_value, 0, 1)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            self._update_chao_value()
            A1 = a * (2 * self.generator.random(self.problem.n_dims) * self.chao_value - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) * self.chao_value - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) * self.chao_value - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims) * self.chao_value
            C2 = 2 * self.generator.random(self.problem.n_dims) * self.chao_value
            C3 = 2 * self.generator.random(self.problem.n_dims) * self.chao_value
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class FuzzyGWO(Optimizer):
    """
    The original version of: Fuzzy Hierarchical Operator - Grey Wolf Optimizer (FHO-GWO or FuzzyGWO or F-GWO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2017.03.048

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodríguez, Luis, Oscar Castillo, José Soria, Patricia Melin, Fevrier Valdez, Claudia I. Gonzalez, Gabriela E. Martinez, and Jesus Soto. "A fuzzy hierarchical operator in the grey wolf optimizer algorithm." Applied Soft Computing 57 (2017): 315-328.
    """

    FUZZY_OPERATORS = ["increase", "decrease"]


    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 fuzzy_name: str = "increase", **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fuzzy_name (str): type of fuzzy operator to use, default = "increase"
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.fuzzy_name = self.validator.check_str("fuzzy_name", fuzzy_name, FuzzyGWO.FUZZY_OPERATORS)
        self.set_parameters(["epoch", "pop_size", "fuzzy_name"])
        self.sort_flag = False

    def initialize_variables(self) -> None:
        self.fuzzy_system = FS(self.fuzzy_name)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)

            # Get fuzzy weights
            FW_alpha, FW_beta, FW_delta = self.fuzzy_system.get_fuzzy_weights(epoch, self.epoch)
            total_weight = FW_alpha + FW_beta + FW_delta
            pos_new = (X1 * FW_alpha + X2 * FW_beta + X3 * FW_delta) / total_weight
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class IncrementalGWO(Optimizer):
    """
    The original version of: Incremental model-based Grey Wolf Optimizer (IncrementalGWO)

    Notes:
        + When calling the solve() function, you need to set the mode to "swarm" to use this algorithm as original version.
        + They update the position of whole population before calculating the fitness of each agent.

    Links:
        1. https://doi.org/10.1007/s00366-019-00837-7

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2021). I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer to solve global optimization problems. Engineering with Computers, 37(1), 509-532.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 explore_factor: float = 1.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            explore_factor (float): factor to control exploration, default = 1.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.explore_factor = self.validator.check_float("explore_factor", explore_factor, [0.0, 5.0])
        self.set_parameters(["epoch", "pop_size", "explore_factor"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 * (1. - (epoch / self.epoch)**self.explore_factor)
        pop_sorted, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                # Alpha wolf updates based on hunting mechanism
                A = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C = 2 * self.generator.random(self.problem.n_dims)
                pos_new = list_best[0].solution - A * np.abs(C * list_best[0].solution - self.pop[idx].solution)
            else:
                # Other wolves update based on all previous wolves (Equation 19)
                # Average position of all previous wolves (n-1 wolves)
                p_temp = np.array([agent.solution for agent in pop_sorted])
                mask = np.arange(p_temp.shape[0]) != idx
                pos_new = p_temp[mask].mean(axis=0)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class ExGWO(Optimizer):
    """
    The original version of: Expanded Grey Wolf Optimizer (Ex-GWO)

    Notes:
        + When calling the solve() function, you need to set the mode to "swarm" to use this algorithm as original version.
        + They update the position of whole population before calculating the fitness of each agent.

    Links:
        1. https://doi.org/10.1007/s00366-019-00837-7

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2021). I-GWO and Ex-GWO: improved algorithms of the Grey Wolf Optimizer to solve global optimization problems. Engineering with Computers, 37(1), 509-532.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 * (1. - epoch / self.epoch)
        pop_sorted, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                pos_new = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            elif idx == 1:
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                pos_new = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            elif idx == 2:
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C3 = 2 * self.generator.random(self.problem.n_dims)
                pos_new = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            else:
                # Other wolves update based on first three + previous wolves (Equation 15)
                # Average of first three wolves + previously updated wolves
                pos_new = np.mean([agent.solution for agent in pop_sorted[:idx]], axis=0)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class DS_GWO(Optimizer):
    """
    The original version of: Diversity enhanced Strategy based Grey Wolf Optimizer (DS-GWO)

    This implementation includes:
        1. Group-stage competition mechanism
        2. Exploration-exploitation balance mechanism

    Links:
        1. https://doi.org/10.1016/j.knosys.2022.109100

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Jiang, Jianhua, Ziying Zhao, Yutong Liu, Weihua Li, and Huan Wang. "DSGWO: An improved grey wolf optimizer with diversity enhanced strategy based on group-stage competition and balance mechanisms." Knowledge-Based Systems 250 (2022): 109100.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 explore_ratio: float = 0.4, n_groups: int = 5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            explore_ratio (float): ratio to control exploration, default = 0.4
            n_groups (int): number of groups for group-stage competition, default = 5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.explore_ratio = self.validator.check_float("explore_ratio", explore_ratio, [0.0, 1.0])
        self.n_groups = self.validator.check_int("n_groups", n_groups, [5, 100])
        self.set_parameters(["epoch", "pop_size", "explore_ratio", "n_groups"])
        self.sort_flag = False

    def initialize_variables(self):
        """
        Initialize any variables needed for the algorithm.
        """
        self.explore_epoch = int(self.epoch * self.explore_ratio)

    def before_main_loop(self):
        """
        Initialize variables before the main loop starts.
        """
        self.group_stage_competition()

    def get_coefficients(self, a: float) -> tuple:
        """
        Generate coefficients A and C for position update equations.

        Args:
            a (float): Coefficient that decreases over epochs

        Returns:
            tuple: Coefficients A, C
        """
        A = a * (2 * self.generator.random(self.problem.n_dims) - 1)
        C = 2 * self.generator.random(self.problem.n_dims)
        return A, C

    def group_stage_competition(self):
        """
        Group-stage competition mechanism:
        1. Divide population into 6 subgroups
        2. Select best wolf from each subgroup as delta candidates
        3. Set best overall as alpha
        4. Set delta candidate farthest from alpha as beta
        """
        # Divide population into n_groups
        group_size = self.pop_size // self.n_groups
        self.delta_candidates = []

        for idx in range(self.n_groups):
            start_idx = idx * group_size
            if idx == self.n_groups - 1:  # Last group takes remaining wolves
                end_idx = self.pop_size
            else:
                end_idx = (idx + 1) * group_size

            # Get group members
            group_population = self.pop[start_idx:end_idx]
            # Find best wolf in group
            group_sorted = self.get_sorted_population(group_population, minmax=self.problem.minmax)
            self.delta_candidates.append(group_sorted[0].copy())

        # Set alpha wolf (best among all delta candidates)
        _, list_best, _ = self.get_special_agents(self.delta_candidates, n_best=1, minmax=self.problem.minmax)
        self.alpha = list_best[0].copy()

        # Set beta wolf (delta candidate farthest from alpha)
        delta_pos = np.array([agent.solution for agent in self.delta_candidates])
        distances = np.linalg.norm(delta_pos - self.alpha.solution, axis=1)
        beta_idx = np.argmax(distances)
        self.beta = self.delta_candidates[beta_idx].copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        pop_new = []
        for idx in range(0, self.pop_size):
            if epoch < self.explore_epoch:
                # Exploration phase: use alpha and beta wolves
                # Exploration phase: Update position using two randomly selected delta candidates
                selected_idxs = self.generator.choice(len(self.delta_candidates), 2, replace=False)
                delta1 = self.delta_candidates[selected_idxs[0]]
                delta2 = self.delta_candidates[selected_idxs[1]]
                A1, C1 = self.get_coefficients(a)
                A2, C2 = self.get_coefficients(a)
                X1 = delta1.solution - A1 * np.abs(C1 * delta1.solution - self.pop[idx].solution)
                X2 = delta2.solution - A2 * np.abs(C2 * delta2.solution - self.pop[idx].solution)
                pos_new = (X1 + X2) / 2.0
            else:
                # Exploitation phase: use classic GWO equations with alpha and beta wolves
                delta_sorted = self.get_sorted_population(self.delta_candidates, minmax=self.problem.minmax)
                delta = delta_sorted[2] if len(delta_sorted) >=3 else delta_sorted[-1]
                A1, C1 = self.get_coefficients(a)
                A2, C2 = self.get_coefficients(a)
                A3, C3 = self.get_coefficients(a)
                X1 = self.alpha.solution - A1 * np.abs(C1 * self.alpha.solution - self.pop[idx].solution)
                X2 = self.beta.solution - A2 * np.abs(C2 * self.beta.solution - self.pop[idx].solution)
                X3 = delta.solution - A3 * np.abs(C3 * delta.solution - self.pop[idx].solution)
                pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Update leading wolves using group-stage competition
        self.group_stage_competition()


class IOBL_GWO(Optimizer):
    """
    The original version of: Improved Grey Wolf Optimizer (IGWO) => IOBL-GWO

    Notes:
        + In the paper, they called it "Improved Grey Wolf Optimizer (IGWO)", but there are many improved versions of GWO.
        + So based on their proposed equations, we called it as "Improved Opposite-based Learning Grey Wolf Optimizer (IOBL-GWO)".
        + This algorithm is heavily (4x - 6X slower than original) because of multiple times of calculating the fitness of agent in each population.

    Links:
        1. https://doi.org/10.1007/s12652-020-02153-1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bansal, J. C., & Singh, S. (2021). A better exploration strategy in Grey Wolf Optimizer. Journal of Ambient Intelligence and Humanized Computing, 12(1), 1099-1118.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        for idx in range(0, self.pop_size):
            # Try explorative equation first
            r1, r2, r3, r4, r5 = self.generator.random(5)
            if r5 >= 0.5:   # Exploration around random wolf
                # Select random wolf from population
                jdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                x_rand = self.pop[jdx].solution
                pos_new = x_rand - r1 * np.abs(x_rand - 2 * r2 * self.pop[idx].solution)
            else:           # Exploration around alpha wolf
                # Calculate average position of all wolves
                x_avg = np.mean([agent.solution for agent in self.pop], axis=0)
                pos_new = (list_best[0].solution - x_avg) - r3 * (self.problem.lb + r4 * (self.problem.ub - self.problem.lb))
            # Apply boundary constraints
            pos_new = self.correct_solution(pos_new)
            tar_new = self.get_target(pos_new)
            if self.compare_target(tar_new, self.pop[idx].target, self.problem.minmax):
                # If new position is better, update the agent
                agent = self.generate_empty_agent(pos_new)
                agent.target = tar_new
                self.pop[idx] = agent
            else:
                # If not better, use original GWO update
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                C3 = 2 * self.generator.random(self.problem.n_dims)

                X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
                X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
                X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
                pos_new = (X1 + X2 + X3) / 3.0
                pos_new = self.correct_solution(pos_new)
                tar_new = self.get_target(pos_new)
                # Create new agent with updated position
                if self.compare_target(tar_new, self.pop[idx].target, self.problem.minmax):
                    agent = self.generate_empty_agent(pos_new)
                    agent.target = tar_new
                    self.pop[idx] = agent

        # Apply Opposition-Based Learning (OBL) for leading wolves
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_sorted, indices = self.get_sorted_population(self.pop, minmax=self.problem.minmax, return_index=True)
        obl_alpha = self.generate_agent(solution=self.problem.lb + self.problem.ub - pop_sorted[0].solution)
        obl_beta = self.generate_agent(solution=self.problem.lb + self.problem.ub - pop_sorted[1].solution)
        obl_delta = self.generate_agent(solution=self.problem.lb + self.problem.ub - pop_sorted[2].solution)
        obl_pop = [obl_alpha, obl_beta, obl_delta]

        # Replace worst 3 wolves with opposite solutions if they are better
        for idx in range(0, 3):
            if self.compare_target(obl_pop[idx].target, self.pop[indices[-3+idx]].target, self.problem.minmax):
                self.pop[idx] = obl_pop[idx]


class OGWO(Optimizer):
    """
    The original version of: Opposition-based learning Grey Wolf Optimizer (OGWO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.107139

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yu, X., Xu, W., & Li, C. (2021). Opposition-based learning grey wolf optimizer for global optimization. Knowledge-Based Systems, 226, 107139.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 miu_factor: float = 2.0, jumping_rate: float = 0.05, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_factor (float): nonlinear coefficient for equation (11), default = 2.0
            jumping_rate (float):  jumping rate for OBL, default = 0.05
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_factor = self.validator.check_float("miu_factor", miu_factor, [0.0, 10.0])
        self.jumping_rate = self.validator.check_float("jumping_rate", jumping_rate, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "miu_factor", "jumping_rate"])
        self.sort_flag = False

    def initialization(self) -> None:
        """Initialize population with opposition-based learning"""
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)

        # Generate opposition population using equation (12)
        pop_opposite = []
        for agent in self.pop:
            pos_opposite = self.problem.lb + self.problem.ub - agent.solution
            agent_opposite = self.generate_empty_agent(pos_opposite)
            agent_opposite.target = self.get_target(pos_opposite)
            pop_opposite.append(agent_opposite)
        # Combine original and opposite populations
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_opposite, self.pop_size, minmax=self.problem.minmax)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2. * (1 - (epoch / self.epoch)**self.miu_factor)
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Apply opposition-based learning
        if self.generator.random() < self.jumping_rate:
            # Generate opposition population using equation (12)
            pop_opposite = []
            for agent in self.pop:
                pos_opposite = self.problem.lb + self.problem.ub - agent.solution
                agent_opposite = self.generate_empty_agent(pos_opposite)
                agent_opposite.target = self.get_target(pos_opposite)
                pop_opposite.append(agent_opposite)
            # Combine original and opposite populations
            self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_opposite, self.pop_size, minmax=self.problem.minmax)


class ER_GWO(Optimizer):
    """
    The original version of: Efficient and Robust Grey Wolf Optimizer (ER-GWO)

    Notes:
        + Slow convergence speed due to the (miu_factor)^(iteration) ==> Big number
        + Three more parameters than original GWO, increase the complexity of the algorithm.

    Links:
        1. https://doi.org/10.1007/s00500-019-03939-y

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Long, W., Cai, S., Jiao, J. et al. An efficient and robust grey wolf optimizer algorithm for large-scale numerical optimization. Soft Comput 24, 997–1026 (2020).
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 a_initial: float = 2.0, a_final: float = 0.0, miu_factor: float = 1.0001, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_initial (float): initial value of coefficient a, default = 2.0
            a_final (float): final value of coefficient a, default = 0.0
            miu_factor (float): nonlinear coefficient for equation (8), default = 1.0001
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.a_initial = self.validator.check_float("a_initial", a_initial, [0.0, 10.0])
        self.a_final = self.validator.check_float("a_final", a_final, [0.0, self.a_initial])
        self.miu_factor = self.validator.check_float("miu_factor", miu_factor, [1.0001, 1.01])     # Required in paper
        self.set_parameters(["epoch", "pop_size", "a_initial", "a_final", "miu_factor"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = self.a_initial - (self.a_initial - self.a_final) * self.miu_factor ** epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            dist1 = np.linalg.norm(X1)
            dist2 = np.linalg.norm(X2)
            dist3 = np.linalg.norm(X3)
            total = dist1 + dist2 + dist3
            if total == 0:
                # Avoid division by zero
                pos_new = (X1 + X2 + X3) / 3.0
            else:
                # Normalize distances to avoid division by zero
                pos_new = (X1 * dist1 + X2 * dist2 + X3 * dist3) / total
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class CG_GWO(Optimizer):
    """
    The original version of: Cauchy‑Gaussian mutation and improved search strategy GWO (CG‑GWO)

    Notes:
        + This algorithm can't be parallelized because of the 'single' update mode.
        + Meaning that the updating of the pack is based on order and sequence of the wolves.

    Links:
        1. https://doi.org/10.1038/s41598-022-23713-9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Li, K., Li, S., Huang, Z. et al. Grey Wolf Optimization algorithm based on Cauchy-Gaussian mutation and improved search strategy. Sci Rep 12, 18961 (2022).
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def cauchy_gaussian_mutation(self, best, leader, epoch):
        """
        Apply Cauchy-Gaussian mutation to leader wolves
        """
        # Calculate dynamic parameters (equations 11 and 12)
        eps2 = (epoch / self.epoch) ** 2
        eps1 = 1 - eps2

        # Calculate sigma (equation 9)
        if abs(best.target.fitness) > 1e-10:
            sigma = np.exp((leader.target.fitness - best.target.fitness) / abs(best.target.fitness))
        else:
            sigma = 1.0
        # Generate Cauchy and Gaussian random variables
        c_rand = self.generator.standard_cauchy(size=self.problem.n_dims) * sigma**2 + 0
        g_rand = self.generator.normal(loc=0, scale=sigma**2, size=self.problem.n_dims)

        # Apply mutation (equation 8)
        mutated_pos = leader.solution * (1 + eps1 * c_rand + eps2 * g_rand)
        return mutated_pos

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)

        # Apply Cauchy-Gaussian mutation to leaders
        alpha_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[0], epoch)
        alpha_pos = self.correct_solution(alpha_pos)
        alpha = self.generate_agent(solution=alpha_pos)

        beta_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[1], epoch)
        beta_pos = self.correct_solution(beta_pos)
        beta = self.generate_agent(solution=beta_pos)

        delta_pos = self.cauchy_gaussian_mutation(list_best[0], list_best[2], epoch)
        delta_pos = self.correct_solution(delta_pos)
        delta = self.generate_agent(solution=delta_pos)

        leaders = [alpha, beta, delta]
        # Greedy selection mechanism
        list_best = self.greedy_selection_population(list_best, leaders, self.problem.minmax)

        pop_new = []
        for idx in range(0, self.pop_size):
            ## Apply improved search strategy

            # Apply improved search strategy (equation 13)
            r1, r2, r3, r4, r5 = self.generator.random(5)
            if r5 >= 0.5:  # Exploration around random wolf
                # Select random wolf from population
                jdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                x_rand = self.pop[jdx].solution
                pos_new = x_rand - r1 * np.abs(x_rand - 2 * r2 * self.pop[idx].solution)
            else:  # Exploration around alpha wolf
                # Calculate average position of all wolves
                x_avg = np.mean([agent.solution for agent in self.pop], axis=0)
                pos_new = (list_best[0].solution - x_avg) - r3 * (self.problem.lb + r4 * (self.problem.ub - self.problem.lb))
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            if self.compare_target(self.pop[idx].target, agent.target, self.problem.minmax):
                # If new position is not better, use original GWO update
                A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
                C1 = 2 * self.generator.random(self.problem.n_dims)
                C2 = 2 * self.generator.random(self.problem.n_dims)
                C3 = 2 * self.generator.random(self.problem.n_dims)
                X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
                X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
                X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
                pos_new = (X1 + X2 + X3) / 3.0
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)

            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                # If new position is better, update the agent
                self.pop[idx] = agent
