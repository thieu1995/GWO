#!/usr/bin/env python
# Created by "Thieu" at 14:43, 16/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import concurrent.futures
from utils.io import *
from mealpy.swarm_based.GWO import OriginalGWO, RW_GWO, GWO_WOA, IGWO
from GWO import ChaoticGWO, FuzzyGWO, IncrementalGWO, ExGWO, DS_GWO, IOBL_GWO, OGWO, ER_GWO, CG_GWO


def run_experiment(list_funcs, get_func, dict_optimizer_classes, n_trials,
                   epoch, pop_size, save_file=True, path_save="cec", verbose=True, n_workers=10):
    results_gbest_fit = []  # Working with global best fitness - statistic test

    # Helper function to execute a single trial for a model
    def run_trial(idx_trial, func, model_name, model_class):
        term_dict = {
            "max_fe": 60000  # number of function evaluation
        }
        prob = get_func(func)
        model = model_class(epoch=epoch, pop_size=pop_size, name=model_name)
        g_best = model.solve(prob, termination=term_dict)
        print(f"{model}, Trial: {idx_trial}, Fitness: {g_best.target.fitness}")

        # Collect best results (Average, Best, SD)
        result_fit = {
            'Function': func,
            'Algorithm': model_name,
            'Trial': idx_trial,
            'Global_Best_Fitness': g_best.target.fitness
        }

        # Collect convergence curve
        curve_trial = model.history.list_global_best_fit
        gbest_box = [model_name, g_best.target.fitness]
        return result_fit, curve_trial, gbest_box

    # Helper function to execute a single model for a function
    def run_model(idx_model, func, model_name, model_class):
        list_curve_trial = []
        results_gbest_box = []
        futures = []

        # Parallelize the trials within each model
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as model_executor:
            for idx_trial in range(n_trials):
                futures.append(model_executor.submit(run_trial, idx_trial, func, model_name, model_class))

            for future in concurrent.futures.as_completed(futures):
                result_fit, curve_trial, gbest_box = future.result()
                results_gbest_fit.append(result_fit)
                list_curve_trial.append(curve_trial)
                results_gbest_box.append(gbest_box)

        return model_name, list_curve_trial, results_gbest_box

    # Main loop for each function
    for idx_func, func in enumerate(list_funcs):
        results_curve = {}
        all_gbest_box = []
        model_futures = []

        # Parallelize the models within each function
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as func_executor:
            for idx_model, (model_name, model_class) in enumerate(dict_optimizer_classes.items()):
                model_futures.append(func_executor.submit(run_model, idx_model, func, model_name, model_class))

            for model_future in concurrent.futures.as_completed(model_futures):
                model_name, list_curve_trial, results_gbest_box = model_future.result()
                results_curve[model_name] = list_curve_trial
                all_gbest_box.extend(results_gbest_box)

            ## Plot each convergence of all models for each trial
        for idx_trial, trial in enumerate(range(n_trials)):
            draw_convergence_curve_for_each_trial(dict_optimizer_classes.keys(), results_curve, func, idx_trial,
                                                  fig_size=(7, 6), save_file=save_file, path_save=path_save, verbose=verbose)

            # Plot average convergence of these models together for each function
        draw_average_convergence_curve(dict_optimizer_classes.keys(), results_curve, func, fig_size=(7, 6),
                                       save_file=save_file, path_save=path_save, verbose=verbose)

        # Plot the stability chart using seaborn boxplot for each function with different colors for each model
        df_gbest_box = pd.DataFrame(all_gbest_box, columns=['Algorithm', 'Global_Best_Fitness'])
        draw_stability_chart(df_gbest_box, func, fig_size=(7, 5), save_file=save_file, path_save=path_save, verbose=verbose)

    # Save results to file
    results_df = pd.DataFrame(results_gbest_fit)
    save_global_best_value(results_df, save_file=save_file, path_save=path_save, verbose=verbose)
    save_hypothesis_test(results_df, SELECTED_MODEL, dict_optimizer_classes.keys(), save_file=save_file, path_save=path_save, verbose=verbose)


def get_func_2017(fname="f1", ndim=30):
    import opfunu
    from mealpy import FloatVar

    FX = opfunu.get_functions_by_classname(f"{fname}2017")[0]
    fx = FX(ndim=ndim)

    problem = {
        "obj_func": fx.evaluate,
        "bounds": FloatVar(lb=fx.lb, ub=fx.ub),
        "minmax": "min",
        "name": fname,
        "log_to": None,
    }
    return problem

if __name__ == "__main__":
    # List of functions for CEC 2017 competition
    # list_funcs_2017 = ["F1", "F2"]
    list_funcs_2017 = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15",
                      "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28", "F29"]

    dict_optimizer_classes = {
        "GWO": OriginalGWO,
        "RW-GWO": RW_GWO,
        "GWO-WOA": GWO_WOA,
        "IGWO": IGWO,
        "ChaoticGWO": ChaoticGWO,
        "FuzzyGWO": FuzzyGWO,
        "IncrementalGWO": IncrementalGWO,
        "ExGWO": ExGWO,
        "DS-GWO": DS_GWO,
        "IOBL-GWO": IOBL_GWO,
        "OGWO": OGWO,
        "ER": ER_GWO,
        "CG-GWO": CG_GWO,
    }

    ## Set up parametes
    SAVE_TO_FILE = True
    SELECTED_MODEL = "GWO"    # Draw analysis of selected algorithm (proposed algorithm) only. And calculate the hypothesis against this algorithm
    PRINT_RESULT = False      # To save resource and for faster computation in colab, please set it to False, you don't need to see it print too much information
    PATH_SAVE_2017 = "cec2017"

    EPOCH = 5000
    POP_SIZE = 30
    N_TRIALS = 30

    run_experiment(list_funcs_2017, get_func_2017, dict_optimizer_classes, N_TRIALS, EPOCH, POP_SIZE,
                   save_file=SAVE_TO_FILE, path_save=PATH_SAVE_2017, verbose=PRINT_RESULT,
                   n_workers=6)
