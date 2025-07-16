#!/usr/bin/env python
# Created by "Thieu" at 14:41, 16/07/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, friedmanchisquare
import matplotlib.pyplot as plt


def save_global_best_value(res, save_file=False, path_save="cec", verbose=True):
    # Group by 'Function' and 'Algorithm'
    grouped = res.groupby(['Function', 'Algorithm'])['Global_Best_Fitness']

    # Calculate the average global best value
    average_df = grouped.mean().reset_index()
    average_df.columns = ['Function', 'Algorithm', 'Average_Global_Best_Value']

    # Calculate the min global best value
    min_df = grouped.min().reset_index()
    min_df.columns = ['Function', 'Algorithm', 'Min_Global_Best_Value']

    # Calculate the standard deviation of the global best values
    std_df = grouped.std().reset_index()
    std_df.columns = ['Function', 'Algorithm', 'Std_Global_Best_Value']

    # Merge average and std DataFrames to calculate ranking
    combined_df = pd.merge(average_df, std_df, on=['Function', 'Algorithm'])
    # Sort by 'Average_Global_Best_Value' first and then by 'Std_Global_Best_Value'
    combined_df.sort_values(by=['Average_Global_Best_Value', 'Std_Global_Best_Value'], ascending=[True, True], inplace=True)
    # Assign ranks
    combined_df['Rank'] = combined_df.groupby('Function').cumcount() + 1

    # Save the results to CSV files
    if save_file:
        Path(f'{path_save}').mkdir(parents=True, exist_ok=True)
        average_df.to_csv(f'{path_save}/average_global_best_value.csv', index=False)
        min_df.to_csv(f"{path_save}/min_global_best_value.csv", index=False)
        std_df.to_csv(f'{path_save}/std_global_best_value.csv', index=False)
        combined_df.to_csv(f'{path_save}/ranked_algorithms.csv', index=False)

    # Display the DataFrames
    if verbose:
        print("Average Global Best Value DataFrame:")
        print(average_df)
        print("Min Global Best Value Dataframe:")
        print(min_df)
        print("\nStandard Deviation Global Best Value DataFrame:")
        print(std_df)
        print("\nRanked Algorithms DataFrame:")
        print(combined_df)


def save_hypothesis_test(df, selected_model, all_models, save_file=False, path_save="cec", verbose=True):

    # Algorithms to compare with selected_model
    other_algorithms = list(set(list(all_models)) - {selected_model})

    # List to store p-values
    results = []

    # Compare BOA with each algorithm for each benchmark function
    for benchmark in df['Function'].unique():
        for algo in other_algorithms:
            # Extract performance values for BOA and the other algorithm
            selected_values = df[(df['Function'] == benchmark) & (df['Algorithm'] == selected_model)]['Global_Best_Fitness']
            compared_values = df[(df['Function'] == benchmark) & (df['Algorithm'] == algo)]['Global_Best_Fitness']

            # Perform the Wilcoxon Rank-Sum Test
            stat, p_val = mannwhitneyu(selected_values, compared_values, alternative='two-sided')

            if p_val < 0.05:
                # better = selected_model if selected_values.mean() < compared_values.mean() else algo
                better = "+" if selected_values.mean() <= compared_values.mean() else "-"
            else:
                better = "ns"

            # Append the results
            results.append({
                'Function': benchmark,
                'Algorithm': algo,
                'P-Value': str(round(p_val, 2)) + f" ({better})"
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Pivot the DataFrame to get the p-values in the desired table format
    p_value_table = results_df.pivot(index='Function', columns='Algorithm', values='P-Value')

    if save_file:
        Path(f'{path_save}').mkdir(parents=True, exist_ok=True)
        p_value_table.to_csv(f'{path_save}/p-value-table.csv', index=False)

    # Display the p-value table
    if verbose:
        print(p_value_table)


def draw_convergence_curve_for_each_trial(models, fit_results, func_name, trial_idx, fig_size=(10, 6),
                                          save_file=False, path_save="cec", verbose=True):
    plt.figure(figsize=fig_size)
    for idx_model, model in enumerate(models):
        plt.plot(fit_results[model][trial_idx], label=model)
    plt.title(f'Convergence curve of compared algorithms for {func_name} - Trial {trial_idx + 1}')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.legend()
    if save_file:
        Path(f'{path_save}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path_save}/convergence-benchmark_{func_name}-trial_{trial_idx}.png", bbox_inches='tight')
    if verbose:
        plt.show()
    return None


def draw_average_convergence_curve(models, fit_results, func_name, fig_size=(10, 6),
                                   save_file=False, path_save="cec", verbose=True):
    plt.figure(figsize=fig_size)
    for idx_model, model in enumerate(models):
        avg_fitness = np.mean(fit_results[model], axis=0)
        plt.plot(avg_fitness, label=model)
    plt.title(f'Average convergence curve of compared algorithms for {func_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Average fitness Value')
    plt.legend()
    if save_file:
        Path(f'{path_save}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path_save}/average-convergence-benchmark_{func_name}.png", bbox_inches='tight')
    if verbose:
        plt.show()
    return None


def draw_stability_chart(fit_results, func_name, fig_size=(10, 6),
                         save_file=False, path_save="cec", verbose=True):
    plt.figure(figsize=fig_size)
    sns.boxplot(x='Algorithm', y='Global_Best_Fitness', data=fit_results, palette='Set2', hue="Algorithm")
    plt.title(f'Stability Chart for {func_name}')
    plt.xlabel('Algorithms')
    plt.ylabel('Global best fitness')
    plt.xticks(rotation=45, ha='right')
    if save_file:
        Path(f'{path_save}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path_save}/stability-chart-benchmark_{func_name}.png", bbox_inches='tight')
    if verbose:
        plt.show()
    return None


def draw_analysis_of_function(model, func_name, fig_size=(18, 4),
                              save_file=False, path_save="cec", verbose=True):
    # Generate a grid of points for the d1-th and d2-th dimensions - We draw the 1st and 2nd dimensions
    d1 = np.linspace(model.problem.lb[0], model.problem.ub[0], 300)
    d2 = np.linspace(model.problem.lb[1], model.problem.ub[1], 300)
    D1, D2 = np.meshgrid(d1, d2)

    # Fix the other dimensions to zero (or another value within the domain)
    mm_values = (model.problem.lb + model.problem.ub) / 2

    # Combine the fixed and varying dimensions into a single array
    solution = np.full((300, 300, model.problem.n_dims), np.array(mm_values))
    solution[:, :, 0] = D1  # d1-th dimension
    solution[:, :, 1] = D2  # d2-th dimension

    # Compute the function values using vectorized operations
    res = np.apply_along_axis(model.problem.get_target, axis=-1, arr=solution)  # This function return List of target objects
    lambda_func = np.vectorize(lambda x: x.fitness)
    Z = lambda_func(res)

    # Plot the 3D surface of the benchmark function
    fig = plt.figure(figsize=fig_size)  # Width, height

    # 3D plot
    ax1 = fig.add_subplot(151, projection='3d')
    ax1.plot_surface(D1, D2, Z, cmap='viridis', edgecolor='none')
    ax1.set_title(f'3D Surface of {func_name} Benchmark')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('Fitness')

    # 2D plot with population positions
    ## List of positions
    pos_list = []
    for pop in model.history.list_population:
        for agent in pop:
            pos_list.append(agent.solution)
    pos_list = np.array(pos_list)
    # pos_list = np.array([agent.solution for pop in model.history.list_population for agent in pop])
    ax2 = fig.add_subplot(152)
    contour = ax2.contourf(D1, D2, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax2)
    ax2.scatter(pos_list[:, 0], pos_list[:, 1], color='red', label='Population', s=10)
    ax2.scatter(model.g_best.solution[0], model.g_best.solution[1], color='blue', s=100, edgecolor='black', linewidth=3, label='Best Position')
    ax2.set_title('Population search history')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.legend()

    # Trajectory of the first dimension of the first agent
    # trajectory = [agent.solution[0] for agent in model.history.list_global_best]
    list_agents = [pop[0] for pop in model.history.list_population]
    trajectory = [agent.solution[0] for agent in list_agents]
    ax3 = fig.add_subplot(153)
    ax3.plot(trajectory, label='Position')
    ax3.set_title('Trajectory of the 1-th D')
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('X')
    ax3.legend()

    # Average fitness value chart of the whole population
    fit_list = []
    for pop in model.history.list_population:
        fit_temp = np.mean([agent.target.fitness for agent in pop])
        fit_list.append(fit_temp)
    ax4 = fig.add_subplot(154)
    ax4.plot(fit_list, label='Average Fitness', color='green')
    ax4.set_title('Average fitness')
    ax4.set_xlabel('Iterations')
    ax4.set_ylabel('Average fitness')
    ax4.legend()

    # Convergence chart of the global best fitness value
    ax5 = fig.add_subplot(155)
    ax5.plot(model.history.list_global_best_fit, label='Global best fitness', color='purple')
    ax5.set_title('Convergence curve')
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel('Fitness')
    ax5.legend()
    plt.tight_layout()
    if save_file:
        Path(f'{path_save}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path_save}/analysis-benchmark_{func_name}-model_{model.name}.png", bbox_inches='tight')
    if verbose:
        plt.show()
    return None
