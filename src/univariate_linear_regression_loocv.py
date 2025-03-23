"""
File: univariate_linear_regression_loocv.py
Author: Hideya Tanaka
Reviewer: Tomoyuki Miyao
Description:
    This script performs univariate linear regression with LOOCV evaluation metrics.
    It computes LOOCV metrics (q2, MAE, RMSE) for each feature and outputs plots and CSV files.
    For visualization, if there are too many features, the results are split into multiple PNGs
    so that each figure fits on one page with legible font sizes. The CSV file contains all features.
    The best feature based on a chosen evaluation metric is reported along with a full regression model fit.
    LOOCV is parallelized and the evaluation metric for sorting can be controlled via an argument.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import cpu_count, Parallel, delayed

def evaluate_model(X, y, train_idx, test_idx):
    # Train and evaluate a linear regression model on a single LOOCV split
    model = LinearRegression()
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    return y_pred[0], y[test_idx][0]

def compute_loocv_metrics(X, y, n_jobs=-1):
    """
    Compute LOOCV evaluation metrics (q2, MAE, RMSE) for a univariate linear regression model
    """
    n_jobs = cpu_count() -1 if n_jobs < 1 else n_jobs
    loo = LeaveOneOut()
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_model)(X, y, train_idx, test_idx) for train_idx, test_idx in loo.split(X)
    )
    preds_list = [result[0] for result in results_list]
    actuals_list = [result[1] for result in results_list]
    q2_value = r2_score(actuals_list, preds_list)
    mae_value = mean_absolute_error(actuals_list, preds_list)
    rmse_value = np.sqrt(mean_squared_error(actuals_list, preds_list))
    return {'q2': q2_value, 'MAE': mae_value, 'RMSE': rmse_value}

def compute_full_regression_metrics(X, y):
    """
    Compute full regression model metrics using all data
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_full = r2_score(y, y_pred)
    mae_full = mean_absolute_error(y, y_pred)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred))
    return model, r2_full, mae_full, rmse_full

def plot_sorted_evaluation_metric_values(sorted_features_list, eval_metric, keyword_name, keyword, outfd, max_features_per_plot, fig_width):
    """
    Generate and save horizontal bar charts for a specified evaluation metric using the sorted order
    """
    features_order_list = [feat for feat, met in sorted_features_list]
    metric_values_list = [met[eval_metric] for feat, met in sorted_features_list]

    sns.set_style('white')
    plt.rcParams.update({'font.size': 10})
    magma_cmap = plt.get_cmap('magma')
    bar_color = magma_cmap(0.8)

    num_features = len(features_order_list)
    num_plots = (num_features - 1) // max_features_per_plot + 1
    
    for i in range(num_plots):
        start_idx = i * max_features_per_plot
        end_idx = min((i+1) * max_features_per_plot, num_features)
        chunk_features = features_order_list[start_idx:end_idx]
        chunk_metrics = metric_values_list[start_idx:end_idx]
        
        fig_height = 0.3 * len(chunk_features) + 2
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
        ax = sns.barplot(
            x=chunk_metrics,
            y=chunk_features,
            color=bar_color,
            edgecolor=None
        )
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

        if eval_metric == 'q2':
            ax.set_xlabel('$q^2$', fontsize=10)
        else:
            ax.set_xlabel(eval_metric, fontsize=10)
        ax.set_ylabel('Features', fontsize=10)

        for j, val in enumerate(chunk_metrics):
            ax.text(val + 0.001, j, f'{val:.3f}', va='center', color='black', fontsize=9)
        
        if keyword_name == 'All_Features':
            plot_name = f'{keyword}'
        else:
            plot_name = f'{keyword}_{keyword_name}'
        
        if num_plots == 1:
            ax.set_title(f'{plot_name}', fontsize=12, loc='right')
            output_png = os.path.join(outfd, f'{plot_name}.png')
        else:
            ax.set_title(f'{plot_name} (Part {i+1})', fontsize=12, loc='right')
            output_png = os.path.join(outfd, f'{plot_name}_part{i+1}.png')
        plt.tight_layout()
        plt.savefig(output_png, dpi=300)
        plt.close(fig)

def run_univariate_linear_regression_loocv_evaluation(fd, dataset_file_path_input, keyword, target_column, exclude_columns, eval_metric='q2', sort_flag=True, n_jobs=-1, max_features_per_plot=53, fig_width=9, search_keywords_list=False):
    """
    Run univariate linear regression with LOOCV evaluation metrics and full regression analysis
    """
    print('LinearRegression_LOOCV_Evaluation')
    print('========== Settings ==========')
    print(f'dataset_file_path_input: {dataset_file_path_input}')
    print(f'keyword: {keyword}')
    print(f'search_keywords: {search_keywords_list if search_keywords_list else "All features"}')
    print(f'target_column: {target_column}')
    print(f'exclude_columns: {exclude_columns}')
    print(f'evaluation_metric: {eval_metric}')
    print(f'sort: {sort_flag}')
    print('------------------------------')

    outfd = os.path.join(fd, keyword)
    os.makedirs(outfd, exist_ok=True)

    df = pd.read_csv(dataset_file_path_input, index_col=0)
    print(f'All molecules in the dataset: {len(df)}')

    features_df = df.drop(columns=exclude_columns)
    features_df = features_df.select_dtypes(include=[float, int])
    print(f'only_numeric_df.shape: {features_df.shape}')

    feature_columns_list = features_df.columns.tolist()
    y = df[target_column].values

    # Determine groups of features based on search keywords; use all features if search_keywords_list is empty
    if not search_keywords_list:
        groups_dict = {'All_Features': feature_columns_list}
    else:
        groups_dict = {}
        for each_key in search_keywords_list:
            group_features_list = [col for col in feature_columns_list if each_key in col]
            if not group_features_list:
                raise ValueError(f"No features found for keyword '{each_key}'")
            groups_dict[each_key] = group_features_list

    # Iterate over each group of features
    for keyword_name, group_features_list in groups_dict.items():
        metrics_dict = {}
        for feature in group_features_list:
            X = df[[feature]].values
            metrics_dict[feature] = compute_loocv_metrics(X, y, n_jobs=n_jobs)
        
        # Sort features based on the selected evaluation metric if required
        # Determine the best feature
        if sort_flag:
            # For q2, higher is better (descending); for MAE or RMSE, lower is better (ascending)
            reverse_order = True if eval_metric == 'q2' else False
            sorted_features_list = sorted(metrics_dict.items(), key=lambda item: item[1][eval_metric], reverse=reverse_order)
            best_feature, best_metrics = sorted_features_list[0]
        else:
            # List without sorting
            sorted_features_list = list(metrics_dict.items())
            if eval_metric == 'q2':
                best_feature, best_metrics = max(metrics_dict.items(), key=lambda item: item[1][eval_metric])
            else:
                best_feature, best_metrics = min(metrics_dict.items(), key=lambda item: item[1][eval_metric])
        
        # Create a DataFrame for CSV output with columns: Feature, q2, MAE, RMSE
        csv_data = {
            'Feature': [feat for feat, met in sorted_features_list],
            'q2': [met['q2'] for feat, met in sorted_features_list],
            'MAE': [met['MAE'] for feat, met in sorted_features_list],
            'RMSE': [met['RMSE'] for feat, met in sorted_features_list]
        }
        metrics_df = pd.DataFrame(csv_data)
        output_csv = os.path.join(outfd, f'{keyword}_{keyword_name}.csv')
        metrics_df.to_csv(output_csv, index=False)
        
        # Plot evaluation metric values using the same sorted order as CSV output
        plot_sorted_evaluation_metric_values(sorted_features_list, eval_metric, keyword_name, keyword, outfd, max_features_per_plot, fig_width)
        
        # Compute full regression on the best feature using all data
        X_best = df[[best_feature]].values
        full_model, r2_full, mae_full, rmse_full = compute_full_regression_metrics(X_best, y)
        # Generate linear regression equation string
        coef = full_model.coef_[0]
        intercept = full_model.intercept_
        equation = f'y = {coef:.4f} * x + {intercept:.4f}'
        
        # Log detailed results in the text file (including full regression results computed on all data)
        log_path = os.path.join(outfd, f'log_metrics_{keyword}.txt')
        with open(log_path, 'a') as log_file:
            log_file.write('------------------------------\n')
            log_file.write(f'Group: {keyword_name}\n')
            log_file.write(f'Best feature based on LOOCV {eval_metric}: {best_feature}\n')
            log_file.write('LOOCV metrics:\n')
            log_file.write(f'  q2 (LOOCV): {best_metrics["q2"]:.4f}\n')
            log_file.write(f'  MAE (LOOCV): {best_metrics["MAE"]:.4f}\n')
            log_file.write(f'  RMSE (LOOCV): {best_metrics["RMSE"]:.4f}\n')
            log_file.write('Full regression using all data:\n')
            log_file.write(f'  Linear equation: {equation}\n')
            log_file.write(f'  R2: {r2_full:.4f}\n')
            log_file.write(f'  MAE: {mae_full:.4f}\n')
            log_file.write(f'  RMSE: {rmse_full:.4f}\n')
            log_file.write('Note: LOOCV metrics are computed using leave-one-out cross-validation, whereas full regression metrics are computed using the entire dataset\n')
        
        # Also print full regression results to standard output
        print('------------------------------')
        print(f'Group: {keyword_name}')
        print(f'Best feature based on LOOCV {eval_metric}: {best_feature}')
        print('LOOCV metrics:')
        print(f'  q2 (LOOCV): {best_metrics["q2"]:.4f}')
        print(f'  MAE (LOOCV): {best_metrics["MAE"]:.4f}')
        print(f'  RMSE (LOOCV): {best_metrics["RMSE"]:.4f}')
        print('Full regression using all data:')
        print(f'  Linear equation: {equation}')
        print(f'  R2: {r2_full:.4f}')
        print(f'  MAE: {mae_full:.4f}')
        print(f'  RMSE: {rmse_full:.4f}')

if __name__ == '__main__':
    fd = os.path.dirname(os.path.abspath(__file__))
    
    # ========== Settings (CHANGE HERE) ==========
    # Input CSV file
    dataset_file_path_input = os.path.join(fd, 'full_dataset_alcohol_oxidation_water_effect.csv')
    
    # Identifier for output directories and files
    keyword = 'LinearRegression_LOOCV_woH2O'
    
    # Name of the target variable column in the dataset
    target_column = 'conv_woH2O_10min'
    
    # List of columns to be excluded from feature selection
    exclude_columns = [target_column, 'smiles', 'confid', 'total_energy_xTB', 'Zero_point_correction', 'Thermal_correction_to_Energy',
                       'Thermal_correction_to_Enthalpy', 'Thermal_correction_to_Gibbs_Free_Energy',
                       'Sum_of_electronic_and_zero_point_Energies', 'Sum_of_electronic_and_thermal_Energies',
                       'Sum_of_electronic_and_thermal_Enthalpies', 'Sum_of_electronic_and_thermal_Free_Energies',
                       'HOMO', 'LUMO', 'filepath', 'HOMO_eV', 'LUMO_eV', 'chemical_potential_eV', 'chemical_hardness_eV',
                       'GEI_eV', 'conv_wH2O_10min']
    
    # Evaluation metric to be used for selection and plotting ('q2', 'MAE', or 'RMSE')
    eval_metric = 'q2'
    
    # Flag to control whether sorting is applied based on the evaluation metric (bool)
    sort_flag = True
    
    # Number of CPU cores to use for parallel LOOCV computation
    n_jobs = -1
    
    # Maximum number of features per plot for clear visualization
    max_features_per_plot = 53
    
    # Figure width for the evaluation metric plots
    fig_width = 9
    
    # List of keywords for selecting features; if an empty list is provided, all features are used ([])
    search_keywords_list = ['center_C', 'center_O', 'center_H(O)', 'center_H(C)']
    # =============================================
    
    log_file_path = os.path.join(fd, f'log_{keyword}.txt')
    sys.stdout = open(log_file_path, 'w')
    run_univariate_linear_regression_loocv_evaluation(fd, dataset_file_path_input, keyword, target_column, exclude_columns, eval_metric, sort_flag, n_jobs, max_features_per_plot, fig_width, search_keywords_list)
    print('Finish')
    sys.stdout.close()
