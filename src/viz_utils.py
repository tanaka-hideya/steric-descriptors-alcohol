"""
File: viz_utils.py
Author: Hideya Tanaka
Reviewer: Tomoyuki Miyao
Description: This module provides functions for visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_line_chart(
    csv_filename: str,
    feature_column: str,
    eval_metric: str,
    x_axis_label: str,
    x_axis_vals_list: list,
    feature_groups_list: list,
    legend_labels_list: list,
    distinct_markers_list: list,
    plot_title: str = "",
    output_filename: str = None,
    font_size: int = 12
):
    """
    Read a CSV file and plot a line chart for a specified evaluation metric.
    This function extracts data points from different groups of features by pairing each provided
    x-axis value with the corresponding feature name from each group. A sanity check is performed to
    ensure that each feature name contains the expected x-axis value as a substring.
    
    Parameters:
    - csv_filename: Path to the CSV file to read data.
    - feature_column: Name of the column in the CSV that contains the feature names.
    - eval_metric: Column name containing evaluation metric values (e.g., q2, MAE, RMSE, etc.).
    - x_axis_label: Label for the x-axis (e.g., 'Radius').
    - x_axis_vals_list: List of x-axis values (e.g., [1.5, 2, 2.5]) to be used in order.
    - feature_groups_list: Double list where each inner list contains feature names corresponding to one group,
                           arranged in the same order as x_axis_vals_list.
    - legend_labels_list: List of legend labels corresponding to each feature group.
    - distinct_markers_list: List of distinct markers for differentiating each line plot.
    - plot_title: Title for the plot.
    - output_filename: Optional file path to save the plot image. If not provided, the plot is displayed.
    - font_size: Base font size for the plot; used for titles, axis labels, and tick labels.
    
    Raises:
    - ValueError: If required columns are missing,
                  if the number of feature groups does not match the number of legend labels,
                  if the length of a feature group does not match the number of x-axis values,
                  or if a feature name does not contain the expected x-axis value.
    """
    
    plt.rcParams.update({'font.size': font_size})

    df = pd.read_csv(csv_filename)
        
    # Check that the required columns exist in the dataframe
    if feature_column not in df.columns:
        raise ValueError(f"Column '{feature_column}' not found in CSV file")
    if eval_metric not in df.columns:
        raise ValueError(f"Column '{eval_metric}' not found in CSV file")
    
    # Check that the number of feature groups equals the number of legend labels and markers
    if len(feature_groups_list) != len(legend_labels_list):
        raise ValueError("Number of feature groups does not match number of legend labels")
    if len(feature_groups_list) != len(distinct_markers_list):
        raise ValueError("Number of feature groups does not match number of distinct markers")
    
    plt.figure()
    
    # Iterate over each feature group and plot corresponding line using a paired loop with x_axis_vals_list
    for group_idx, current_feature_group_list in enumerate(feature_groups_list):
        # Check that current feature group length matches the number of x-axis values
        if len(current_feature_group_list) != len(x_axis_vals_list):
            raise ValueError(f"Feature group {group_idx} length does not match number of x-axis values")
        
        x_values_group_list = []
        y_values_group_list = []
        
        # Iterate over x_axis_vals_list and the corresponding feature names directly
        for x_val, feature_name in zip(x_axis_vals_list, current_feature_group_list):
            # Check if the x-axis value (converted to string) is present in the feature name
            if str(x_val) not in feature_name:
                raise ValueError(f"Feature name '{feature_name}' does not contain the expected x value '{x_val}'")
            
            # Locate the row in the dataframe that exactly matches the feature name
            feature_row_df = df[df[feature_column] == feature_name]
            if feature_row_df.empty:
                raise ValueError(f"Feature name '{feature_name}' not found in CSV file")
            if len(feature_row_df) > 1:
                raise ValueError(f"Multiple rows found for feature name '{feature_name}'; expected unique entry")
            
            metric_value = feature_row_df[eval_metric].values[0]
            
            try:
                numeric_x_val = float(x_val)
            except Exception:
                numeric_x_val = x_val
            x_values_group_list.append(numeric_x_val)
            y_values_group_list.append(metric_value)
        
        current_marker = distinct_markers_list[group_idx]
        plt.plot(x_values_group_list, y_values_group_list, marker=current_marker, label=legend_labels_list[group_idx])
    
    if plot_title:
        plt.title(plot_title)
    plt.xlabel(x_axis_label)
    plt.ylabel(eval_metric)
    legend_size = font_size - 5
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': legend_size})
    plt.grid(True)
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_scatter_with_regression(
    csv_filename: str,
    x_feature: str,
    x_display_name: str,
    y_feature: str,
    y_display_name: str,
    plot_title: str = "",
    output_filename: str = None,
    scatter_size: int = 50,
    regression_line_width: float = 1.0,
    font_size: int = 12
):
    """
    Create a scatter plot with a regression line from a CSV file containing feature columns.
    This function reads a CSV file, plots a scatter plot with a regression line using the specified
    x and y feature columns, and customizes the plot aesthetics for clarity.
    
    Parameters:
    - csv_filename: Path to the CSV file to load data.
    - x_feature: Column name in the CSV file used for x-axis values.
    - x_display_name: Label to display on the x-axis.
    - y_feature: Column name in the CSV file used for y-axis values.
    - y_display_name: Label to display on the y-axis.
    - plot_title: Title of the plot.
    - output_filename: Path to save the plot image (e.g., 'plot.png'); if empty, the plot is displayed.
    - scatter_size: Marker size for the scatter plot (default: 50).
    - regression_line_width: Line width for the regression line (default: 1.0).
    - font_size: Base font size for the plot (default: 12).
    
    Raises:
    - ValueError: If the specified columns are not found in the CSV.
    """

    plt.rcParams.update({'font.size': font_size})
    
    df = pd.read_csv(csv_filename)
    
    # Check that the x_feature and y_feature exist in the dataframe
    if x_feature not in df.columns:
        raise ValueError(f"Column '{x_feature}' not found in CSV file")
    if y_feature not in df.columns:
        raise ValueError(f"Column '{y_feature}' not found in CSV file")
    
    scatter_plot = sns.lmplot(
        x=x_feature,
        y=y_feature,
        data=df,
        aspect=1.414,
        scatter_kws={'s': scatter_size},
        line_kws={'linewidth': regression_line_width}
    )
    
    scatter_plot.ax.tick_params(axis='both', direction='in', width=1, colors='black')
    
    scatter_plot.ax.spines['top'].set_visible(True)
    scatter_plot.ax.spines['right'].set_visible(True)
    scatter_plot.ax.spines['left'].set_visible(True)
    scatter_plot.ax.spines['bottom'].set_visible(True)
    for spine_key, spine_obj in scatter_plot.ax.spines.items():
        spine_obj.set_linewidth(1)
        spine_obj.set_color('black')
    
    if plot_title:
        scatter_plot.ax.set_title(plot_title, fontsize=font_size + 2)
    scatter_plot.ax.set_xlabel(x_display_name, fontsize=font_size)
    scatter_plot.ax.set_ylabel(y_display_name, fontsize=font_size)
    
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

