import pandas as pd 
import os 
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Load the data
df = pd.read_csv('results/final_results_3.csv')

# Define the grid parameters
GRID = {
    # Model size
    "N_e": [1000],                # e.g. [500, 1000, 2000]
    "N_b": [1000],
    "M_e": [100],
    "M_b": [100],

    # Economic parameters
    "K":  [600],                  # e.g. [400, 600, 800]
    "p":  [1],                    # e.g. [0.5, 0.75, 1.0]
    "c":  [0.6],                  # e.g. [0.2, 0.4, 0.6]
    "t":  [0.08],                 # e.g. [0.02, 0.05, 0.08]
    "rf": [0.05],                 # e.g. [0.01, 0.03, 0.05]

    # Behavioral parameters
    "precision_e": [10, 100, 1000],
    "precision_b": [10, 100, 1000],
}

def get_all_combinations(GRID):
    """Generate all possible combinations of parameter values"""
    param_names = list(GRID.keys())
    param_values = list(GRID.values())
    
    # Get all combinations
    combinations = list(product(*param_values))
    
    # Convert to list of dictionaries
    combo_dicts = []
    for combo in combinations:
        combo_dict = dict(zip(param_names, combo))
        combo_dicts.append(combo_dict)
    
    return combo_dicts

def create_mask_for_combination(df, combo_dict):
    """Create a mask for a specific parameter combination"""
    mask = pd.Series([True] * len(df))
    
    for param, value in combo_dict.items():
        mask &= (df[param] == value)
    
    return mask

def prepare_heatmap_data(df, combo_dict):
    """Prepare data for heatmap for a specific parameter combination"""
    
    # Create mask for this specific combination
    mask = create_mask_for_combination(df, combo_dict)
    
    results_df = df[mask].copy()
    
    if len(results_df) == 0:
        print(f"Warning: No data found for combination: {combo_dict}")
        return None, combo_dict
    
    # Check if we have theta_e and theta_b variations
    if len(results_df['theta_e'].unique()) <= 1 or len(results_df['theta_b'].unique()) <= 1:
        print(f"Warning: Insufficient theta variations for combination: {combo_dict}")
        return None, combo_dict
    
    # Determine which variable is highest at each point
    def get_winner(row):
        values = {
            'CF_BL': row['CF_BL'],
            'BL': row['BL'],
        }
        if any(pd.isna(v) for v in values.values()):
            return 'Undefined'
        # if both strategies yield negative payoffs → exit
        if all(v <= 0 for v in values.values()):
            return 'Exit'
        
        return max(values, key=values.get)

    results_df['winner'] = results_df.apply(get_winner, axis=1)

    # Map winners to integer labels
    label_mapping = {
        'CF_BL': 0,
        'BL': 1,
        'Exit': 2,
        'Undefined': 3,
    }

    results_df['winner_label'] = results_df['winner'].map(label_mapping)

    return results_df, combo_dict

def create_filename_and_title(combo_dict):
    """Create descriptive filename and title from parameter combination"""
    
    # Create filename
    filename_parts = []
    title_parts = []
    
    for param, value in combo_dict.items():
        filename_parts.append(f"{param}_{value}")
        
        # Format value for title
        if isinstance(value, float):
            if value < 0.01:
                formatted_value = f"{value:.4f}"
            elif value < 0.1:
                formatted_value = f"{value:.3f}"
            elif value < 1:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
        else:
            formatted_value = str(value)
        
        # Use Greek letters for special parameters
        if param == 'precision_e':
            title_parts.append(f"λₑ={formatted_value}")
        elif param == 'precision_b':
            title_parts.append(f"λᵦ={formatted_value}")
        else:
            title_parts.append(f"{param}={formatted_value}")
    
    filename = "images/heatmap_" + "_".join(filename_parts) + ".png"
    title = "Dominant Profit Region: " + ", ".join(title_parts)
    
    return filename, title

def plot_heatmap(results_df, combo_dict):
    """Plot heatmap for the given parameter combination"""
    
    if results_df is None or len(results_df) == 0:
        print(f"Cannot plot heatmap for combination {combo_dict}: No data available")
        return
    
    # Prepare grid
    theta_e_list = np.sort(results_df['theta_e'].unique())
    theta_b_list = np.sort(results_df['theta_b'].unique())
    
    print(f"Combination {combo_dict}:")
    print(f"  theta_e range = {theta_e_list.min():.3f} to {theta_e_list.max():.3f}")
    print(f"  theta_b range = {theta_b_list.min():.3f} to {theta_b_list.max():.3f}")
    print(f"  Data points = {len(results_df)}")

    try:
        grid = results_df.pivot(index='theta_b', columns='theta_e', values='winner_label').values
    except Exception as e:
        print(f"Error creating pivot table for combination {combo_dict}: {e}")
        return

    # Define custom colormap and normalization
    cmap = mcolors.ListedColormap(['#1f77b4', '#2ca02c', '#ff7f0e', '#7f7f7f'])  # CF_BL, BL, Exit, Undefined
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]  # 4 intervals -> 4 colors
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create filename and title
    filename, title = create_filename_and_title(combo_dict)

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        grid,
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=[theta_e_list.min(), theta_e_list.max(), theta_b_list.min(), theta_b_list.max()],
        aspect='auto'
    )

    # Legend
    legend_elements = [
        Patch(facecolor='#1f77b4', label='CF_BL'),
        Patch(facecolor='#2ca02c', label='BL'),
        Patch(facecolor='#ff7f0e', label='Exit'),
        Patch(facecolor='#7f7f7f', label='Undefined')
    ]
    plt.legend(handles=legend_elements, title='Highest Profit', loc='upper right')

    plt.xlabel(r'$\theta_e$')
    plt.ylabel(r'$\theta_b$')
    plt.title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved as: {filename}\n")

if __name__ == "__main__":
    
    # Get all parameter combinations
    all_combinations = get_all_combinations(GRID)
    print(f"Total combinations to process: {len(all_combinations)}")
    
    # Generate heatmaps for each combination
    successful_plots = 0
    for i, combo_dict in enumerate(all_combinations):
        print(f"\nProcessing combination {i+1}/{len(all_combinations)}")
        
        results_df, combo_dict = prepare_heatmap_data(df, combo_dict)
        
        if results_df is not None:
            plot_heatmap(results_df, combo_dict)
            successful_plots += 1
    
    print(f"\nCompleted! Successfully generated {successful_plots} heatmaps out of {len(all_combinations)} combinations.")