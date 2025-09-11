import pandas as pd 
import os 
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('final_results_3.csv')




def prepare_heatmap_data(precision):
    # Load results

    mask =  (df['precision_e'] == precision) & (df['precision_b'] == precision)

    results_df = df[mask].copy()

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

    return results_df, precision


def plot_heatmap(results_df, precision):
    
    # Prepare grid
    theta_e_list = np.sort(results_df['theta_e'].unique())
    theta_b_list = np.sort(results_df['theta_b'].unique())

    grid = results_df.pivot(index='theta_b', columns='theta_e', values='winner_label').values

    # Define custom colormap and normalization
    cmap = mcolors.ListedColormap(['#1f77b4', '#2ca02c','#ff7f0e', '#7f7f7f'])  # CF_BL, BL,  Undefined
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]  # 4 intervals -> 4 colors
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot
    plt.figure(figsize=(8, 6))
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
    plt.title('Dominant Profit Region, lambda = ' + str(precision) )

    plt.tight_layout()
    plt.savefig('heatmap_3_lambda_' + str(precision) + '.png', dpi=300)

if __name__ == "__main__":

    for precision in [10,100,1000]:

        results_df, precision = prepare_heatmap_data(precision)

        plot_heatmap(results_df, precision)
