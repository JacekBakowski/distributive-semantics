### HERE ARE SOME USEFUL FUNCTIONS I USE IN THE WHOLE PAPER ###

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from gensim.models import Word2Vec as MODEL
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.pyplot import figure
from prettytable import PrettyTable
from scipy.special import comb
from scipy.stats import binom

# ANSI color codes for terminal output
CYAN = '\033[1;36m'
YELLOW = '\033[1;33m'
WHITE = '\033[1;37m'
MAGENTA = '\033[1;35m'
GREEN = '\033[1;32m'
BLUE = '\033[1;34m'
ORANGE = '\033[38;5;208m'
RED = '\033[1;31m'
RESET = '\033[0m'

# Color assignments for word origins
PA_COLOR = GREEN      # Perso-Arabic words in green
SKRT_COLOR = ORANGE   # Sanskrit words in orange


#############################################################################
#                                                                           #
#                    L O A D  W O R D 2 V E C  M O D E L                    #
#                                                                           #
#############################################################################

def load_model(MODEL_FILE):
    """
    Load Word2Vec model and display its specifications.
    
    Args:
        MODEL_FILE: Path to the Word2Vec model file
        
    Returns:
        Loaded Word2Vec model object
    """
    
    model_loaded = MODEL.load(MODEL_FILE)
    model_name = os.path.basename(MODEL_FILE)
    
    
    # Display Model Specifications with colored headers and values:
    header = [
        f'{CYAN}Vector Size{RESET}',
        f'{YELLOW}Corpus count{RESET}',
        f'{YELLOW}Corpus total words{RESET}',
        f'{WHITE}Min count{RESET}',
        f'{WHITE}Effective min count{RESET}',
        f'{MAGENTA}Epochs{RESET}',
        f'{GREEN}Total train time{RESET}',
        f'{BLUE}Filename{RESET}'
    ]
    table = [[
        f"{CYAN}{model_loaded.vector_size}{RESET}",
        f"{YELLOW}{model_loaded.corpus_count:,}{RESET}",
        f"{YELLOW}{model_loaded.corpus_total_words:,}{RESET}",
        model_loaded.min_count, 
        model_loaded.effective_min_count, 
        f"{MAGENTA}{model_loaded.epochs}{RESET}",
        f"{GREEN}{model_loaded.total_train_time:.2f}s{RESET}",
        f"{BLUE}{model_name}{RESET}"
    ]]
    tab = PrettyTable()
    tab.title = f'{BLUE}W2V Model Specifications{RESET}'
    tab.field_names = header
    tab.add_rows(table)
    print(tab)
    return model_loaded


#############################################################################
#                                                                           #
#                 D I S P L A Y  M O D E L  A C C U R A C Y                 #
#                                                                           #
#############################################################################

def plot_model_efficiency(x_axe, xlabel, title, total_results, grayscale=False):
    """
    Plot model efficiency metrics including accuracy and precision/recall/f1 scores.
    
    Args:
        x_axe: Number of data points for x-axis
        xlabel: Label for x-axis
        title: Plot title
        total_results: Dictionary containing metric arrays (accuracy, precision_score_PA, etc.)
        grayscale: If True, use grayscale with different line styles for print publication;
                   If False (default), use colors for online publication
    """
    plt.figure(dpi=1200)
    x_axe = list(range(1, x_axe))
    
    if grayscale:
        # Grayscale version for print publications - only line styles, no markers
        # Accuracy - thick solid black line (most prominent, on top)
        line1, = plt.plot(x_axe, total_results['accuracy'], color="black", 
                         label="Accuracy", linewidth=4, linestyle='-', 
                         solid_capstyle='round', solid_joinstyle='round', zorder=10)
        
        # PA metrics - dark gray with different line styles
        line2, = plt.plot(x_axe, total_results['precision_score_PA'], color="dimgray", 
                         label="Precision score PA", linewidth=1, linestyle='--', zorder=5)
        line3, = plt.plot(x_axe, total_results['recall_score_PA'], color="dimgray", 
                         label="Recall score PA", linewidth=1, linestyle=(0, (3, 1, 1, 1)), zorder=4)
        line4, = plt.plot(x_axe, total_results['f1_score_PA'], color="dimgray", 
                         label="f1 score PA", linewidth=1, linestyle=':', zorder=3)
        
        # SKRT metrics - lighter gray with different line styles and markers
        line5, = plt.plot(x_axe, total_results['precision_score_SKRT'], color="gray", 
                         label="Precision score SKRT", linestyle='none', 
                         marker='^', markersize=2, markevery=1, markerfacecolor='none', markeredgecolor='gray', markeredgewidth=1, zorder=2)
        line6, = plt.plot(x_axe, total_results['recall_score_SKRT'], color="gray", 
                         label="Recall score SKRT", linestyle='none', 
                         marker='o', markersize=2, markevery=1, markerfacecolor='none', markeredgecolor='gray', markeredgewidth=1, zorder=1)
        line7, = plt.plot(x_axe, total_results['f1_score_SKRT'], color="lightgray", 
                         label="f1 score SKRT", linestyle='none', 
                         marker='s', markersize=2, markerfacecolor='none', markeredgecolor='darkgray', markeredgewidth=1, zorder=0)
    else:
        # Color version for online publications
        line1, = plt.plot(x_axe, total_results['accuracy'], color="#1f77b4", 
                         label="Accuracy", linewidth=4, 
                         solid_capstyle='round', solid_joinstyle='round', zorder=9)
     
        line2, = plt.plot(x_axe, total_results['precision_score_PA'], color="red", 
                         label="Precision score PA")
        line3, = plt.plot(x_axe, total_results['recall_score_PA'], color="green", 
                         label="Recall score PA")
        line4, = plt.plot(x_axe, total_results['f1_score_PA'], color='black', 
                         label="f1 score PA")
        
        line5, = plt.plot(x_axe, total_results['precision_score_SKRT'], color="orange", 
                         label="Precision score SKRT")
        line6, = plt.plot(x_axe, total_results['recall_score_SKRT'], color="gray", 
                         label="Recall score SKRT")
        line7, = plt.plot(x_axe, total_results['f1_score_SKRT'], color='yellow', 
                         label="f1 score SKRT")
        
    plt.title(title, fontsize = 8)
    # Create custom handler map with more points for marker-based lines
    if grayscale:
        plt.legend(handler_map={
            line1: HandlerLine2D(numpoints=2),
            line5: HandlerLine2D(numpoints=5),
            line6: HandlerLine2D(numpoints=5),
            line7: HandlerLine2D(numpoints=5)
        })
    else:
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.xlabel(xlabel, fontsize = 6)
    plt.ylabel('Average Model Efficiency', fontsize = 6)
    
    # Add subtle grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.show()



#############################################################################
#                                                                           #
#      D I S P L A Y  E R R O R S  A C C O R D I N G  T O  O R I G I N      #
#                                                                           #
#############################################################################

def display_errors(errors, sort_by='Errors', title='', grayscale=False):
    """
    Display scatter plot of errors by origin (Sanskrit vs Perso-Arabic).
    
    Args:
        errors: DataFrame containing error data with 'Origin' and 'Errors' columns
        sort_by: Column name to sort by (default: 'Errors')
        title: Optional plot title
        grayscale: If True, use black/white colors for print publication; 
                   If False (default), use red/green colors for online publication
    """
    errors_coloured = errors.copy()
    
    if grayscale:
        # Black and white version for print publications
        errors_coloured['color'] = np.where(errors_coloured['Origin']=='SKRT', 'black', 'lightgray')
        errors_coloured['marker'] = np.where(errors_coloured['Origin']=='SKRT', 'o', '^')  # circle vs triangle
        # Legend will be created after plotting with actual markers
    else:
        # Color version for online publications
        errors_coloured['color'] = np.where(errors_coloured['Origin']=='SKRT', 'red', 'green')
        errors_coloured['marker'] = 'o'
        skrt_label = mpatches.Patch(color='red', label='Sanskrit Words')
        pa_label = mpatches.Patch(color='green', label='P-A Words')
    
    errors_coloured = errors_coloured.sort_values(by=[sort_by], ascending=False).reset_index()

    figure(figsize=(10, 7), dpi=140)
    
    if grayscale:
        # Plot with different markers for better distinction in grayscale
        for origin in ['SKRT', 'PA']:
            mask = errors_coloured['Origin'] == origin
            if origin == 'SKRT':
                # Sanskrit: unfilled circles with thick black outline
                plt.scatter(x=errors_coloured[mask].index, 
                           y=errors_coloured[mask]['Errors'], 
                           marker='o',
                           s=60,
                           facecolors='none',
                           edgecolors='black',
                           linewidth=2)
            else:
                # Perso-Arabic: filled gray triangles
                plt.scatter(x=errors_coloured[mask].index, 
                           y=errors_coloured[mask]['Errors'], 
                           marker='^',
                           s=60,
                           facecolors='darkgray',
                           edgecolors='black',
                           linewidth=0.5)
    else:
        plt.scatter(x=errors_coloured.index, y=errors_coloured['Errors'], c=errors_coloured['color'])
    
    # Create legend
    if grayscale:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Sanskrit Words',
                   markerfacecolor='none', markeredgecolor='black', markersize=8, markeredgewidth=2),
            Line2D([0], [0], marker='^', color='w', label='P-A Words',
                   markerfacecolor='darkgray', markeredgecolor='black', markersize=8, markeredgewidth=0.5)
        ]
        plt.legend(handles=legend_elements)
    else:
        skrt_label = mpatches.Patch(color='red', label='Sanskrit Words')
        pa_label = mpatches.Patch(color='green', label='P-A Words')
        plt.legend(handles=[skrt_label, pa_label])
    
    plt.xlabel('Word rank')
    plt.ylabel('Error rate')
    if title:
        plt.title(title)
    plt.show()


def analyze_errors(errors):
    """
    Analyze error patterns for top 20 and top 50 misclassified words.
    
    Args:
        errors: DataFrame with error data sorted by error frequency
        
    Prints summary tables of error distribution by origin (SKRT vs PA).
    """
    first_10_SKRT_errors = errors[0:10].loc[(errors['Origin'] == 'SKRT')]['Errors'].sum()
    first_10_PA_errors = errors[0:10].loc[(errors['Origin'] == 'PA')]['Errors'].sum()
    first_10_total_errors = errors[0:10]['Errors'].sum()
    first_20_SKRT_errors = errors[0:20].loc[(errors['Origin'] == 'SKRT')]['Errors'].sum()
    first_20_PA_errors = errors[0:20].loc[(errors['Origin'] == 'PA')]['Errors'].sum()
    first_20_total_errors = errors[0:20]['Errors'].sum()
    first_50_SKRT_errors = errors[0:50].loc[(errors['Origin'] == 'SKRT')]['Errors'].sum()
    first_50_PA_errors = errors[0:50].loc[(errors['Origin'] == 'PA')]['Errors'].sum()
    first_50_total_errors = errors[0:50]['Errors'].sum()
    first_75_SKRT_errors = errors[0:75].loc[(errors['Origin'] == 'SKRT')]['Errors'].sum()
    first_75_PA_errors = errors[0:75].loc[(errors['Origin'] == 'PA')]['Errors'].sum()
    first_75_total_errors = errors[0:75]['Errors'].sum()
    first_100_SKRT_errors = errors[0:100].loc[(errors['Origin'] == 'SKRT')]['Errors'].sum()
    first_100_PA_errors = errors[0:100].loc[(errors['Origin'] == 'PA')]['Errors'].sum()
    first_100_total_errors = errors[0:100]['Errors'].sum()
    perc_first_10_SKRT_errors = (first_10_SKRT_errors * 100) / first_10_total_errors
    perc_first_10_PA_errors = (first_10_PA_errors * 100) / first_10_total_errors
    perc_first_20_SKRT_errors = (first_20_SKRT_errors * 100) / first_20_total_errors
    perc_first_20_PA_errors = (first_20_PA_errors * 100) / first_20_total_errors
    perc_first_50_SKRT_errors = (first_50_SKRT_errors * 100) / first_50_total_errors
    perc_first_50_PA_errors = (first_50_PA_errors * 100) / first_50_total_errors
    perc_first_75_SKRT_errors = (first_75_SKRT_errors * 100) / first_75_total_errors
    perc_first_75_PA_errors = (first_75_PA_errors * 100) / first_75_total_errors
    perc_first_100_SKRT_errors = (first_100_SKRT_errors * 100) / first_100_total_errors
    perc_first_100_PA_errors = (first_100_PA_errors * 100) / first_100_total_errors

    def _print_error_table(title, total, skrt, pa, perc_skrt, perc_pa):
        headers = [
            f'{YELLOW}Total Errors{RESET}',
            f'{SKRT_COLOR}Total SKRT Errors{RESET}',
            f'{PA_COLOR}Total PA Errors{RESET}',
            f'{SKRT_COLOR}% SKRT Errors{RESET}',
            f'{PA_COLOR}% PA Errors{RESET}'
        ]
        table = [[
            f"{YELLOW}{total}{RESET}",
            f"{SKRT_COLOR}{skrt}{RESET}",
            f"{PA_COLOR}{pa}{RESET}",
            f"{SKRT_COLOR}{perc_skrt}{RESET}",
            f"{PA_COLOR}{perc_pa}{RESET}"
        ]]
        tab = PrettyTable()
        tab.title = f'{BLUE}{title}{RESET}'
        tab.field_names = headers
        tab.add_rows(table)
        print(tab)

    _print_error_table(
        'First 10 words most prone to misclassification',
        first_10_total_errors, first_10_SKRT_errors, first_10_PA_errors,
        perc_first_10_SKRT_errors, perc_first_10_PA_errors
    )
    
    _print_error_table(
        'First 20 words most prone to misclassification',
        first_20_total_errors, first_20_SKRT_errors, first_20_PA_errors,
        perc_first_20_SKRT_errors, perc_first_20_PA_errors
    )
    
    _print_error_table(
        'First 50 words most prone to misclassification',
        first_50_total_errors, first_50_SKRT_errors, first_50_PA_errors,
        perc_first_50_SKRT_errors, perc_first_50_PA_errors
    )
    
    _print_error_table(
        'First 75 words most prone to misclassification',
        first_75_total_errors, first_75_SKRT_errors, first_75_PA_errors,
        perc_first_75_SKRT_errors, perc_first_75_PA_errors
    )

    _print_error_table(
        'First 100 words most prone to misclassification',
        first_100_total_errors, first_100_SKRT_errors, first_100_PA_errors,
        perc_first_100_SKRT_errors, perc_first_100_PA_errors
    )


#############################################################################
#                                                                           #
#                  P L O T  W O R D  P R O P O R T I O N S                  #
#                                                                           #
#############################################################################

def plot_word_proportions(embeddings, figsize=(10, 6)):
    """
    Create pie charts showing word proportions by Origin and Part of Speech.
    
    Args:
        embeddings (pd.DataFrame): DataFrame containing vocabulary with 'Origin' and 'POS' columns
        figsize (tuple): Figure size as (width, height). Default is (10, 6)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Data visualization: Word distributions by Origin and POS
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Distribution by Origin (PA vs SKRT)
    origin_counts = embeddings['Origin'].value_counts()
    origin_colors = ['darkorange' if origin == 'SKRT' else 'green' for origin in origin_counts.index]

    ax1 = axes[0]
    ax1.pie(origin_counts.values, 
            labels=origin_counts.index, 
            autopct='%1.1f%%', 
            colors=origin_colors,
            startangle=90)
    ax1.set_title('Word Proportions by Origin', fontweight='bold')

    # Right plot: Distribution by Part of Speech
    pos_counts = embeddings['POS'].value_counts()
    pos_colors = plt.cm.Set3(range(len(pos_counts)))  # Use a colormap for variety

    ax2 = axes[1]
    ax2.pie(pos_counts.values, 
            labels=pos_counts.index, 
            autopct='%1.1f%%',
            colors=pos_colors,
            startangle=90)
    ax2.set_title('Word Proportions by POS', fontweight='bold')

    # Adjust layout and display
    #plt.tight_layout()
    #plt.show()
    #plt.close(fig)  # Close the figure to prevent memory leaks and duplicate displays
    
    #return fig

#############################################################################
#                                                                           #
#                   P L O T  D A T A  D I S T R I B U T I O N S             #
#                                                                           #
#############################################################################

def plot_data_distributions(embeddings):
    """
    Create scatter plots showing data distributions by Frequency, Freq_diff and Similarity.
    
    This function creates a 3-panel visualization with:
    - Top left: Distribution by Frequency (log scale)
    - Bottom left: Distribution by Frequency difference 
    - Right: Distribution by Similarity scores
    
    Args:
        embeddings (pd.DataFrame): DataFrame containing vocabulary with 'Freq', 'Freq_diff', and 'Similarity' columns
        title (str): Main title for the plot. Default is empty string
        xlabel (str): Label for x-axis. Default is empty string
        ylabel (str): Label for y-axis. Default is 'Frequency'
        figsize (tuple): Figure size as (width, height). Default is (10, 6) but will be overridden to (18.5, 10.5)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Plotting of the data distribution by Frequency, Freq_diff and Similarity
    fig, axs = plt.subplot_mosaic([['left_top', 'right'],
                                ['left_bottom', 'right']])

    fig.set_size_inches(18.5, 10.5)

    list_of_graphs = [('left_top', 'Freq', 'Frequency'), ('left_bottom', 'Freq_diff', 'Freq_diff'), ('right', 'Similarity', 'Similarity')]

    for graph in list_of_graphs:

            ax = axs[ graph[0] ]
            x = embeddings.copy()
            x = x.sort_values(by=[graph[1]], ascending=True).reset_index()
            #x = x.head(len(x) - 1)

            ax.set_yscale('log')
            ax.scatter(x=x.index, y=x[graph[1]])
            ax.set_title(f'Distribution by {graph[2]}')

    fig.subplots_adjust(wspace=.2)


#############################################################################
#                                                                           #
#                      D I S P L A Y  S H A P  P L O T                      #
#                                                                           #
#############################################################################

def display_shap(shap_values, X_test, grayscale=False, xlabel=None):
    """
    Display SHAP summary plot with customizable styling for publication.
    
    Args:
        shap_values: SHAP values array from TreeExplainer
        X_test: Test data features
        grayscale: If True, use grayscale colormap for print publication;
                   If False (default), use color for online publication
        xlabel: Custom label for x-axis. If None, keeps default SHAP label
    """
    if grayscale:
        # Grayscale version for print publications
        plt.figure(figsize=(8, 5), dpi=1200)
        shap.summary_plot(shap_values[:,:,0], X_test, plot_size=None, show=False, cmap='Greys')
        plt.title('SHAP Feature Importance', fontsize=12, pad=10)
        ax = plt.gca()
        
        # Add edge color to points for better visibility of light points
        for collection in ax.collections:
            collection.set_edgecolor('black')
            collection.set_linewidth(0.3)
            collection.set_sizes([25])  # Slightly larger points for better visibility
        
        ax.tick_params(axis='y', labelsize=8)  # Feature names on the left
        ax.tick_params(axis='x', labelsize=8)  # X-axis values
        ax.xaxis.label.set_size(8)  # X-axis label size
        ax.yaxis.label.set_size(8)  # Y-axis label size
        
        # Add vertical reference line at x=0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2, alpha=0.5, zorder=1)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Change x-axis label if custom label provided
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=8)
        
        cbar_ax = plt.gcf().axes[-1]  # Get colorbar axes
        cbar_ax.set_ylabel('Feature value', fontsize=12, labelpad=-20)  # Colorbar label
        cbar_ax.tick_params(labelsize=10)  # Font size for "High" and "Low"
        
        # Add border around colorbar for better visibility
        for spine in cbar_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
            spine.set_visible(True)
        
        plt.tight_layout()
        plt.show()
    else:
        # Color version for online publications
        plt.figure(figsize=(8, 5), dpi=1200)
        shap.summary_plot(shap_values[:,:,0], X_test, plot_size=None, show=False)
        plt.title('SHAP Feature Importance', fontsize=12, pad=10)
        ax = plt.gca()
        
        # Add edge color to points for better visibility of light points
        for collection in ax.collections:
            collection.set_edgecolor('black')
            collection.set_linewidth(0.3)
            collection.set_sizes([25])  # Slightly larger points for better visibility
        
        ax.tick_params(axis='y', labelsize=8)  # Feature names on the left
        ax.tick_params(axis='x', labelsize=8)  # X-axis values
        ax.xaxis.label.set_size(8)  # X-axis label size
        ax.yaxis.label.set_size(8)  # Y-axis label size
        
        # Add vertical reference line at x=0
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Change x-axis label if custom label provided
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=8)
        
        cbar_ax = plt.gcf().axes[-1]  # Get colorbar axes
        cbar_ax.set_ylabel('Feature value', fontsize=12, labelpad=-20)  # Colorbar label
        cbar_ax.tick_params(labelsize=10)  # Font size for "High" and "Low"
        
        # Add border around colorbar
        for spine in cbar_ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
            spine.set_visible(True)
        
        plt.tight_layout()
        plt.show()


#############################################################################
#                                                                           #
#           D I S P L A Y  C O R R E L A T I O N  M A T R I X               #
#                                                                           #
#############################################################################

def display_correlation_matrix(data, columns, title='Correlation Matrix', grayscale=False):
    """
    Display correlation matrix as a heatmap with optional grayscale version for print.
    
    Args:
        data: DataFrame containing the data
        columns: List of column names to include in correlation matrix
        grayscale: If True, creates grayscale version for print publication
        
    Returns:
        None (displays the plot)
    """
    
    # Create DataFrame with selected columns
    df = pd.DataFrame(data, columns=columns)
    
    # Calculate correlation matrix
    corr_matrix = df.corr(method='pearson')
    
    # Set figure parameters based on grayscale option
    if grayscale:
        cmap = "Greys"
    else:
        cmap = "YlGnBu"
    
    # Create the heatmap
    plt.figure(figsize=(8, 6), dpi=1200)
    ax = sns.heatmap(corr_matrix, cmap=cmap, annot=True, 
                     cbar_kws={'orientation': 'horizontal', 'pad': 0.08}, 
                     linewidths=0.5, linecolor='black')
    
    # Adjust tick label size
    ax.tick_params(axis='both', labelsize=9)
    
    # Set title
    plt.title(title, fontsize=12, pad=10)
    
    # Add border around colorbar
    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)
    
    plt.tight_layout()
    plt.show()


#############################################################################
#                                                                           #
#              C A L C U L A T E  J A C C A R D  I N D E X                  #
#                                                                           #
#############################################################################
def jaccard_index(list1, list2):
    """
    Calculate Jaccard index between two lists.
    
    Jaccard Index = |A ∩ B| / |A ∪ B|
    
    Args:
        list1, list2: Lists to compare
        
    Returns:
        float: Jaccard index (0 to 1)
    """
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def expected_intersection_with_replacement(N, n):
    """Expected value of |A ∩ B| for independent sampling."""
    # E[|A ∩ B|] = N × P(element in both) = N × (n/N)²
    return N * (n / N) ** 2


def expected_jaccard_with_replacement(N, n):
    """
    Expected Jaccard index for independent sampling.
    This is more complex because E[X/Y] ≠ E[X]/E[Y]
    """
    # Approximate: E[|A ∩ B|] / E[|A ∪ B|]
    e_intersection = expected_intersection_with_replacement(N, n)
    # E[|A|] = n, E[|B|] = n
    # E[|A ∪ B|] ≈ E[|A|] + E[|B|] - E[|A ∩ B|] = 2n - e_intersection
    e_union = 2*n - e_intersection
    return e_intersection / e_union

def prob_intersection_k_with_replacement(N, n, k):
    """
    Probability that |A ∩ B| = k when A and B are sampled 
    COMPLETELY INDEPENDENTLY (with replacement / Bernoulli trials)
    
    Each element independently:
    - P(in A) = n/N
    - P(in B) = n/N
    - P(in both) = (n/N)²
    
    Therefore |A ∩ B| ~ Binomial(N, (n/N)²)
    
    Args:
        N: size of the universe X
        n: expected size of sets (parameter, not fixed size!)
        k: number of common elements
    
    Returns:
        P(|A ∩ B| = k)
    """
    if k < 0 or k > N:
        return 0.0
    
    p = (n / N) ** 2  # probability that element is in both A and B
    
    # Binomial distribution
    return binom.pmf(k, N, p)

def display_jaccard_indexes(df1, df2, set_length):
    jaccard_summary = PrettyTable()
    jaccard_summary.title = f'{BLUE}Jaccard Index for words most prone to misclassification{RESET}'
        
    # Colored headers
    headers = [
            f'{GREEN}X first words{RESET}',
            f'{CYAN}Common elements{RESET}',
            f'{YELLOW}Union of elements{RESET}',
            f'{RED}Jaccard Coef.{RESET}',
            f'{MAGENTA}Expected Jaccard{RESET}',
            f'{WHITE}Probability of the result{RESET}'
        ]
    jaccard_summary.field_names = headers
    for i in [10, 20, 30, 40, 50, 60, 100]:
        intersection = set(df1['Word'][0:i]) & set(df2['Word'][0:i])
        union = set(df1['Word'][0:i]).union(set(df2['Word'][0:i]))
        jaccard_coef = len(intersection) / len(union) if len(union) != 0 else 0
        expected_jaccard = expected_jaccard_with_replacement(set_length, i)
        
        # Colored values
        table = [[
            f"{GREEN}{i} first words{RESET}",
            f"{CYAN}{len(intersection)} words{RESET}",
            f"{YELLOW}{len(union)} words{RESET}",
            f"{RED}{jaccard_coef}{RESET}",
            f"{MAGENTA}{expected_jaccard}{RESET}",
            f"{WHITE}{prob_intersection_k_with_replacement(set_length, i, len(intersection)):.10f}{RESET}"
        ]]
        jaccard_summary.add_rows(table)
    print(jaccard_summary)