# # #
# بِسْمِ ٱللّٰهِ ٱلرَّحْمٰنِ ٱلرَّحِيمِ
# Bismillāh ir-raḥmān ir-raḥīm
# 
# In the name of God, the Most Gracious, the Most Merciful
# Em nome de Deus, o Clemente, o Misericordioso
# # #
# # #


# #
# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import plt_show_close, plot_central_tendency, label_plot

def explore_distribution(df: pd.DataFrame, feature: str, bins="auto") -> None:
    """
    Explore and visualize the distribution of a specified feature in a DataFrame.

    This function prints the value counts of the specified feature and creates
    three types of plots to visualize its distribution: a horizontal bar chart,
    a box plot, and a histogram.

    Parameters
    ----------
    df (pd.DataFrame):
        The DataFrame containing the data.
    feature (str):
        The name of the feature/column to explore.
    bins (int):
        The number of bins to use for the histogram. Default is "auto".

    Returns
    ----------
    None
    """
    # Print the value counts of the specified feature
    print(df[feature].value_counts())
    print('-' * 30)

    # Plot the value counts as a horizontal bar chart
    plt.barh(df[feature].value_counts().index, list(df[feature].value_counts().values))
    label_plot(title=f'Distribution of {feature}', ylabel=feature, xlabel='Count')
    plt_show_close()

    # Plot the distribution as a box plot
    sns.boxplot(list(df[feature]))
    label_plot(title=f'Boxplot of {feature}', ylabel=feature)
    plt_show_close()

    # Plot the distribution as a histogram
    sns.histplot(list(df[feature]), bins=bins, kde=True)
    if df[feature].dtype == 'int64' or df[feature].dtype == 'float64':
        # Plot the mean, median, and mode lines
        plot_central_tendency(df[feature], linewidth=2, colors=['red', 'yellow', 'blue'])
    label_plot(title=f'Histogram of {feature}', ylabel='Count', xlabel=feature)
    plt_show_close()