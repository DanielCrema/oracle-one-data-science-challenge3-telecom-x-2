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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils.plt_show_close import plt_show_close

def plot_correlation(df: pd.DataFrame, features: list[str]) -> None:
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt_show_close()