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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from plot_utils.plt_show_close import plt_show_close


def get_feature_importances(model: RandomForestClassifier | DecisionTreeClassifier | LogisticRegression, x_train: pd.DataFrame) -> pd.DataFrame:
    # Generate Dataframe
    feature_names = [feature.replace('onehotencoder__', '').replace('remainder__', '') for feature in x_train.columns] # type: ignore
    importances = model.coef_[0] if isinstance(model, LogisticRegression) else model.feature_importances_

    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Plot
    top_features = []
    for i in range(0, 5):
        feature = feature_importances.iloc[i]['Feature']
        if len(feature) > 17:
            feature = feature.replace('_', '\n')
        top_features.append(feature)
    top_features = pd.DataFrame({
        'Feature': top_features,
        'Importance': feature_importances.iloc[:5]['Importance']
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Feature', y='Importance', data=top_features,
        hue='Feature', dodge=False, palette='Set2', ax=ax
    )

    ax.set_title(f'Feature Importance - Top 5\n{type(model).__name__}', fontsize=16)
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Importance', fontsize=14)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2) # type: ignore

    plt.tight_layout()
    plt_show_close()

    return feature_importances