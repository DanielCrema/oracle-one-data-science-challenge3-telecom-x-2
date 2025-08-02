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
import os
import sys
import contextlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old_stderr

def get_feature_permutation_importances(model: RandomForestClassifier | DecisionTreeClassifier | LogisticRegression, x: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    def generate_feature_permutation_importances_summary(feature_permutation_importances: dict['str', pd.DataFrame]) -> pd.DataFrame:
        summary = pd.DataFrame()
        first_key = list(feature_permutation_importances.keys())[0]

        summary['Feature'] = feature_permutation_importances[first_key]['Feature']
        summary.sort_values(by='Feature', ascending=False)

        for key, value in feature_permutation_importances.items():
            summary[key] = value.sort_values(by='Feature', ascending=False)['Mean Importance']
        
        return summary

    with suppress_stderr():
        perm_importance: dict[str, Bunch] = permutation_importance(
            model,
            pd.DataFrame(x),
            y,
            n_repeats=10,
            n_jobs=-1,
            # Use all scores
            scoring=['roc_auc', 'precision', 'recall', 'f1', 'accuracy']
        )

    feature_names = [feature.replace('onehotencoder__', '').replace('remainder__', '') for feature in x.columns] # type: ignore

    perm_importances: dict[str, pd.DataFrame] = {}
    for key, value in perm_importance.items():
        df = pd.DataFrame({
            'Feature': feature_names,
            'Mean Importance': value['importances_mean'],
            'Std Dev': value['importances_std']
        }).sort_values(by='Mean Importance', ascending=False)
        perm_importances[key] = df
    perm_importances['summary'] = generate_feature_permutation_importances_summary(perm_importances)

    return perm_importances