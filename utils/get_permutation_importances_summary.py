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

def get_permutation_importances_summary(feature_permutation_importances: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    feature_importances_summary = feature_permutation_importances['random_forest']['recall'].drop(columns=['Std Dev']).sort_values('Feature', ascending=False)
    feature_importances_summary = feature_importances_summary.merge(feature_permutation_importances['decision_tree']['recall'].drop(columns=['Std Dev']), how='outer', on='Feature')
    feature_importances_summary = feature_importances_summary.merge(feature_permutation_importances['logistic_regression']['recall'].drop(columns=['Std Dev']), how='outer', on='Feature')
    feature_importances_summary = feature_importances_summary.rename(columns={'Mean Importance_x': 'random_forest', 'Mean Importance_y': 'decision_tree', 'Mean Importance': 'logistic_regression'})

    # Adding average
    feature_importances_summary['average'] = feature_importances_summary[['random_forest', 'decision_tree', 'logistic_regression']].mean(axis=1)

    # Sorting by absolute average
    feature_importances_summary['abs_average'] = feature_importances_summary['average'].abs()
    feature_importances_summary = feature_importances_summary.sort_values('abs_average', ascending=False)
    feature_importances_summary.drop(columns=['abs_average'], inplace=True)

    return feature_importances_summary