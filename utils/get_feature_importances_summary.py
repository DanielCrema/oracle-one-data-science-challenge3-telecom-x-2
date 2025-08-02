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

def get_feature_importances_summary(feature_importances: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Merging
    feature_importances_summary = feature_importances['random_forest'].sort_values('Feature')
    feature_importances_summary = feature_importances_summary.merge(feature_importances['decision_tree'], how='outer', on='Feature')
    feature_importances_summary = feature_importances_summary.merge(feature_importances['logistic_regression'], how='outer', on='Feature')
    feature_importances_summary = feature_importances_summary.rename(columns={'Importance_x': 'random_forest', 'Importance_y': 'decision_tree', 'Importance': 'logistic_regression'})

    # Adding average
    feature_importances_summary['average'] = feature_importances_summary[['random_forest', 'decision_tree', 'logistic_regression']].mean(axis=1)

    # Sorting by absolute average
    feature_importances_summary['abs_average'] = feature_importances_summary['average'].abs()
    feature_importances_summary = feature_importances_summary.sort_values('abs_average', ascending=False)
    feature_importances_summary.drop(columns=['abs_average'], inplace=True)

    return feature_importances_summary