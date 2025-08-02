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
from pprint import pprint
from typing import Literal


class FeatureDropCandidates:
    def __init__(self, reinitialize: bool = False) -> None:
        self.candidates_to_drop: dict[str, set[str]] = {
            'random_forest': set(),
            'decision_tree': set(),
            'logistic_regression': set(),
            'all': set()
        }
        init_word = 'Initialized' if not reinitialize else 'Reinitialized'
        print(f"=> {init_word} FeatureDropCandidates object.")
        self._print_structure()

    def _print_structure(self) -> None:
        print('=> candidates_to_drop structure:\n')
        print(f'type = {type(self.candidates_to_drop)}')
        for key, value in self.candidates_to_drop.items():
            print(f'\t{key}: {value}')

    def get_candidates(self) -> dict[str, set[str]]:
        return self.candidates_to_drop

    def reset(self) -> None:
        self.__init__(reinitialize=True)

    def append(self, model: Literal['random_forest', 'decision_tree', 'logistic_regression'], data: list[str]) -> None:
        '''
        Appends a list of features to the candidates_to_drop dictionary for a given model.
        '''
        self.candidates_to_drop[model].update(data)

    def get_from_importances(self, model: Literal['random_forest', 'decision_tree', 'logistic_regression'], feature_importances: dict[str, pd.DataFrame], threshold: float = 0.01) -> None:
        '''
        Appends features with low importance (below or equal to threshold) from feature_importances to candidates_to_drop.
        '''
        threshold = abs(threshold)
        new_candidates = list(
            feature_importances[model][
                feature_importances[model]['Importance'].abs() <= threshold
            ]['Feature']
        )
        self.append(model=model, data=new_candidates)
    
    def get_from_permutation(
    self,
    feature_permutation_importances: dict[str, dict[str, pd.DataFrame]],
    model: Literal['random_forest', 'decision_tree', 'logistic_regression'],
    thresholds: dict[str, float]
) -> pd.DataFrame:
        '''
        Filters features from permutation importances that fall below defined thresholds and appends them to the drop list.
        '''
        candidates_df = feature_permutation_importances[model]['summary'][
            (feature_permutation_importances[model]['summary']['roc_auc'] <= thresholds['roc_auc']) &
            (feature_permutation_importances[model]['summary']['precision'] <= thresholds['precision']) &
            (feature_permutation_importances[model]['summary']['recall'] <= thresholds['recall']) &
            (feature_permutation_importances[model]['summary']['f1'] <= thresholds['f1']) &
            (feature_permutation_importances[model]['summary']['accuracy'] <= thresholds['accuracy'])
        ]

        candidates = list(candidates_df['Feature'])
        self.append(model, data=candidates)

        return candidates_df

    def generalize(self) -> pd.DataFrame:
        '''
        Builds a DataFrame combining all unique features marked for dropping and marks where each model included them.
        '''
        # Ensure 'all' is reset before combining
        self.candidates_to_drop['all'] = set()
        for model in ['random_forest', 'decision_tree', 'logistic_regression']:
            self.candidates_to_drop['all'].update(self.candidates_to_drop[model])

        all_features = self.candidates_to_drop['all']

        df = pd.DataFrame({
            'Feature': list(all_features),
            'random_forest': ['Yes' if f in self.candidates_to_drop['random_forest'] else 'No' for f in all_features],
            'decision_tree': ['Yes' if f in self.candidates_to_drop['decision_tree'] else 'No' for f in all_features],
            'logistic_regression': ['Yes' if f in self.candidates_to_drop['logistic_regression'] else 'No' for f in all_features],
            'best_candidates': [
                'Yes' if (
                    f in self.candidates_to_drop['random_forest'] and
                    f in self.candidates_to_drop['decision_tree'] and
                    f in self.candidates_to_drop['logistic_regression']
                ) else 'No' for f in all_features
            ]
        })

        return df