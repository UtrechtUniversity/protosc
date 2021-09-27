import numpy as np

from protosc.model.utils import compute_accuracy
from protosc.model.base import BaseModel
from protosc.model.filter import compute_filter_fold


class RandomModel(BaseModel):
    def _execute_fold(self, fold):
        filter_data = compute_filter_fold(fold)
        return self.execute_with_clusters(**filter_data)

    @staticmethod
    def execute_with_clusters(cur_fold, clusters, selected_features):
        # Random
        np.random.shuffle(clusters)

        random_selection = []
        for cluster in clusters:
            if len(random_selection) >= len(selected_features):
                break
            random_selection.extend(cluster)
        accuracy = compute_accuracy(cur_fold, random_selection)
        return {'features': random_selection, 'accuracy': accuracy}
