from protosc.model.utils import compute_accuracy
from protosc.model.base import BaseModel
from protosc.model.wrapper import Wrapper
from protosc.model.filter import compute_filter_fold


class PseudoRandomModel(BaseModel):
    def __init__(self, *args, **kwargs):
        self.wrapper_args = args
        self.wrapper_kwargs = kwargs
        self.n_fold = Wrapper(*args, **kwargs).n_fold

#     def execute(self, X, y, *args, fold_seed=None, seed=None, **kwargs):
#         filter_data = compute_filter_data(X, y, self.n_fold, fold_seed, seed)
#         wrapper = Wrapper(*self.wrapper_args, **self.wrapper_kwargs)
#         wrapper_results = wrapper.execute(X, y, *args, fold_seed=fold_seed,
#                                           seed=seed, **kwargs)
#         pseudo_results = []
#         for i_fold in len(wrapper_results):
#             cur_res = self.execute_from_wrap_results(
#                 **filter_data[i_fold],
#                 wrapper_features=wrapper_results[i_fold])
#             pseudo_results.append(cur_res)
#         return pseudo_results

    def _execute_fold(self, fold, *args, **kwargs):
        filter_data = compute_filter_fold(fold)
        wrapper = Wrapper(*self.wrapper_args, **self.wrapper_kwargs)
        wrapper_results = wrapper._execute_fold(fold, *args, **kwargs)
        return self.execute_from_wrap_results(
            **filter_data,
            wrapper_features=wrapper_results)

    @staticmethod
    def execute_from_wrap_results(cur_fold, clusters, selected_features,
                                  wrapper_features):
        # Pseudo-random
        pseudo_selection = []
        for cluster in clusters:
            if len(pseudo_selection) >= len(selected_features):
                break
            for feat in cluster:
                if (feat not in selected_features and
                        feat not in wrapper_features):
                    pseudo_selection.append(feat)
        accuracy = compute_accuracy(cur_fold, pseudo_selection)
        return {'features': pseudo_selection, 'accuracy': accuracy}
