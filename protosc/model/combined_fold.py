from protosc.model.utils import select_features, compute_accuracy
from protosc.model.utils import compute_null_distribution
from protosc.model.wrapper import WrapperModel
from protosc.model.random import RandomModel
from protosc.model.pseudo_random import PseudoRandomModel
from protosc.model.final_selection import final_selection
from protosc.model.base import BaseFoldModel
from protosc.utils import Settings
from copy import deepcopy


class CombinedFoldModel(BaseFoldModel):
    def __init__(self, n_fold=None, settings=None):
        if settings is None:
            fast_wrapper_settings = WrapperModel().default_param
            fast_wrapper_settings.pop("max_features")
            fast_wrapper_settings.pop("n_fold")
            fast_wrapper_settings["greedy"] = True
            slow_wrapper_settings = WrapperModel().default_param
            slow_wrapper_settings["greedy"] = True
            slow_wrapper_settings.pop("max_features")
            slow_wrapper_settings.pop("n_fold")
            settings = Settings({
                "n_fold": n_fold,
                "fast_wrapper": Settings(fast_wrapper_settings),
                "slow_wrapper": Settings(slow_wrapper_settings),
            })
        self.n_fold = settings.n_fold
        self.settings = settings

    @property
    def default_param(self):
        return self.__class__().settings

    def _execute_fold(self, fold):
        output = {}
        selected_features, clusters = select_features(*fold[:2])

        # Filtermodel
        filter_accuracy = compute_accuracy(fold, selected_features)
        output['filter'] = {'features': selected_features,
                            'accuracy': filter_accuracy}

        # Wrapper fast
        fast_wrapper = WrapperModel(max_features=len(selected_features),
                                    **self.settings.fast_wrapper.todict())
        output['fast_wrapper'] = fast_wrapper._execute_fold(fold)

        # Wrapper slow
        slow_wrapper = WrapperModel(max_features=len(selected_features),
                                    **self.settings.slow_wrapper.todict())
        output['slow_wrapper'] = slow_wrapper._execute_fold(fold)

        # Random
        output['random'] = RandomModel.execute_with_clusters(
            fold, clusters, selected_features)

        # Pseudo random
        output['pseudo_random'] = PseudoRandomModel.execute_from_wrap_results(
            fold, clusters, selected_features,
            output['fast_wrapper']['features'])

        output["null_distribution"] = compute_null_distribution(output, fold)
        return output

    def _convert_interim(self, results):
        results = deepcopy(results)
        null_dist = [r.pop("null_distribution") for r in results]
        final_result = {}
        for model in list(results[0]):
            model_output = [res[model] for res in results]
            selection = final_selection(model_output, null_dist)
            final_result[model] = selection
        return final_result

    def add_null_distribution(self, _, __):
        pass
