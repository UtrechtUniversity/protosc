from protosc.wrapper import Wrapper
from protosc import filter_model
import numpy as np


class Benchmark:
    def __init__(self, X, y, n_fold=8, fold_seed=213874):
        """
        Args:
            X: np.array, FeatureMatrix
                Feature matrix to wrap.
            y: np.array
                Outcomes, categorical (0/1).
            n_fold: int,
                number of folds you want to divide your X,y data over,
                i.e., number of model runs.
            fold_seed: int,
                seed.
        """
        self.X = X
        self.y = y
        self.n_fold = n_fold
        self.fold_seed = fold_seed

    def __run_models(self):
        """ Run (different) filter and wrapper models.
        Returns:
            output: dict,
                dictionary of model outputs.
        """
        # Run filter model
        output_filter = filter_model(self.X, self.y, n_fold=self.n_fold,
                                     fold_seed=self.fold_seed)
        # Run slow wrapper
        slow = Wrapper(self.X, self.y, n_fold=self.n_fold,
                       fold_seed=self.fold_seed)
        output_slow = slow.wrapper(n_jobs=-1)
        # Run fast wrapper
        fast = Wrapper(self.X, self.y, add_im=True, n_fold=self.n_fold,
                       fold_seed=self.fold_seed)
        output_fast = fast.wrapper(n_jobs=-1)
        # Combine output
        output = {'filter': output_filter, 'wrapper_slow': output_slow,
                  'wrapper_fast': output_fast}
        return output

    def __av_accuracy(self, accuracies):
        """ Calculate mean accuracy over all runs (n_folds).
        Args:
            accuracies: list,
                list of all accuracy scores (length = n_fold).
        Returns:
            Accuracy: float,
                average accuracy over all runs (n_folds).
        """
        av_accuracy = np.mean(accuracies)
        return av_accuracy

    def __fq_features(self, all_features):
        """ Determine feature frequencies over all runs (n_folds).
        Args:
            all_features: list,
                list of all occuring features over all runs.
        Returns:
            rec_features: dict,
                {feat:int} frequency per feature.
        """
        fq_features = {}
        for x in set(all_features):
            fq_features[x] = all_features.count(x)
        fq_features = dict(sorted(fq_features.items(),
                                  key=lambda item: item[1], reverse=True))
        return fq_features

    def __rec_features(self, all_features):
        """ Determine which features occur in each run.
        Args:
            all_features: list,
                list of all occuring features over all runs.
        Returns:
            rec_features: list,
                list of features occuring in each run.
        """
        rec_features = []
        for x in set(all_features):
            if all_features.count(x) == self.n_fold:
                rec_features.append(x)
        return sorted(rec_features)

    def __model_output(self, accuracies, all_features):
        """ Summarize output of model.
        Args:
            accuracies: list,
                list of all accuracy scores (length = n_fold).
            all_features: list,
                list of all occuring features over all runs.
        Returns:
            results:
                Accuracy: average accuracy over all runs (n_folds).
                Recurring features: features occuring in each run.
                Feature frequencies: feature frequencies over all runs.
        """
        # Calculate average accuracy score
        av_accuracy = self.__av_accuracy(accuracies)
        # Determine feature frequencies
        fq_features = self.__fq_features(all_features)
        # Find recurring features
        rec_features = self.__rec_features(all_features)
        # Add all findings to one dicitonary
        results = {'Accuracy': av_accuracy, 'Recurring features': rec_features,
                   'Feature frequencies': fq_features}
        return results

    def __output_filter(self, output):
        """ Summarize output of wrapper model.
        Args:
            output: dict,
                output of wrapper model.
        Returns:
            summary: dict,
                summary of model outputs (i.e., average accuracy,
                feature frequency, recurring features).
        """
        all_features = [f for feat in output for f in feat[0]]
        accuracies = [a[1] for a in output]
        summary = self.__model_output(accuracies, all_features)
        return summary

    def __output_wrapper(self, output):
        """ Summarize output of wrapper model.
        Args:
            output: dict,
                output of wrapper model.
        Returns:
            summary: dict,
                summary of model outputs (i.e., average accuracy,
                feature frequency, recurring features).
        """
        all_features = [f for feat in output['features'] for f in feat]
        accuracies = output['accuracy']
        summary = self.__model_output(accuracies, all_features)
        return summary

    def __compare_models(self, output):
        """ Summarize output of different models.
        Args:
            output: dict,
                {model:output} dictionary of model outputs.
        Returns:
            overview: dict,
                {model:summary} summary of model outputs.
        """
        overview = output
        for model, out in output.items():
            if model == 'filter':
                overview[model] = self.__output_filter(out)
            else:
                overview[model] = self.__output_wrapper(out)
        return overview

    def execute(self):
        # Run filter and (multiple) wrapper models
        output = self.__run_models()
        # Compare output of different models
        overview = self.__compare_models(output)
        return overview
