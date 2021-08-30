from protosc.wrapper import Wrapper
from protosc import filter_model
import numpy as np


class Benchmark:
    def __init__(self, X, y, n, n_fold=8, fold_seed=213874):
        """
        Args:
            n: int,
                number of times you want to run the Benchmark
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
        self.n = n
        self.X = X
        self.y = y
        self.n_fold = n_fold
        self.fold_seed = fold_seed

    def __run_models(self, fold_seed):
        """ Run (different) filter and wrapper models.
        Returns:
            output: dict,
                dictionary of model outputs.
        """
        # Run filter model
        print("Running filter_model...")
        output_filter = filter_model(self.X, self.y, n_fold=self.n_fold,
                                     fold_seed=fold_seed)
        # Run slow wrapper
        print("Running 'slow' wrapper...")
        slow = Wrapper(self.X, self.y, n_fold=self.n_fold,
                       fold_seed=fold_seed)
        output_slow = slow.wrapper(n_jobs=-1)
        # Run fast wrapper
        print("Running 'fast' wrapper...")
        fast = Wrapper(self.X, self.y, add_im=True, n_fold=self.n_fold,
                       fold_seed=fold_seed)
        output_fast = fast.wrapper(n_jobs=-1)
        # Combine output
        output = {'filter': output_filter, 'wrapper_slow': output_slow,
                  'wrapper_fast': output_fast}
        return output

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

    def __model_output(self, accuracies, all_features, features):
        """ Summarize output of model.
        Args:
            accuracies: list,
                list of all accuracy scores (length = n_fold).
            all_features: list,
                list of all occuring features over all runs.
        Returns:
            results:
                Accuracy: average accuracy over all runs (i.e., n_folds).
                Unique features: total number of unique features over all runs.
                Mean features: mean number of features per run.
                Recurring features: list of features occuring in each run.
                Feature frequencies: feature frequencies over all runs.
        """
        # Calculate average accuracy score
        av_accuracy = np.mean(accuracies)
        # Count number of unique features
        un_features = len(set(all_features))
        # Average number of features per run (n_fold)
        av_features = round(np.mean(features))
        # Determine feature frequencies
        fq_features = self.__fq_features(all_features)
        # Find recurring features
        rec_features = self.__rec_features(all_features)
        # Add all findings to one dicitonary
        results = {'Mean accuracy': av_accuracy,
                   'Unique features': un_features,
                   'Mean features': av_features,
                   'Recurring features': rec_features,
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
                number of unique features, feature frequency,
                recurring features).
        """
        all_features = [f for feat in output for f in feat[0]]
        accuracies = [a[1] for a in output]
        features = [len(f[0]) for f in output]
        summary = self.__model_output(accuracies, all_features, features)
        return summary

    def __output_wrapper(self, output):
        """ Summarize output of wrapper model.
        Args:
            output: dict,
                output of wrapper model.
        Returns:
            summary: dict,
                summary of model outputs (i.e., average accuracy,
                number of unique features, feature frequency,
                recurring features).
        """
        all_features = [f for feat in output['features'] for f in feat]
        accuracies = output['accuracy']
        features = [len(f) for f in output['features']]
        summary = self.__model_output(accuracies, all_features, features)
        return summary

    def __compare_models(self, output):
        """ Summarize output of different models.
        Args:
            output: dict,
                {model:output} dictionary of model and corresponding outputs.
        Returns:
            overview: dict,
                {model:summary} summary of all models.
        """
        overview = output
        for model, out in output.items():
            if model == 'filter':
                overview[model] = self.__output_filter(out)
            else:
                overview[model] = self.__output_wrapper(out)
        return overview

    def execute(self):
        overview = []
        np.random.seed(self.fold_seed)
        for round in range(self.n):
            print(f"--- Round {round+1}/{self.n} ---")
            # Run filter and (multiple) wrapper models
            output = self.__run_models(
                fold_seed=np.random.randint(self.fold_seed))
            # Compare output of different models
            overview.append(self.__compare_models(output))
        return overview
