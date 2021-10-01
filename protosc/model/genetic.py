from math import ceil

import numpy as np
from scipy import stats
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from protosc.model.utils import compute_accuracy
from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel


class GeneticModel():
    def __init__(self, n_chromo=100, mutation_rate=0.1, k_tournament=5,
                 num_penalty=0.005, n_gen_data=3, n_random_features=100,
                 signif_criterion=0.5):
        self.n_chromo = n_chromo
        self.mutation_rate = mutation_rate
        self.k_tournament = k_tournament
        self.num_penalty = num_penalty
        self.n_gen_data = n_gen_data
        self.n_random_features = n_random_features
        self.signif_criterion = signif_criterion

    def execute(self, X, y, seed=None, n_jobs=-1, progress_bar=True):
        # Make a copy of the feature matrix so we can add random columns.
        X_copy = X.copy()
        X_copy.add_random_columns(self.n_random_features)

        # Initialize the population and chromosomes.
        pop = Population(
            X_copy, y, n_chromo=self.n_chromo,
            mutation_rate=self.mutation_rate, k_tournament=self.k_tournament,
            num_penalty=self.num_penalty, n_jobs=n_jobs)

        # Perform the generations.
        n_gen = ceil(self.n_gen_data*X_copy.shape[1]/len(pop))
        results = []
        with tqdm(total=n_gen) as pbar:
            for _ in range(n_gen):
                results.extend(pop.next_generation(pbar=pbar))

        # Initialize and fill the result matrix and outcomes.
        arr_results = np.zeros((len(results), X_copy.shape[1]))
        i_row = 0
        y = np.zeros(len(results))
        for res, acc in results:
            arr_results[i_row, res] = 1
            y[i_row] = acc
            i_row += 1

        coefs = compute_coefs(arr_results, y, self.n_random_features)
        features = compute_significant_features(
            coefs, self.n_random_features, self.signif_criterion)
        return features


class Chromosome():
    def __init__(self, n_tot_features, features):
        """Chromosome for feature selection

        Arguments
        ---------
        n_tot_features: int
            Number of total features that can be selected.
        features: list[int]
            List of feature id's that are selected.
        """
        self.n_tot_features = n_tot_features
        self.features = set(features)

    def __len__(self):
        return len(self.features)

    @classmethod
    def random(cls, n_tot_features, n_start_features):
        """Initialize a chromosome with random features.

        Arguments
        ---------
        n_tot_features: int
            Number of total features that can be selected.
        n_start_features: int
            Number of features that the chromosome is initialized with.
        """
        features = np.random.choice(n_tot_features, size=n_start_features,
                                    replace=False)
        return cls(n_tot_features, features)

    def accuracy(self, X, y, n_test=1, **kwargs):
        """Compute the accuracy of the ML model of the chromosome.

        Arguments
        ---------
        X: FeatureMatrix
            Feature matrix with all features
        y: np.ndarray
            Outcomes for all samples.
        n_test: int
            Number of repeat tests for computing the accuracy.

        Returns
        -------
        accuracy: float
            The average accuracy of all the runs.
        """
        return np.mean([self._accuracy_once(X, y, **kwargs)
                        for _ in range(n_test)])

    def _accuracy_once(self, X, y):
        if len(self.features) == 0:
            return 0.0
        accuracy = []
        for cur_fold in X.kfold(y, k=8):
            accuracy.append(compute_accuracy(cur_fold, self.features))
        return np.mean(accuracy)

    @classmethod
    def crossover(cls, chrom_1, chrom_2):
        """Perform a crossover procedure between two chromosomes.

        Arguments
        ---------
        chrom_1: Chromosome
            First chromome to perform crossover with.
        chrom_2: Chromosome
            Second chromosome to perform crossover with.

        Returns
        -------
        x_chromosomes: tuple[Chromosome]
            Two resulting chromosomes that are the crossovers of
            their parents.
        """
        # Features that the chromosomes have in common are always inherited.
        common_features = chrom_1.features.intersection(chrom_2.features)

        # Find the chromosomes that are unique to either chromosome.
        chrom_1_uniq = chrom_1.features-common_features
        chrom_2_uniq = chrom_2.features-common_features
        chrom_3_features = common_features
        chrom_4_features = common_features

        # Split the features that are unique to chromosome 1 in two.
        select = np.random.choice(list(chrom_1_uniq),
                                  size=len(chrom_1_uniq)//2, replace=False)
        if isinstance(select, int):
            select = [select]

        # Assign half to the first child and the other to the second.
        chrom_3_features = chrom_3_features | set(select)
        chrom_4_features = chrom_4_features | (chrom_1.features
                                               - chrom_3_features)

        # Do the same for the unique features of chromosome 2.
        select = np.random.choice(list(chrom_2_uniq),
                                  size=(len(chrom_2_uniq)+1)//2, replace=False)
        if isinstance(select, int):
            select = [select]

        chrom_3_features = chrom_3_features | set(select)
        chrom_4_features = chrom_4_features | (chrom_2.features
                                               - chrom_3_features)

        # Return the two children.
        return (cls(chrom_1.n_tot_features, chrom_3_features),
                cls(chrom_1.n_tot_features, chrom_4_features))

    def mutate(self, rate=0.05):
        """Perform a set of mutations depending on the chromosome.

        There are three kinds of mutations:
            - Split off 1/3 of the chromosome (10% chance)
            - Add a new random feature (chance depends on #features)
            - Remove a feature (chance depends on # features)

        Arguments
        ---------
        rate: float
            The rate determines the number of mutations that will take place.
        """

        # The number of mutations depends on the number of features.
        n_mutate = int(len(self)*rate)
        if len(self)*rate-n_mutate > np.random.rand():
            n_mutate += 1

        for _ in range(n_mutate):
            rand_val = np.random.rand()
            rand_rest = (rand_val-0.1)/0.9
            if rand_val < 0.1:
                self.mutate_split()
            elif rand_rest > len(self)/self.n_tot_features:
                self.mutate_add()
            else:
                self.mutate_remove()

    def mutate_add(self):
        """Add a random new feature."""
        choices = np.delete(np.arange(self.n_tot_features),
                            list(self.features))
        if len(choices) == 0:
            return
        self.features = self.features | set(np.random.choice(choices, size=1))

    def mutate_remove(self):
        """Remove a random feature."""
        if len(self.features) <= 1:
            return
        rem_id = np.random.choice(list(self.features))
        self.features = self.features - set([rem_id])

    def mutate_split(self):
        """Split of 1/3 of the number of features."""
        if len(self.features) < 2:
            return
        n_remove = len(self.features)//3
        if n_remove <= 1:
            self.mutate_remove()
            return
        rem_id = np.random.choice(list(self.features),
                                  size=n_remove, replace=False)
        self.features = self.features - set(rem_id)

    def __str__(self):
        return str(list(self.features))


class Population():
    def __init__(self, X, y, n_chromo=100, mutation_rate=0.01, k_tournament=5,
                 num_penalty=0.005, n_jobs=-1):
        """Population of chromosome for feature selection procedures.

        Arguments
        ---------
        X: FeatureMatrix
            Feature matrix to compute the feature matrix for.
        y: np.ndarray[int]
            Outcomes.
        n_chromo: int
            Number of chromosomes to populate (should be even).
        mutation_rate: float
            Rate at which the chromosomes mutate.
        k_tournament: int
            Number of participant in the tournament selection procedure.
        num_penalty: float
            Penalty for having more features selected.
        """
        n_tot_features = X.shape[1]
        n_start_features = min(max(5, n_tot_features // 10), n_tot_features-1)
        self.chromosomes = [Chromosome.random(n_tot_features, n_start_features)
                            for _ in range(n_chromo)]
        self.X = X
        self.y = y
        self.mutation_rate = mutation_rate
        self.k_tournament = k_tournament
        self.num_penalty = num_penalty
        self.n_jobs = n_jobs

    def __len__(self):
        return len(self.chromosomes)

    def parallel_accuracy(self):
        """Compute the accuracy of all chromosomes in parallel."""
        jobs = [{"chrom": chrom} for chrom in self.chromosomes]
        return np.array(execute_parallel(
            jobs, _compute_parallel_accuracy,
            n_jobs=self.n_jobs,
            args=(self.X, self.y)))

    def next_generation(self, pbar=None):
        """Evolve the chromosomes to the next generation (destructively)"""
        accuracy = self.parallel_accuracy()

        # Compute the fitness from accuracy and the number of features.
        num_features = np.array([len(chrom) for chrom in self.chromosomes])
#         nz = accuracy > 0
#         fitness = np.zeros_like(accuracy)
#         fitness[nz] = (-np.log(1/accuracy[nz]-1))**2
#         fitness[nz] -= self.num_penalty*num_features[nz]
        fitness = accuracy - self.num_penalty*num_features

        # Gather all information from the current generation.
        results = self.current_results(accuracy)

        # Perform crossovers until the new generation is formed.
        new_chromo = []
        for _ in range(len(self)//2):
            # Select two parents that are the same.
            parents = []
            while len(parents) < 2:
                possible_choices = np.random.choice(
                    len(self), size=self.k_tournament, replace=False)
                new_parent = possible_choices[
                    np.argmax(fitness[possible_choices])]
                if new_parent not in parents:
                    parents.append(new_parent)
            chrom_1 = self.chromosomes[parents[0]]
            chrom_2 = self.chromosomes[parents[1]]
            new_chromo.extend(Chromosome.crossover(chrom_1, chrom_2))

        # Mutate all the chromosomes.
        for chrom in new_chromo:
            chrom.mutate(self.mutation_rate)
        self.chromosomes = new_chromo

        # Give feedback on the current status, either with tqdm or text.
        mean_features = np.mean([len(x) for x in self.chromosomes])
        if pbar is None:
            print(f"Avg fitness: {np.mean(accuracy)}, Max: {np.max(accuracy)}."
                  f" stdev: {np.std(accuracy)}")
            print(f"Avg # of features: {mean_features}")
        else:
            pbar.set_description(f"Acc: {np.mean(accuracy):.2f} "
                                 f"(max: {np.max(accuracy):.2f}, "
                                 f"#f: {mean_features:.0f})")
            pbar.update(1)
        return results

    def _test_extra_feature(self, fitness):
        """Currently unused, but can be used to test the feature penalty."""
        chrom = self.chromosomes[np.argmax(fitness)]
        boot_fitness = [chrom.fitness(self.X, self.y, 1) for _ in range(100)]
        new_chrom = Chromosome(chrom.n_tot_features+1,
                               chrom.features | set([chrom.n_tot_features]))
        new_boot_fitness = []
        for _ in range(100):
            new_X = FeatureMatrix(
                np.hstack((self.X.X,
                           np.random.randn(self.X.shape[0]).reshape(-1, 1))))
            new_boot_fitness.append(new_chrom.fitness(new_X, self.y))
        dif_mean = np.mean(boot_fitness) - np.mean(new_boot_fitness)
        var_std = np.var(new_boot_fitness) - np.var(boot_fitness)
        print(dif_mean/np.sqrt(var_std))

    def current_results(self, accuracy):
        """Gather all the accuracy > 0 results of this generation."""
        return [(list(self.chromosomes[i].features), accuracy[i])
                for i in range(len(self)) if accuracy[i] > 0]


def genetic_algorithm(X, y, *args, n_data=3, n_random=100, **kwargs):
    """Apply the genetic algorithm to the feature matrix.

    Arguments
    ---------
    X: FeatureMatrix
        Feature matrix to be tested/selected from.
    y: np.ndarray[int]
        Outcomes of the samples.
    n_data: float
        Amount of data that needs to be generated. The number of generations
        is directly dependent on this number. It is chosen such that:
        #chromosomes * #generations >= #features.
        It is advised to put this above 1, larger numbers result in a
        performance penalty though.
    n_random: int
        Number of random columns to add. This is used for finding an
        appropriate cutoff point for the features. At least 100 are advised.
        While more is generally better, there are diminishing and eventually
        decreasing returns.

    Returns
    -------
    gen_X, gen_y: (np.ndarray[float, float], np.ndarray[float])
        A new feature matrix + outcome that can be used to predict the
        outcome as a function of the included features.
    """
    # Make a copy of the feature matrix so we can add random columns.
    X_copy = X.copy()
    X_copy.add_random_columns(n_random)

    # Initialize the population and chromosomes.
    pop = Population(X_copy, y, *args, **kwargs)

    # Perform the generations.
    n_gen = ceil(n_data*X_copy.shape[1]/len(pop))
    results = []
    with tqdm(total=n_gen) as pbar:
        for _ in range(n_gen):
            results.extend(pop.next_generation(pbar=pbar))

    # Initialize and fill the result matrix and outcomes.
    arr_results = np.zeros((len(results), X_copy.shape[1]))
    i_row = 0
    y = np.zeros(len(results))
    for res, acc in results:
        arr_results[i_row, res] = 1
        y[i_row] = acc
        i_row += 1
    return arr_results, y


def _compute_parallel_accuracy(X, y, chrom, n_compute=2):
    return chrom.accuracy(X, y, n_compute)


def compute_coefs(X_gen, y_gen, n_random=100):
    """Compute coefficients for the genetic feature matrix.

    The ElasticNet regularization parameter alpha is adjusted so that
    approximately half of the random features have their coefficients
    set to zero.
    """
    alpha = 0.0002
    fac = 2
#     y_tilde = (-np.log(1/y_gen-1))**2
    y_tilde = y_gen
    random_features = np.arange(X_gen.shape[1]-n_random, X_gen.shape[1])

    # Set the boundaries for the number of random features with non-zero coefs.
    min_nz = round(n_random*0.45)
    max_nz = round(n_random*0.6)
    last_dir = -1

    # Do a maximum of 100 loops before giving up.
    for _ in range(100):
        model = ElasticNet(alpha=alpha)
        model.fit(X_gen, y_tilde)
        n_random_nz = np.sum(model.coef_[random_features] != 0)

        # Decrease the step size if we're going back in the other direction.
        if n_random_nz > max_nz:
            if last_dir == 0:
                fac /= 2
            last_dir = 1
            alpha *= fac
        elif n_random_nz < min_nz:
            if last_dir == 1:
                fac /= 2
            last_dir = 0
            alpha /= fac
        else:
            break
    return model.coef_


def compute_significant_features(coefs, n_random=100, sign_criterion=0.5):
    """Find a selection from the coefficients of the features.

    The random features are used to determine which features are significant.

    Arguments
    ---------
    coefs: np.ndarray[float]
        Coefficients computed for the features.
    n_random: int
        Number of random features appended at the end.
    sign_criterion: float
        The expected number of false positives.

    Returns
    -------
    features: np.ndarray[int]
        The significant features.
    """
    n_features = len(coefs)
    test_features = np.arange(0, n_features-n_random)
    random_features = np.arange(n_features-n_random, n_features)

    rand_nz = coefs[random_features]
    rand_nz = rand_nz[rand_nz != 0]
    sd_random = np.std(rand_nz)
    # mn_random = min(0, np.mean(rand_nz))
    mn_random = 0

    # Assume the non-zero coefficients are normally distributed.
    num_sd = -stats.norm.ppf(sign_criterion/len(test_features))
    limit = mn_random + num_sd*sd_random
    return np.where(coefs[test_features] > limit)[0]
