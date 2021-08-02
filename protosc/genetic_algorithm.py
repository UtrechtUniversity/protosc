from math import ceil

import numpy as np
from scipy import stats
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from protosc.filter_model import train_xvalidate
from protosc.feature_matrix import FeatureMatrix
from protosc.parallel import execute_parallel


class Chromosome():
    def __init__(self, n_tot_features, features):
        self.n_tot_features = n_tot_features
        self.features = set(features)

    def __len__(self):
        return len(self.features)

    @classmethod
    def random(cls, n_tot_features, n_start_features):
        features = np.random.choice(n_tot_features, size=n_start_features,
                                    replace=False)
        return cls(n_tot_features, features)

    def accuracy(self, X, y, n_test=1, **kwargs):
        return np.mean([self.accuracy_once(X, y, **kwargs)
                        for _ in range(n_test)])

    def accuracy_once(self, X, y):
        if len(self.features) == 0:
            return 0.0
        accuracy = []
        for cur_fold in X.kfold(y, k=8):
            X_train, y_train, X_val, y_val = cur_fold
            new_acc = train_xvalidate(X_train[:, list(self.features)], y_train,
                                      X_val[:, list(self.features)], y_val)
            accuracy.append(new_acc)
        return np.mean(accuracy)

    @classmethod
    def crossover(cls, chrom_1, chrom_2):
        common_features = chrom_1.features.intersection(chrom_2.features)
        chrom_1_uniq = chrom_1.features-common_features
        chrom_2_uniq = chrom_2.features-common_features
        chrom_3_features = common_features
        chrom_4_features = common_features
        select = np.random.choice(list(chrom_1_uniq),
                                  size=len(chrom_1_uniq)//2, replace=False)
        if isinstance(select, int):
            select = [select]
        chrom_3_features = chrom_3_features | set(select)
        chrom_4_features = chrom_4_features | (chrom_1.features
                                               - chrom_3_features)

        select = np.random.choice(list(chrom_2_uniq),
                                  size=(len(chrom_2_uniq)+1)//2, replace=False)
        if isinstance(select, int):
            select = [select]
        chrom_3_features = chrom_3_features | set(select)
        chrom_4_features = chrom_4_features | (chrom_2.features
                                               - chrom_3_features)
        return (cls(chrom_1.n_tot_features, chrom_3_features),
                cls(chrom_1.n_tot_features, chrom_3_features))

    def mutate(self, rate=0.05):
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
        choices = np.delete(np.arange(self.n_tot_features),
                            list(self.features))
        if len(choices) == 0:
            return
        self.features = self.features | set(np.random.choice(choices, size=1))

    def mutate_remove(self):
        if len(self.features) == 0:
            return
        rem_id = np.random.choice(list(self.features))
        self.features = self.features - set([rem_id])

    def mutate_split(self):
        if len(self.features) < 2:
            return
        n_remove = len(self.features)//3
        if n_remove <= 1:
            self.mutate_remove()
        rem_id = np.random.choice(list(self.features),
                                  size=n_remove, replace=False)
        self.features = self.features - set(rem_id)

    def __str__(self):
        return str(list(self.features))


class Population():
    def __init__(self, X, y, n_chromo=100, mutation_rate=0.01):
        n_tot_features = X.shape[1]
        n_start_features = n_tot_features // 10
        self.chromosomes = [Chromosome.random(n_tot_features, n_start_features)
                            for _ in range(n_chromo)]
        self.X = X
        self.y = y
        self.mutation_rate = mutation_rate
        self.counter = [[] for _ in range(n_tot_features)]

    def __len__(self):
        return len(self.chromosomes)

    def all_accuracy(self):
        def compute_parallel_accuracy(X, y, chrom, n_compute=2):
            return chrom.accuracy(X, y, n_compute)

        jobs = [{"chrom": chrom} for chrom in self.chromosomes]
        return np.array(execute_parallel(jobs, compute_parallel_accuracy,
                                         args=(self.X, self.y)))

    def next_generation(self, k_tournament=5, pbar=None):
        num_penalty = 0.005
        accuracy = self.all_accuracy()
        num_features = np.array([len(chrom) for chrom in self.chromosomes])
        nz = accuracy > 0
        fitness = np.zeros_like(accuracy)
        fitness[nz] = (-np.log(1/accuracy[nz]-1))**2
        fitness[nz] -= num_penalty*num_features[nz]

        results = self.current_results(accuracy)
        new_chromo = []
        for _ in range(len(self)//2):
            parents = []
            while len(parents) < 2:
                possible_choices = np.random.choice(
                    len(self), size=k_tournament, replace=False)
                new_parent = possible_choices[
                    np.argmax(fitness[possible_choices])]
                if new_parent not in parents:
                    parents.append(new_parent)
            chrom_1 = self.chromosomes[parents[0]]
            chrom_2 = self.chromosomes[parents[1]]
            new_chromo.extend(Chromosome.crossover(chrom_1, chrom_2))

        for chrom in new_chromo:
            chrom.mutate(self.mutation_rate)
        self.chromosomes = new_chromo
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

    def test_extra_feature(self, fitness):
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
        return [(list(self.chromosomes[i].features), accuracy[i])
                for i in range(len(self)) if accuracy[i] > 0]


def genetic_algorithm(X, y, n_gen=None, n_random=100, mutation_rate=0.05):
    X_copy = X.copy()
    X_copy.add_random_columns(n_random)
    pop = Population(X_copy, y, mutation_rate=mutation_rate)

    if n_gen is None:
        n_gen = ceil(3*X_copy.shape[1]/len(pop))
    results = []
    with tqdm(total=n_gen) as pbar:
        for _ in range(n_gen):
            results.extend(pop.next_generation(pbar=pbar))

    arr_results = np.zeros((len(results), X_copy.shape[1]))
    i_row = 0
    y = np.zeros(len(results))
    for res, acc in results:
        arr_results[i_row, res] = 1
        y[i_row] = acc
        i_row += 1
    return arr_results, y


def compute_coefs(X_gen, y_gen, n_random=100):
    alpha = 0.0002
    fac = 2
    y_tilde = (-np.log(1/y_gen-1))**2
    random_features = np.arange(X_gen.shape[1]-n_random, X_gen.shape[1])

    min_nz = round(n_random*0.45)
    max_nz = round(n_random*0.6)
    last_dir = -1
    for _ in range(100):
        model = ElasticNet(alpha=alpha)
        model.fit(X_gen, y_tilde)
        n_random_nz = np.sum(model.coef_[random_features] != 0)

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


def compute_significant_features(coefs, n_random=100, sign_critereon=0.5):
    n_features = len(coefs)
    test_features = np.arange(0, n_features-n_random)
    random_features = np.arange(n_features-n_random, n_features)

    rand_nz = coefs[random_features]
    rand_nz = rand_nz[rand_nz != 0]
    sd_random = np.std(rand_nz)
    # mn_random = min(0, np.mean(rand_nz))
    mn_random = 0

    num_sd = -stats.norm.ppf(sign_critereon/len(test_features))
    limit = mn_random + num_sd*sd_random
    return np.where(coefs[test_features] > limit)[0]
