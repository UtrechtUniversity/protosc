import numpy as np
from protosc.filter_model import train_xvalidate
from protosc.feature_matrix import FeatureMatrix


class Chromosome():
    def __init__(self, n_tot_features, features):
        self.n_tot_features = n_tot_features
        self.features = set(features)

    def __len__(self):
        return len(self.features)

    @classmethod
    def random(cls, n_tot_features, n_start_features):
        features = np.random.choice(n_tot_features, size=n_start_features, replace=False)
        return cls(n_tot_features, features)

    def fitness(self, X, y, n_test=1, **kwargs):
        return np.mean([self.fitness_once(X, y, **kwargs) for _ in range(n_test)])

    def fitness_once(self, X, y, num_penalty=0.01):
        if len(self.features) == 0:
            return 0.0
        accuracy = []
        for cur_fold in X.kfold(y, k=8):
            X_train, y_train, X_val, y_val = cur_fold
            new_acc = train_xvalidate(X_train[:, list(self.features)], y_train,
                                      X_val[:, list(self.features)], y_val)
            accuracy.append(new_acc)
        return np.mean(accuracy) - num_penalty*len(self.features)

    @classmethod
    def crossover(cls, chrom_1, chrom_2):
        common_features = chrom_1.features.intersection(chrom_2.features)
        chrom_1_uniq = chrom_1.features-common_features
        chrom_2_uniq = chrom_2.features-common_features
        chrom_3_features = common_features
        chrom_4_features = common_features
        select = np.random.choice(list(chrom_1_uniq), size=len(chrom_1_uniq)//2, replace=False)
        if isinstance(select, int):
            select = [select]
        chrom_3_features = chrom_3_features | set(select)
        chrom_4_features = chrom_4_features | (chrom_1.features - chrom_3_features)

        select = np.random.choice(list(chrom_2_uniq), size=(len(chrom_2_uniq)+1)//2, replace=False)
        if isinstance(select, int):
            select = [select]
        chrom_3_features = chrom_3_features | set(select)
        chrom_4_features = chrom_4_features | (chrom_2.features - chrom_3_features)
#         print(chrom_1.features, chrom_2.features)
#         print(common_features, chrom_1_uniq, chrom_2_uniq)
#         print(chrom_3_features, chrom_4_features)
#         print("-------------------------------------")
        return (cls(chrom_1.n_tot_features, chrom_3_features),
                cls(chrom_1.n_tot_features, chrom_3_features))

    def mutate(self, rate=0.05):
        n_mutate = int(self.n_tot_features*rate)
        if self.n_tot_features*rate-n_mutate > np.random.rand():
            n_mutate += 1

        for _ in range(n_mutate):
            if np.random.rand() < 0.5:
                self.mutate_add()
            else:
                self.mutate_remove()

    def mutate_add(self):
        choices = np.delete(np.arange(self.n_tot_features), list(self.features))
        if len(choices) == 0:
            return
        self.features = self.features | set(np.random.choice(choices, size=1))

    def mutate_remove(self):
        if len(self.features) == 0:
            return
        rem_id = np.random.choice(list(self.features))
        self.features = self.features - set([rem_id])

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

    def next_generation(self, k_tournament=5, truth=None):
        num_penalty = 0.005
        fitness = np.array([chrom.fitness(self.X, self.y, 2, num_penalty=num_penalty) for chrom in self.chromosomes])
        accuracy = [fitness[i] + num_penalty*len(self.chromosomes[i].features) for i in range(len(self))]
        self.test_extra_feature(fitness)
        if truth is not None:
            true_features = truth["selected_features"]
            counts = np.zeros(self.X.shape[1], dtype=int)
            above_average = np.argsort(fitness)
            for i_chrom in above_average:
                chrom = self.chromosomes[i_chrom]
                counts[list(chrom.features)] += 1
            for i_feature in range(self.X.shape[1]):
                self.counter[i_feature].append(counts[i_feature])
            fake_features = np.delete(np.arange(self.X.shape[1]), true_features)
            print(np.mean(counts[true_features]), np.mean(counts[fake_features]))
            print(np.sort(counts[true_features]))
            print(np.sort(counts[fake_features]))

        print(f"Avg fitness: {np.mean(accuracy)}, Max: {np.max(accuracy)}. stdev: {np.std(accuracy)}")
        print(f"Avg # of features: {np.mean([len(x) for x in self.chromosomes])}")

        new_chromo = []
        for _ in range(len(self)//2):
            parents = []
            while len(parents) < 2:
                possible_choices = np.random.choice(len(self), size=k_tournament, replace=False)
                new_parent = possible_choices[np.argmax(fitness[possible_choices])]
                if new_parent not in parents:
                    parents.append(new_parent)
            chrom_1, chrom_2 = self.chromosomes[parents[0]], self.chromosomes[parents[1]]
#             chrom_1, chrom_2 = np.random.choice(len(self), size=2, replace=False, p=rel_fitness)
            new_chromo.extend(Chromosome.crossover(chrom_1, chrom_2))

        for chrom in new_chromo:
            chrom.mutate(self.mutation_rate)
        self.chromosomes = new_chromo

    def test_extra_feature(self, fitness):
        chrom = self.chromosomes[np.argmax(fitness)]
        boot_fitness = [chrom.fitness(self.X, self.y, 1) for _ in range(100)]
        new_chrom = Chromosome(chrom.n_tot_features+1, chrom.features | set([chrom.n_tot_features]))
        new_boot_fitness = []
        for _ in range(100):
            new_X = FeatureMatrix(np.hstack((self.X.X, np.random.randn(self.X.shape[0]).reshape(-1, 1))))
            new_boot_fitness.append(new_chrom.fitness(new_X, self.y))
        dif_mean = np.mean(boot_fitness) - np.mean(new_boot_fitness)
        var_std = np.var(new_boot_fitness) - np.var(boot_fitness)
        print(dif_mean/np.sqrt(var_std))


def genetic_algorithm(X, y, n_gen=10, mutation_rate=0.05, truth=None):
    pop = Population(X, y, mutation_rate=mutation_rate)
    for _ in range(n_gen):
        pop.next_generation(truth=truth)
    return pop.counter
