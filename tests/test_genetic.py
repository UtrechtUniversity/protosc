from copy import deepcopy

import numpy as np
from pytest import mark

from protosc.simulation import create_simulation_data
from protosc.model.genetic import genetic_algorithm
from protosc.model.genetic import compute_coefs
from protosc.model.genetic import Chromosome, compute_significant_features


@mark.parametrize(
    "n_examples,n_features", [
        (200, 20),
        (40, 2),
        (139, 5),
        (200, 7)
    ]
)
def test_chromosome(n_examples, n_features):
    n_true_features = n_features // 2
    X, y, _ = create_simulation_data(n_examples=n_examples,
                                     n_features=n_features,
                                     n_true_features=n_true_features)
    for _ in range(5):
        chrom = Chromosome.random(n_features, n_features//2)
        acc = chrom.accuracy(X, y, n_test=10)
        assert acc >= 0 and acc <= 1
        chrom2 = Chromosome.random(n_features, n_features//2)
        chrom3, chrom4 = Chromosome.crossover(chrom, chrom2)
        assert len(chrom3)+len(chrom4) == len(chrom) + len(chrom2)
        assert (chrom3.features | chrom4.features) == (chrom.features | chrom2.features)
        old_features = deepcopy(chrom.features)
        chrom.mutate_remove()
        assert len(chrom) == max(1, len(old_features)-1)
        chrom.features = deepcopy(old_features)
        chrom.mutate_add()
        assert len(chrom) == min(n_features, len(old_features)+1)
        chrom.features = deepcopy(old_features)
        chrom.mutate_split()
        if len(old_features) == 1:
            assert len(chrom) == 1
        elif len(old_features) == 2:
            assert len(chrom) == 1
        else:
            assert len(chrom) == len(old_features) - len(old_features)//3

        chrom.features = deepcopy(old_features)
        for i in range(1000):
            chrom.mutate(rate=1)
            if len(chrom) != len(old_features):
                break
            assert i < 999


@mark.parametrize(
    "n_examples,n_features,njobs", [
        (200, 20, 1),
        (40, 2, -1),
        (139, 5, 2),
        (200, 7, 4)
    ]
)
def test_ga(n_examples, n_features, njobs):
    n_true_features = n_features // 2
    X, y, _ = create_simulation_data(
        n_examples=n_examples, n_features=n_features,
        n_true_features=n_true_features)

    X_gen, y_gen = genetic_algorithm(X, y, n_jobs=1, n_random=100)
    print(X_gen.shape)
    assert X_gen.shape[1] == n_features + 100
    assert y_gen.size == X_gen.shape[0]

    coefs = compute_coefs(X_gen, y_gen, n_random=100)
    assert len(coefs) == n_features + 100
    features = compute_significant_features(coefs, n_random=100,
                                            sign_criterion=n_features//2)
    assert isinstance(features, np.ndarray)
    assert np.all(features < n_features)
