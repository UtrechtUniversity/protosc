from protosc.genetic_algorithm import Chromosome

from pytest import mark
from protosc.simulation import create_simulation_data
from copy import deepcopy


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
    X, y, _ = create_simulation_data(n_examples=n_examples, n_features=n_features, n_true_features=n_true_features)
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
