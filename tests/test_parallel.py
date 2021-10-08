import numpy as np

from protosc.simulation import create_independent_data
from protosc.model.combined_fold import CombinedFoldModel
from tqdm import tqdm


def test_parallel():
    X, y, _ = create_independent_data(n_features=20, n_samples=100, n_true_features=5)
    model = CombinedFoldModel()
    all_res = []
    np.random.seed(192873)
    fold_seed = 28744,
    seed = np.random.randint(1239823)
    for _ in range(3):
        single_res = model.execute(X, y, fold_seed, seed=seed, n_jobs=1)
        par_res = model.execute(X, y, fold_seed, seed=seed, n_jobs=3)
        pbar_res = model.execute(X, y, fold_seed, seed, n_jobs=1, progress_bar=True)
        pbar_par_res = model.execute(X, y, fold_seed, seed, n_jobs=3, progress_bar=True)
        pbar = tqdm(total=8)
        pbar_x_res = model.execute(X, y, fold_seed, seed, n_jobs=3, progress_bar=pbar)
        assert pbar.n == 8
        pbar.close()
        check_same(single_res, par_res)
        check_same(single_res, pbar_res)
        check_same(single_res, pbar_par_res)
        check_same(single_res, pbar_x_res)
        all_res.append(par_res)

    for i in range(1, 3):
        check_same(all_res[0], all_res[i])


def check_same(a, b):
    for name in a:
        assert len(a[name]) == len(b[name])
        assert np.allclose(a[name], b[name])
