import numpy as np

from protosc.feature_extraction import transform_matrix
import pytest


@pytest.mark.parametrize('shape', [(21, 31), (22, 22), (30, 21)])
@pytest.mark.parametrize('cut_circle', [True, False])
@pytest.mark.parametrize('n_angular', [5, 8])
@pytest.mark.parametrize('n_spatial', [6, 7])
def test_transform_matrix(shape, cut_circle, n_angular, n_spatial):
    size = shape[0]*shape[1]
    n_parts = n_angular*n_spatial
    trans_matrix, all_id, inv_trans_matrix = transform_matrix(
        shape, n_angular=n_angular, n_spatial=n_spatial, cut_circle=cut_circle,
        return_inverse=True, return_ids=True)

    if cut_circle:
        assert trans_matrix.shape[1] == size
        assert trans_matrix.shape[0] <= n_parts
        assert inv_trans_matrix.shape[0] == size
        assert inv_trans_matrix.shape[1] <= n_parts
    else:
        assert trans_matrix.shape[1] == size
        assert trans_matrix.shape[0] > n_parts
        assert inv_trans_matrix.shape[0] == size
        assert inv_trans_matrix.shape[1] > n_parts

    for i_pixel, surf_id in enumerate(all_id.reshape(-1)):
        if surf_id == -1:
            assert np.sum(trans_matrix[:, i_pixel]) == 0
        else:
            assert trans_matrix[surf_id, i_pixel] == 1
            assert inv_trans_matrix[i_pixel, surf_id] > 0

    assert np.allclose(inv_trans_matrix.sum(axis=0), 1)
