import numpy as np
from protosc.feature_extraction import transform_matrix
from protosc.feature_extraction import hog_features
from protosc.feature_extraction import color_features
from protosc.feature_extraction import pixel_features
# from protosc.feature_extraction import set_color_channels
import pytest

# for test_hog_features:
@pytest.mark.parametrize('orientations', [9, 8, 7])
@pytest.mark.parametrize('hog_cellsize', [[10, 10],[5, 5]])
def test_hog_features(orientations, hog_cellsize):
    test_img = np.random.rand(200, 200, 3)
    hogs, ref_grid_hog = hog_features(test_img, orientations, hog_cellsize)

    assert hogs.shape[0] == (test_img.shape[0]/hog_cellsize[0])*(test_img.shape[1]/hog_cellsize[1])*orientations
    assert ref_grid_hog.shape == (test_img.shape[0]/hog_cellsize[0], test_img.shape[1]/hog_cellsize[1], orientations)

# for test_color_features, test_pixel_features
@pytest.mark.parametrize('nchannels', [1, 2, 3])
# for test_color_features
@pytest.mark.parametrize('nsteps', [25, 10, 50])
def test_color_features(nchannels, nsteps):
    test_img = np.random.rand(200, 200, nchannels)
    color_distributions, ref_grid = color_features(test_img, nsteps)

    assert color_distributions.shape[0] == nsteps*nchannels
    assert ref_grid.shape == (nchannels, nsteps)

# for test_color_features, test_pixel_features
@pytest.mark.parametrize('nchannels', [1, 2, 3])
# test_pixel_features
@pytest.mark.parametrize('newsize', [[25, 25], [50, 50]])
def test_pixel_features(nchannels, newsize):
    test_img = np.random.rand(200, 200, nchannels)
    pixel_intensities, ref_grid = pixel_features(test_img,newsize)

    assert pixel_intensities.shape == (1, newsize[0]*newsize[1], nchannels)
    assert ref_grid.shape == (newsize[0], newsize[1], nchannels)

# for test_transform_matrix
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
