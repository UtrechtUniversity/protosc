import numpy as np
from protosc.feature_extraction import transform_matrix
# from protosc.feature_extraction import hog_features
import pytest


# def test_hog_features(orientations, hog_cellsize)
#     test_img = np.random.rand(200, 200, 3)
#     # get hog features
#     hogs = hog(img, orientations,
#                hog_cellsize,
#                cells_per_block=(1, 1),
#                visualize=False,
#                multichannel=True)
#     # preallocate hog reference frame
#     ref_grid_hog = np.zeros(
#         [np.int(np.floor(img.shape[0]/hog_cellsize[0])),
#          np.int(np.floor(img.shape[1]/hog_cellsize[1])),
#          orientations])
#     c = 0
#     for x in range(0, ref_grid_hog.shape[1]):
#         for y in range(0, ref_grid_hog.shape[0]):
#             for z in range(0, ref_grid_hog.shape[2]):
#                 ref_grid_hog[y, x, z] = c
#                 c = c+1

#     return hogs, ref_grid_hog

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
