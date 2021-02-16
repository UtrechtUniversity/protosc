import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix


def transform_matrix(shape, n_angular=80, n_spatial=70, return_inverse=True,
                     return_ids=False, circle_cut=True):
    size = shape[0]*shape[1]
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    middle = np.array([shape[0]//2, shape[1]//2])
    X -= middle[0]
    Y -= middle[1]

    radius = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)

    d_angle = 2*np.pi/n_angular
    d_radius = np.max(middle)/n_spatial
    angle_id = ((angle/d_angle + 0.5*(n_angular+1))%n_angular).astype(int)
    radius_id = (radius/d_radius).astype(int)
    all_id = angle_id+radius_id*n_angular

    indptr = np.arange(size+1)
    indices = all_id.reshape(-1)
    data = np.ones(size, dtype=int)
    trans_shape = (all_id.max()+1, size)

    if circle_cut:
        circle_mask = (radius_id.reshape(-1) < n_spatial)
        trans_shape = (all_id.reshape(-1)[circle_mask].max()+1, size)
        indptr = np.append([0], np.cumsum(circle_mask))
        indices = indices[circle_mask]
        data = data[circle_mask]

    trans_matrix = csc_matrix((data, indices, indptr),
                              shape=trans_shape)

    results = []

    results.append(trans_matrix)
    if return_ids:
        results.append(all_id.reshape(-1))

    if not return_inverse:
        if len(results) == 1:
            return trans_matrix
        return results

    idx, temp_counts = np.unique(all_id, return_counts=True)
    counts = np.zeros(all_id.max()+1)
    counts[idx] = temp_counts
    indptr = np.arange(size+1)
    indices = all_id.reshape(-1)
    data = 1/counts[all_id.reshape(-1)]

    inv_trans_matrix = csr_matrix((data, indices, indptr), shape=(size, all_id.max()+1))
    results.append(inv_trans_matrix)
    return results


def fourier_features(img, *args, **kwargs):
    fft_map = np.fft.fftshift(np.fft.fft2(img-np.mean(img)))
    trans = transform_matrix(fft_map.shape, *args, return_inverse=False, **kwargs)
    return trans.dot(fft_map.reshape(-1, fft_map.shape[2]))