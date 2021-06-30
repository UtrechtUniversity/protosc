from protosc.pipeline import BasePipeElement
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix


class FourierFeatures(BasePipeElement):
    def __init__(self, n_angular=8, n_spatial=7, cut_circle=True,
                 absolute=True):
        """Use fourier transformation on an image.

        At the moment reverse transformation/visualization is half implemented.
        The results are symmetrized: surfaces on opposite sides of the middle
        are averaged.

        Arguments
        ---------
        n_angular: int
            The number of angular steps in the coarse graining.
        n_spatial: int
            The number ofradial steps in the coarse graining.
        cut_circle: bool
            Whether only the inner circle has data (preprocessing).
        absolute: bool
            Whether to take the absolute values before coarse graining.

        Returns
        -------
        X: np.ndarray
            Feature matrix. If cut_circle is true, then the dimensions are
            n_absolute*n_spatial, otherwise it will be slightly larger.
        """
        self.n_angular = n_angular
        self.n_spatial = n_spatial
        self.cut_circle = cut_circle
        self.absolute = absolute

    def _execute(self, img):
        return fourier_features(
            img, n_angular=self.n_angular, n_spatial=self.n_spatial,
            cut_circle=self.cut_circle, absolute=self.absolute)

    @property
    def name(self):
        name = super(FourierFeatures, self).name
        name += f"_a{self.n_angular}s{self.n_spatial}c{self.cut_circle}"
        name += f"ab{self.absolute}"
        return name


class AbsoluteFeatures(BasePipeElement):
    def _execute(self, features):
        return np.absolute(features)


def transform_matrix(shape, n_angular=8, n_spatial=7, return_inverse=True,
                     return_ids=False, cut_circle=True):
    # Compute the x and y values for all pixels from the middle.
    size = shape[0]*shape[1]
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    middle = np.array([shape[0]//2, shape[1]//2])
    X -= middle[0]
    Y -= middle[1]

    # Compute the radius and angle for each pixel.
    radius = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)

    # Compute the coarse graining for each pixel.
    d_angle = 2*np.pi/n_angular
    d_radius = np.min(middle)/n_spatial
    angle_id = ((2*angle/d_angle + 0.5*(2*n_angular+1)
                 ) % (2*n_angular)).astype(int)
    angle_id = angle_id % n_angular
    radius_id = (radius/d_radius).astype(int)
    all_id = angle_id+radius_id*n_angular

    # Set up the sparse matrix that transforms image data.
    indptr = np.arange(size+1)
    indices = all_id.reshape(-1)
    data = np.ones(size, dtype=int)
    trans_shape = (all_id.max()+1, size)

    # If there are no values outside the inner circle:
    if cut_circle:
        circle_mask = (radius_id.reshape(-1) < n_spatial)
        trans_shape = (all_id.reshape(-1)[circle_mask].max()+1, size)

        # Remove ids outside the inner circle.
        all_id[radius_id >= n_spatial] = -1
        indptr = np.append([0], np.cumsum(circle_mask))
        indices = indices[circle_mask]
        data = data[circle_mask]

    # Create transformation matrix
    trans_matrix = csc_matrix((data, indices, indptr),
                              shape=trans_shape)
    results = []

    results.append(trans_matrix)
    # Return the coarse grained ids for all pixels.
    if return_ids:
        results.append(all_id.reshape(-1))

    if not return_inverse:
        if len(results) == 1:
            return trans_matrix
        return results

    # Compute the inverse matrix
    # Count the number of pixels for each used cell.
    idx, temp_counts = np.unique(all_id[all_id != -1], return_counts=True)
    counts = np.zeros(all_id.max()+1)
    counts[idx] = temp_counts
    indptr = np.arange(size+1)
    indices = all_id.reshape(-1)
    data = 1/counts[all_id.reshape(-1)]
    index_mask = (indices >= 0)
    indptr = np.cumsum(np.append([False], index_mask))
    data = data[index_mask]
    indices = indices[index_mask]

    # Create sparse matrix.
    inv_trans_matrix = csr_matrix((data, indices, indptr),
                                  shape=(trans_shape[1], trans_shape[0]))
    results.append(inv_trans_matrix)
    return results


def fourier_features(img, *args, absolute=True, **kwargs):
    fft_map = np.fft.fftshift(
        np.fft.fft2(img-np.mean(img, axis=(0, 1)), axes=(0, 1)))
    if absolute:
        fft_map = np.absolute(fft_map)
    trans = transform_matrix(fft_map.shape, *args, return_inverse=False,
                             **kwargs)
    return trans.dot(fft_map.reshape(-1, fft_map.shape[2]))
