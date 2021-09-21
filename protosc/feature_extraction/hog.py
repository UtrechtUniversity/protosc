import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog

from protosc.pipeline import BasePipeElement


class HOGFeatures(BasePipeElement):
    """Extract HOG feature from an image.

    Arguments
    ---------
    orientations: int
        The number of orientation bins
    HOG_cellsize: [int,int]
        The size of the (non-overlapping) cells

    Returns
    -------
    HOGs: vector of HOG feature values
    refGrid_HOG: matrix where each value corresponds to an index in HOGs.
    Use this to find where in the image a particular HOG feature
    value comes from
    """
    def __init__(self, orientations=9, hog_cellsize=[10, 10]):
        self.orientations = orientations
        self.hog_cellsize = hog_cellsize

    def _execute(self, img):
        return hog_features(
            img, orientations=self.orientations,
            hog_cellsize=self.hog_cellsize)

    def _get_ref_func(self, img):
        name = str(self.__class__)+str(img.shape)
        ref_func = hog_ref_func
        ref_kwargs = {
            "img_shape": img.shape,
            "hog_cellsize": self.hog_cellsize,
            "orientations": self.orientations,
        }
        return name, ref_func, ref_kwargs

#     def _get_ref_func(self, img):
#         def ref_func():
#             # preallocate hog reference frame
#             ref_grid_hog = np.zeros(
#                 [np.int(np.floor(img.shape[0]/self.hog_cellsize[0])),
#                  np.int(np.floor(img.shape[1]/self.hog_cellsize[1])),
#                  self.orientations])
#             c = 0
#             for x in range(0, ref_grid_hog.shape[1]):
#                 for y in range(0, ref_grid_hog.shape[0]):
#                     for z in range(0, ref_grid_hog.shape[2]):
#                         ref_grid_hog[y, x, z] = c
#                         c = c+1
#             return ref_grid_hog
#         return str(self.__class__)+str(img.shape), ref_func

    @property
    def _plot_func(self):
        return hog_plot
#         def plot(ref_grid, i_feature):

#         return plot


def hog_plot(ref_grid, i_feature):
    data = np.zeros(ref_grid.shape[:2])
    for i in i_feature:
        x, y, _ = np.where(ref_grid == i)
        data[x[0], y[0]] += 1/ref_grid.shape[2]
    plt.imshow(data, cmap="binary", vmin=0, vmax=1)
    plt.show()


def hog_ref_func(img_shape, hog_cellsize, orientations):
    ref_grid_hog = np.zeros(
        [np.int(np.floor(img_shape[0]/hog_cellsize[0])),
         np.int(np.floor(img_shape[1]/hog_cellsize[1])),
         orientations])
    c = 0
    for x in range(0, ref_grid_hog.shape[1]):
        for y in range(0, ref_grid_hog.shape[0]):
            for z in range(0, ref_grid_hog.shape[2]):
                ref_grid_hog[y, x, z] = c
                c = c+1
    return ref_grid_hog


def hog_features(img, orientations=9, hog_cellsize=[10, 10]):
    # get hog features
    hogs = hog(img, orientations,
               hog_cellsize,
               cells_per_block=(1, 1),
               visualize=False,
               multichannel=True)
    return hogs
