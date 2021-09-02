import numpy as np
from skimage.transform import resize

from protosc.pipeline import BasePipeElement


class PixelFeatures(BasePipeElement):
    """Extract pixel intesity features from image
    Arguments
    ---------
    newsize: [int,int]
       prior to extracting the pixel intensities,
       the image is converted to this size to reduce the
       number of features
    Returns
    -------
    Pixel_Intensities: vector of pixel intensities
    refGrid: matrix where each value corresponds to an index
    in Pixel_Intensities.
    Use this to find where in the image a particular feature
    value comes from.
    """
    def __init__(self, newsize=[25, 25]):
        self.newsize = newsize

    def _execute(self, img):
        return pixel_features(
            img, newsize=self.newsize)


def pixel_features(img, newsize=[25, 25]):
    img = resize(img, newsize)
    pixel_intensities = np.reshape(img,
                                   [1, img.shape[0]*img.shape[1],
                                    img.shape[2]])
    ref_grid_pixel_intensities = np.zeros([img.shape[0],
                                           img.shape[1],
                                           img.shape[2]])
    c = 0
    for x in range(0, img.shape[1]-1):
        for y in range(0, img.shape[0]-1):
            for z in range(0, img.shape[2]-1):
                ref_grid_pixel_intensities[y, x, z] = c
                c = c+1
    return pixel_intensities, ref_grid_pixel_intensities
