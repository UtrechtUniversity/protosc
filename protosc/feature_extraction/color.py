import numpy as np

from protosc.pipeline import BasePipeElement


class ColorFeatures(BasePipeElement):
    """Extract Color distribution features from image
    Arguments
    ---------
    nsteps: int
        The number of bins used on the pdf of color values
    Returns
    -------
    color_distributions: vector of color pdf values
    ref_grid: matrix where each value corresponds to an
    index in ColorDistributions.
    Use this to find where in the image a particular
    feature value comes from
    """
    def __init__(self, nsteps=25):
        self.nsteps = nsteps

    def _execute(self, img):
        return color_features(
            img, nsteps=self.nsteps)


def color_features(img, nsteps=25):
    # preallocate color_distributions
    color_distributions = []
    # preallocate reference frame
    ref_grid = np.zeros([img.shape[2], nsteps])
    count = 0
    for channel in range(0, img.shape[2]):
        count = count+1
        color_distributions_temp, b = np.histogram(
            np.reshape(img[:, :, channel], img.shape[0]*img.shape[1]),
            nsteps,
            density=True)
        color_distributions = np.concatenate((color_distributions,
                                              color_distributions_temp))
        ref_grid[count-1, :] = np.array(range(nsteps*(count-1),
                                              nsteps*(count)))

    return color_distributions, ref_grid
