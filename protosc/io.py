import cv2
from protosc.pipeline import BasePipeElement


class ReadImage(BasePipeElement):
    """Read an image from a file.

    Arguments
    ---------
    fp: (str, pathlib.Path)
        Path to file.

    Returns
    -------
    img: np.ndarray
        Numpy array with the image data (x, y, 3)
    """
    def _execute(self, fp):
        return read_image(fp)


def read_image(fp):
    img = cv2.imread(str(fp))
    if img is None:
        raise FileNotFoundError(f"Cannot read file '{str(fp)}'.")
    return img
