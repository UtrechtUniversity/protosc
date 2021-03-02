import cv2
from protosc.pipeline import BasePipeElement


class ReadImage(BasePipeElement):
    def _execute(self, fp):
        return read_image(fp)


def read_image(fp):
    img = cv2.imread(str(fp))
    if img is None:
        raise FileNotFoundError(f"Cannot read file '{str(fp)}'.")
    return img
