import cv2
from protosc.pipeline import BasePipeElement


class ReadImage(BasePipeElement):
    def execute(self, fp):
        return read_image(fp)


def read_image(fp):
    return cv2.imread(str(fp))
