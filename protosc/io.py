import cv2


def read_image(fp):
    return cv2.imread(str(fp))
