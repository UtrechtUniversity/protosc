import numpy as np
from PIL import Image


def read_image(fp, grayscale=True):
    img = Image.open(fp)
    if grayscale:
        img = img.convert('LA')
    return np.array(img)[:, :, 0]
