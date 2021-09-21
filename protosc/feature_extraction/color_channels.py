import skimage as sk

from protosc.pipeline import BasePipeElement


class SetColorChannels(BasePipeElement):
    """Image preprocessing step for color
    conversion and selecting specific color channels
    Arguments
    ---------
    convert2cielab: bool
       use to convert rgb to cielab (convert2cielab = True/Flase)
    get_layers: array
       Select which channels of the image to keep (get_layers= [0,1,2])
    Returns
    -------
    img: the adjusted image
    """
    def __init__(self, convert2cielab=False, get_layers=[]):
        self.convert2cielab = convert2cielab
        self.get_layers = get_layers

    def _execute(self, img):
        return set_color_channels(
            img,
            convert2cielab=self.convert2cielab,
            get_layers=self.get_layers)


def set_color_channels(img, convert_to_cielab=False, get_layers=[]):
    # preprocessing step for images
    # use to convert rgb to cielan (convert2cielab = True/Flase)
    # Select which channels of the image to keep (get_layers= [0,1,2])
    # Convert RGB to Cie Lab
    if convert_to_cielab:
        img = sk.color.rgb2lab(img)
    # Check which image layers to include
    if get_layers == []:
        get_layers = range(0, img.shape[2])
    newimg = img[:, :, get_layers]
    return newimg
