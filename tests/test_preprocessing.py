from pathlib import Path
from protosc.io import read_image
from protosc.preprocessing import GreyScale


def test_greyscale():
    test_files = Path("tests", "data").glob("*.jpg")
    grey = GreyScale()
    for test_fp in test_files:
        img = read_image(test_fp)
        new_img = grey.execute(img)
        new_img_2 = grey.execute(new_img)
        assert new_img.shape[0] == img.shape[0]
        assert new_img.shape[1] == img.shape[1]
        assert new_img.shape[2] == 1
        assert new_img_2.shape == new_img.shape

        try:
            grey.execute(None)
            assert False, "Expecting type error."
        except TypeError:
            pass
