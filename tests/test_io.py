from pathlib import Path

import pytest
from protosc.io import read_image, ReadImage


@pytest.mark.parametrize(
    "filename,resolution", [
        ("christopher.jpg", (5184, 3456)),
        ("dorell.jpg", (1891, 2836)),
        ("prince.jpg", (1500, 2248)),
        ("sergio.jpg", (3000, 2000))],
)
def test_read_image(filename, resolution):
    portrait_fp = Path("tests", "data", filename)
    img = read_image(portrait_fp)
    assert img.shape == (resolution[1], resolution[0], 3)
    ri = ReadImage()
    try:
        ri.execute("Non-existing file")
        assert False, "Expecting FileNotFoundError."
    except FileNotFoundError:
        pass
