from pathlib import Path

import numpy as np

from protosc.io import read_image
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle


def test_greyscale():
    test_files = Path("tests", "data").glob("*.jpg")
    grey = GreyScale()
    assert grey.name == "GreyScale"
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


def test_VJ():
    vj = ViolaJones()
    vj2 = ViolaJones(add_perc=10)
    assert vj.name == "ViolaJones_20"
    assert vj2.name == "ViolaJones_10"
    test_files = Path("tests", "data").glob("*.jpg")
    for test_fp in test_files:
        img = read_image(test_fp)
        gs_img = GreyScale().execute(img).reshape(img.shape[:2])
        res = vj.execute(img)
        res2 = vj2.execute(img)
        res3 = vj.execute(gs_img)
        assert isinstance(res, np.ndarray)
        assert res.shape[2] == 3
        assert not np.all(res == res2)
        assert np.all(res3.shape == (200, 200, 1))
        assert np.all(res.shape == (200, 200, 3))
    rand_img = np.random.randint(256, size=(200, 300, 3)).astype(np.uint8)
    try:
        vj.execute(rand_img)
        assert False, "Expecting failure to converge."
    except ValueError as e:
        assert str(e) == "ViolaJones: Cannot find face in picture!"

    try:
        vj.execute(False)
        assert False, "Expecting type error"
    except TypeError:
        pass


def test_cut_circle():
    cc = CutCircle()
    assert cc.name == "CutCircle"
    test_files = Path("tests", "data").glob("*.jpg")
    for test_fp in test_files:
        img = read_image(test_fp)
        res = cc.execute(img)
        assert img.shape == res.shape
        assert res[0, 0, 0] == 0
        assert res[-1, -1, -1] == 0

    try:
        cc.execute(False)
        assert False, "Expecting type error"
    except TypeError:
        pass
