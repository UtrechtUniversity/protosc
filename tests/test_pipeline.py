from pathlib import Path

from protosc.io import ReadImage
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.feature_extraction import FourierFeatures
from protosc.pipe_complex import PipeComplex
from protosc.pipeline import Pipeline


def test_pipe_complex():
    pipe1 = ReadImage()*GreyScale()*ViolaJones()*CutCircle()*FourierFeatures()
    pipe2 = ReadImage()*ViolaJones()*FourierFeatures(cut_circle=False)
    pipe3 = ReadImage()*ViolaJones()*FourierFeatures(cut_circle=False,
                                                     absolute=False)
    pipe_complex = pipe1 + pipe2 + pipe3
    pipe_complex2 = PipeComplex()
    pipe_complex2 += pipe1
    pipe_complex2 += PipeComplex(pipe2)
    pipe_complex2 = pipe_complex2+PipeComplex(pipe3)

    assert isinstance(pipe_complex, PipeComplex)

    test_files = Path("tests", "data").glob("*.jpg")
    test_files = [x for x in test_files]
    results = pipe_complex.execute(test_files)
    assert len(results) == len(test_files)
    assert len(results[0]) == 3
    results = pipe_complex.execute([test_files])
    assert len(results) == 1
    assert len(results[0]) == len(test_files)
    results2 = pipe_complex.execute([test_files], max_depth=2)
    assert len(results2[0]) == len(test_files)

    assert len(str(pipe_complex).split("\n")) == len(pipe_complex)
    assert str(pipe_complex) == str(pipe_complex2)
    pipe_complex2 += ReadImage()


def test_pipe_complex_mul():
    pipe_complex = ReadImage()*(GreyScale()+ViolaJones())*(CutCircle()+FourierFeatures())*FourierFeatures()
    assert len(pipe_complex) == 4
    try:
        pipe_complex*4
    except TypeError:
        pass

    try:
        4*pipe_complex
    except TypeError:
        pass

    pipe_complex += ReadImage()
    pipe_complex + ReadImage()

    try:
        4+pipe_complex
    except TypeError:
        pass


def test_pipe_complex_error():
    pipe_complex = PipeComplex(ReadImage()*GreyScale())
    test_files = ["doesnt_exist"]
    res = pipe_complex.execute(test_files)
    assert isinstance(res[0]["GreyScale"], BaseException)


def test_pipeline():
    pipe = ReadImage()*GreyScale()
    pipe2 = ViolaJones()*CutCircle()
    pipe_merge = pipe*pipe2
    pipe_merge2 = pipe*ViolaJones()
    test_files = [p for p in Path("tests", "data").glob("*.jpg")]
    res = pipe_merge.execute(test_files)
    assert len(res) == len(test_files)
    try:
        Pipeline(None)
        assert False
    except TypeError:
        pass

    res = pipe_merge.execute("Nonsense")
    assert isinstance(res, FileNotFoundError)
    try:
        3*pipe_merge
        assert False
    except TypeError:
        pass
    try:
        3+pipe_merge2
    except TypeError:
        pass
    try:
        pipe_merge*3
    except TypeError:
        pass


def test_element():
    elem = ReadImage()
    test_files = [p for p in Path("tests", "data").glob("*.jpg")]

    elem.execute(test_files)
    try:
        elem+3
    except TypeError:
        pass
