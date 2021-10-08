from pathlib import Path

from protosc.feature_extraction.hog import HOGFeatures
from protosc.io import ReadImage
from protosc.preprocessing import GreyScale, ViolaJones, CutCircle
from protosc.feature_extraction import FourierFeatures
from protosc.feature_matrix import FeatureMatrix
from _collections import defaultdict


def remove_force(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def test_plotting():
    pipe1 = ReadImage()*GreyScale()*ViolaJones()*CutCircle()*FourierFeatures()
    pipe2 = ReadImage()*GreyScale()*ViolaJones()*CutCircle()*HOGFeatures()

    pipe_complex = pipe1 + pipe2

    test_files = Path("tests", "data").glob("*.jpg")
    test_files = [x for x in test_files]

    data = pipe_complex.execute(test_files)
    X = FeatureMatrix.from_pipe_data(data)
    feature_dict = defaultdict(lambda: 0)
    for feature in X.rev_lookup_table:
        feature_dict[feature["pipeline"]] += 1

    hog_pipe = [x for x in feature_dict if x.endswith("HOGFeatures")][0]
    fourier_pipe = [x for x in feature_dict if x.endswith("FourierFeatures")][0]
    assert feature_dict[hog_pipe] == 3600
    assert feature_dict[fourier_pipe] == 56
    output_dir = Path("tests", "temp")
    output_dir.mkdir(exist_ok=True)
    remove_force(Path(output_dir, "hog.png"))
    remove_force(Path(output_dir, "fourier.png"))
    X.plot([23, 15, 1349, 2348], plot_dir=output_dir)
    assert Path(output_dir, "hog.png").is_file()
    assert Path(output_dir, "fourier.png").is_file()

    remove_force(Path(output_dir, "hog.png"))
    remove_force(Path(output_dir, "fourier.png"))
    X.plot(23, plot_dir=output_dir)
    assert Path(output_dir, "fourier.png").is_file()
    X.plot(1349, plot_dir=output_dir)
    assert Path(output_dir, "hog.png").is_file()
