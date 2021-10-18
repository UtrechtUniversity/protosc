from protosc.model.combined_fold import CombinedFoldModel
from protosc.model.genetic import GeneticModel
from protosc.utils import Settings
from protosc.pipe_complex import PipeComplex
from protosc.io import ReadImage
from protosc.preprocessing import ViolaJones, CutCircle
from protosc.feature_extraction.fourier_features import FourierFeatures
from protosc.feature_extraction.hog import HOGFeatures


class ProtoscSettings(Settings):
    def __init__(self, pipeline=None):
        settings_dict = {
            "combined": Settings.from_model(CombinedFoldModel),
            "genetic": Settings.from_model(GeneticModel),
        }
        self._settings_dict = settings_dict


default_visual_pipeline = PipeComplex(
    ReadImage()*ViolaJones()*CutCircle()*FourierFeatures(),
    ReadImage()*ViolaJones()*CutCircle()*HOGFeatures(),
    ReadImage()*FourierFeatures(),
)
