from protosc.model.combined_fold import CombinedFoldModel
from protosc.model.genetic import GeneticModel
from protosc.utils import Settings


class ProtoscSettings(Settings):
    def __init__(self):
        settings_dict = {
            "combined": Settings.from_model(CombinedFoldModel),
            "genetic": Settings.from_model(GeneticModel),
        }
        self._settings_dict = settings_dict
