import numpy as np
import inspect


def is_iterable(i):
    """Check if a variable is iterable, but not a string."""
    try:
        iter(i)
        if isinstance(i, str):
            return False
        return True
    except TypeError:
        return False


def get_new_level(package, max_depth=None):
    iterate = False
    new_max_depth = max_depth
    if is_iterable(package):
        if max_depth is None:
            new_max_depth = None
            if isinstance(package, np.ndarray):
                if package.dtype == object:
                    iterate = True
                else:
                    iterate = False
            else:
                iterate = True
        elif max_depth > 0:
            new_max_depth = max_depth - 1
            iterate = True
    return iterate, new_max_depth


def sig_to_param(signature):
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class Settings():
    def __init__(self, settings_dict):
        if isinstance(settings_dict, Settings):
            settings_dict = settings_dict._settings_dict
        self._settings_dict = settings_dict

    def __str__(self):
        return self.create_str()

    def __repr__(self):
        return self.__str__()

    def create_str(self, indent=0):
        cur_str = ""
        for key, value in self._settings_dict.items():
            if isinstance(value, Settings):
                cur_str += f"{' '*indent}{key}:\n"
                cur_str += value.create_str(indent+4)
            else:
                cur_str += f"{' '*indent}{key} = {value}\n"
        return cur_str

    @classmethod
    def from_model(cls, model):
        return cls(model().default_param)

    def __setattr__(self, key, value):
        if key == "_settings_dict" or key.startswith("__"):
            super().__setattr__(key, value)
            return

        self.set_key(key, value)

    def __setitem__(self, key, value):
        self._settings_dict[key] = value

    def __getattr__(self, key):
        if key == "_settings_dict":
            return self._settings_dict
        return self.find_key(key)

    def find_key(self, key):
        if key in self._settings_dict:
            return self._settings_dict[key]
        for value in self._settings_dict.values():
            if isinstance(value, Settings):
                try:
                    return value.find_key(key)
                except KeyError:
                    pass
        raise KeyError(f"Setting '{key}' not found.")

    def set_key(self, key, value):
        if key in self._settings_dict:
            self._settings_dict[key] = value
            return
        for section in self._settings_dict.values():
            if isinstance(section, Settings):
                try:
                    section.set_key(key, value)
                    return
                except KeyError:
                    pass
        raise KeyError(f"Setting '{key}' not found.")

    def todict(self, recursive=False):
        return self._settings_dict
