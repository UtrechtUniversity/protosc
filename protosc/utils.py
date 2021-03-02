import numpy as np


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
