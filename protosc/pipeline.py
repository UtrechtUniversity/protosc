from abc import ABC, abstractmethod


def is_iterable(i):
    """Check if a variable is iterable, but not a string."""
    try:
        iter(i)
        if isinstance(i, str):
            return False
        return True
    except TypeError:
        return False


class BasePipe(ABC):
    def __init__(self, *args):
        self._elements = []
        for arg in args:
            if isinstance(arg, BasePipeElement):
                self._elements.append(arg)
            elif isinstance(arg, BasePipe):
                self._elements.extend(arg._elements)
            else:
                raise ValueError(f"Cannot extend pipe with type: {type(arg)}")

    def execute(self, package, max_depth=0):
        if is_iterable(package) and max_depth > 0:
            return [self.execute(part, max_depth-1) for part in package]

        new_package = package
        for element in self._elements:
            new_package = element.execute(new_package)
        return new_package

    def __mul__(self, other):
        return BasePipe(self, other)

    def __rmul__(self, other):
        return BasePipe(other, self)

    def __str__(self):
        return "Pipe: " + " -> ".join([x.name for x in self._elements])


class BasePipeElement(ABC):
    @abstractmethod
    def execute(self, _):
        raise NotImplementedError

    def __mul__(self, other):
        return BasePipe(self, other)

    def __rmul__(self, other):
        return BasePipe(other, self)

    @property
    def name(self):
        return type(self).__name__
