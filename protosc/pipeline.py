from abc import ABC, abstractmethod

from protosc.utils import is_iterable


class Pipeline(ABC):
    def __init__(self, *pipe_elements):
        self._elements = []
        for arg in pipe_elements:
            if isinstance(arg, BasePipeElement):
                self._elements.append(arg)
            elif isinstance(arg, Pipeline):
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
        if isinstance(other, (Pipeline, BasePipeElement)):
            return Pipeline(self, other)
        return NotImplemented

    def __rmul__(self, other):
        return Pipeline(other, self)

    def __add__(self, other):
        from protosc.pipe_complex import PipeComplex
        if isinstance(other, (Pipeline, BasePipeElement)):
            return PipeComplex(self, other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return "Pipe: " + " -> ".join([x.name for x in self._elements])

    @property
    def name(self):
        return "+".join([x.name for x in self._elements])


class BasePipeElement(ABC):
    @abstractmethod
    def execute(self, _):
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, BasePipeElement):
            return Pipeline(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, BasePipeElement):
            from protosc.pipe_complex import PipeComplex
            return PipeComplex(self, other)
        return NotImplemented
#     def __rmul__(self, other):
#         return BasePipe(other, self)

    @property
    def name(self):
        return type(self).__name__
