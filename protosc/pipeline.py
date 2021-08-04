from abc import ABC, abstractmethod
import inspect

from protosc.utils import get_new_level


class Pipeline(ABC):
    def __init__(self, *pipe_elements):
        self._elements = []
        for arg in pipe_elements:
            if isinstance(arg, BasePipeElement):
                self._elements.append(arg)
            elif isinstance(arg, Pipeline):
                self._elements.extend(arg._elements)
            else:
                raise TypeError(f"Cannot extend pipe with type: {type(arg)}")

    def execute(self, package, max_depth=None):
        iterate, new_max_depth = get_new_level(package, max_depth)
        if iterate:
            return [
                self.execute(part, new_max_depth)
                for part in package]

        new_package = package
        for element in self._elements:
            try:
                new_package = element.execute(new_package)
            except BaseException as e:
                e.source = element.name
                return e
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


def sig_to_param(signature):
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class Plotter():
    def __init__(self, name, ref_func, plot_func):
        self.ref_func = ref_func
        self.plot_func = plot_func
        self.name = name
        self._ref_data = None

    @property
    def ref_data(self):
        if self._ref_data is None:
            self._ref_data = self.ref_func()
        return self._ref_data

    def plot(self, *args, **kwargs):
        self.plot_func(self.ref_data, *args, **kwargs)


class BasePipeElement(ABC):
    @abstractmethod
    def _execute(self, _package):
        raise NotImplementedError

    def _get_ref_func(self, _package):
        """Function to compute references used for plotting.

        It is not necessary to implement this function, but visualization
        will not be available for this feature.

        Arguments
        ---------
        package: ?
            The data that the pipeline element is processing.

        Returns
        -------
        name: str
            Some identification as to the reference. For example in the case
            of an image it could be the input size of the image converted to
            a string. It is used to check whether all samples have the same =
            format.
        f: function
            It should give back a function that computes the reference data to
            be used in plotting. The function should take no argument, and can
            return anything. Whatever it returns will be the the first argument
            to the plotting function, see _plot_func.
        """
        return None

    @property
    def _plot_func(self):
        """Function that should return a plotting function.

        It is not necessary to implement this function, but visualization will
        not be available.

        Returns
        -------
        f: function
            The function that is returned should take two arguments:
            ref_data (the result of the ref_function, see above), and
            i_feature, which is a list of features to be plotted.
        """
        return None

    def execute(self, package, max_depth=None):
        iterate, new_max_depth = get_new_level(package, max_depth)
        if iterate:
            return [self.execute(part, new_max_depth) for part in package]
        new_package = self._execute(package)
        ref_func = self._get_ref_func(package)
        plot_func = self._plot_func
        if plot_func is None:
            return new_package
        return new_package, Plotter(*ref_func, plot_func)

    def __mul__(self, other):
        if isinstance(other, BasePipeElement):
            return Pipeline(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, BasePipeElement):
            from protosc.pipe_complex import PipeComplex
            return PipeComplex(self, other)
        return NotImplemented

    @property
    def name(self):
        base_name = type(self).__name__
        def_param = self.default_param
        for key, value in def_param.items():
            if getattr(self, key) != value:
                base_name += f"_{key}{getattr(self, key)}"

        return base_name

    @property
    def default_param(self):
        """Get the default parameters of the model.

        Returns
        -------
        dict:
            Dictionary with parameter: default value
        """
        cur_class = self.__class__
        default_parameters = sig_to_param(inspect.signature(self.__init__))
        while cur_class != BasePipeElement:
            signature = inspect.signature(super(cur_class, self).__init__)
            new_parameters = sig_to_param(signature)
            default_parameters.update(new_parameters)
            cur_class = cur_class.__bases__[0]
        return default_parameters
