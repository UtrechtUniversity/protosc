from protosc.pipeline import BasePipeElement, Pipeline
from protosc.utils import get_new_level
from _collections import defaultdict
from protosc.utils import Settings


class PipeComplex():
    """Set of pipelines that combines them efficiently.

    On a conceptual level, pipe complexes consist of a set of pipelines,
    which in turn consist of pipe elements, which are simply preprocessing
    or feature extraction steps.

    Caching is currently not supported.
    """
    def __init__(self, *pipes):
        self.pipelines = {}
        self._feature_counts = defaultdict(lambda: 0)
        self.settings = Settings({})
        for pipe in pipes:
            self.add_pipeline(pipe)

    def __str__(self):
        pipe_str = ""
        max_len = max([len(x) for x in self.pipelines])
        for name, pipe in self.pipelines.items():
            name_str = f"{name}".ljust(max_len)
            pipe_str += f"{name_str}: {str(pipe)}\n"
        return pipe_str[:-1]

    def __iadd__(self, other):
        """Add another parallel pipeline/complex/element."""
        if isinstance(other, PipeComplex):
            self.add_complex(other)
        if isinstance(other, (Pipeline, BasePipeElement)):
            self.add_pipeline(other)
        return self

    def __add__(self, other):
        """Add another parallel pipeline/complex/element."""
        my_pipelines = [x for x in self]
        if isinstance(other, PipeComplex):
            other_pipelines = [x for x in other]
        elif isinstance(other, Pipeline):
            other_pipelines = [other]
        elif isinstance(other, BasePipeElement):
            other_pipelines = [Pipeline(other)]
        else:
            return NotImplemented
        cls = self.__class__
        return cls(*my_pipelines, *other_pipelines)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, PipeComplex):
            my_pipelines = [p for p in self]
            other_pipelines = [p for p in other]
            new_pipelines = []
            for my_pipe in my_pipelines:
                for other_pipe in other_pipelines:
                    new_pipelines.append(Pipeline(my_pipe, other_pipe))
            return self.__class__(*new_pipelines)
        if isinstance(other, (Pipeline, BasePipeElement)):
            all_pipelines = [Pipeline(p, other) for p in self]
            return __class__(*all_pipelines)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (Pipeline, BasePipeElement)):
            all_pipelines = [Pipeline(other, p) for p in self]
            return self.__class__(*all_pipelines)
        return NotImplemented

    def __iter__(self):
        """Looping over a pipe complex generates pipelines."""
        for pipe in self.pipelines.values():
            yield pipe
#         def generate_pipelines(pipe_tree, cur_elements):
#             for key, new_pipe_tree in pipe_tree.items():
#                 if key is None:
#                     yield Pipeline(*cur_elements)
#                 else:
#                     cur_elements.append(self._pipe_elements[key])
#                     yield from generate_pipelines(new_pipe_tree, cur_elements)
#                     cur_elements.pop()
#         return generate_pipelines(self._pipe_tree, [])

    def __len__(self):
        """Return the number of pipelines."""
        return len(self.pipelines)

    def add_complex(self, other):
        for pipeline in other:
            self.add_pipeline(pipeline)

    def get_settings(self):
        my_settings_dict = {}
        for name, pipe in self.pipelines.items():
            my_settings_dict[name] = pipe.settings
        return Settings(my_settings_dict)

    def add_pipeline(self, other):
        """Add a single parallel pipeline to the complex (self)."""
        if isinstance(other, BasePipeElement):
            other = Pipeline(other)

        counts = self._feature_counts[other.name]
        if counts == 0:
            pipe_name = f"{other.name}"
        else:
            pipe_name = f"{other.name}_{counts+1}"
        self.pipelines[pipe_name] = other

        self._feature_counts[other.name] += 1
        self.settings[pipe_name] = other.settings

    def execute(self, package, max_depth=None):
        """Execute the pipelines on data.

        The format of the data should be the same as what the
        first pipe element expects, or iterables thereof.
        """
        # If the package is iterable and we haven't reached max_depth
        # return a list with the results of all parts of the package.
        iterate, new_max_depth = get_new_level(package, max_depth)
        if iterate:
            return [self.execute(part, new_max_depth) for part in package]
        else:
            return self.execute_single(package)

    def execute_single(self, package):
        for pipe_name, pipeline in self.pipelines.items():
            for elem in pipeline:
                kwargs = getattr(getattr(self.settings, pipe_name, {}),
                                 elem.name, {}).todict()
                for key, val in kwargs.items():
                    setattr(elem, key, val)

        results = get_result(package, self.pipelines)
        return results


def split(pipelines, i_elem):
    all_pipelines = []
    cur_set = {}
    cur_compare = None
    for name, pipe in pipelines.items():
        if len(cur_set) == 0 or pipe[i_elem] == cur_compare:
            cur_set[name] = pipe
            cur_compare = pipe[i_elem]
        else:
            all_pipelines.append(cur_set)
            cur_set = {name: pipe}
    if len(cur_set) > 0:
        all_pipelines.append(cur_set)
    return all_pipelines


def get_result(package, pipelines, i_elem=0):
    results = {}
    unfinished_pipelines = {}
    for name, pipe in pipelines.items():
        if len(pipe) == i_elem:
            results[name] = package
        else:
            unfinished_pipelines[name] = pipe
    split_pipelines = split(unfinished_pipelines, i_elem)
    for pipes in split_pipelines:
        element = pipes[list(pipes)[0]][i_elem]
        try:
            if not isinstance(package, BaseException):
                new_package = element.execute(package)
            else:
                new_package = package
            new_result = get_result(new_package, pipes, i_elem+1)
        except BaseException as e:
            e.source = element.name
            new_result = get_result(e, pipes, i_elem+1)
        results.update(new_result)
    return results
