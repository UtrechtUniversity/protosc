from protosc.pipeline import BasePipeElement, Pipeline
from protosc.utils import get_new_level


class PipeComplex():
    def __init__(self, *pipes):
        self._pipe_elements = {}
        self._pipe_tree = {}
        for pipe in pipes:
            self.add_pipeline(pipe)

    def __str__(self):
        return "\n".join([str(p) for p in self])

    def __iadd__(self, other):
        if isinstance(other, PipeComplex):
            self.add_complex(other)
        if isinstance(other, (Pipeline, BasePipeElement)):
            self.add_pipeline(other)
        return self

    def __add__(self, other):
        my_pipelines = [l for l in self]
        if isinstance(other, PipeComplex):
            other_pipelines = [l for l in other]
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
            my_pipelines = [Pipeline(p, other) for p in self]
            other_pipelines = [Pipeline(p, other) for p in self]
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
        if isinstance(other, PipeComplex):
            raise ValueError("Uhuh?")
        if isinstance(other, (Pipeline, BasePipeElement)):
            all_pipelines = [Pipeline(other, p) for p in self]
            return self.__class__(*all_pipelines)
        return NotImplemented

    def __iter__(self):
        def generate_pipelines(pipe_tree, cur_elements):
            for key, new_pipe_tree in pipe_tree.items():
                if key is None:
                    yield Pipeline(*cur_elements)
                else:
                    cur_elements.append(self._pipe_elements[key])
                    yield from generate_pipelines(new_pipe_tree, cur_elements)
                    cur_elements.pop()
        return generate_pipelines(self._pipe_tree, [])

    def add_complex(self, other):
        for pipeline in other:
            self.add_pipeline(pipeline)

    def add_pipeline(self, other):
        if isinstance(other, BasePipeElement):
            other = Pipeline(other)
        tree_pointer = self._pipe_tree
        for elem in other._elements:
            if elem.name in tree_pointer:
                tree_pointer = tree_pointer[elem.name]
            else:
                tree_pointer[elem.name] = {}
                tree_pointer = tree_pointer[elem.name]

            if elem.name not in self._pipe_elements:
                self._pipe_elements[elem.name] = elem
        tree_pointer[None] = other.name

    def execute(self, package, max_depth=None):
        iterate, new_max_depth = get_new_level(package, max_depth)
        if iterate:
            return [self.execute(part, new_max_depth) for part in package]

        def get_result(package, pipe_tree):
            results = {}
            for key, new_pipe_tree in pipe_tree.items():
                if key is None:
                    results[new_pipe_tree] = package
                else:
                    element = self._pipe_elements[key]
                    new_package = element.execute(package)
                    results.update(get_result(new_package, new_pipe_tree))
            return results
        return get_result(package, self._pipe_tree)
