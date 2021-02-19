from protosc.pipeline import BasePipeElement
from protosc.utils import is_iterable


class PipeComplex():
    def __init__(self, *pipes):
        self._pipe_elements = {}
        self._pipe_tree = {}
        for pipe in pipes:
            tree_pointer = self._pipe_tree
            if isinstance(pipe, BasePipeElement):
                elements = [pipe]
            else:
                elements = pipe._elements

            pipe_name = "+".join([x.name for x in pipe._elements])
            for elem in elements:
                if elem.name in tree_pointer:
                    tree_pointer = tree_pointer[elem.name]
                else:
                    tree_pointer[elem.name] = {}
                    tree_pointer = tree_pointer[elem.name]
                if elem.name not in self._pipe_elements:
                    self._pipe_elements[elem.name] = elem
            tree_pointer[None] = pipe_name

    def __str__(self):
        def get_pipe_names(pipe_tree, cur_elements):
            cur_str = ""
            for key, new_pipe_tree in pipe_tree.items():
                if key is None:
                    cur_str += " -> ".join(cur_elements) + "\n"
                else:
                    cur_elements.append(key)
                    cur_str += get_pipe_names(new_pipe_tree, cur_elements)
                    cur_elements.pop()
            return cur_str
        return get_pipe_names(self._pipe_tree, [])

    def execute(self, package, max_depth=0):
        if is_iterable(package) and max_depth > 0:
            return [self.execute(part, max_depth-1) for part in package]

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
#         new_package = package
#         for element in self._elements:
#             new_package = element.execute(new_package)
#         return new_package
