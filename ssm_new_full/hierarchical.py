class _Hierarchical:
    def __init__(self, base_class, *args, tags=(None,), **kwargs):
        self.parent = base_class(*args, **kwargs)
        self.tags = tags
        self.children = {tag: base_class(*args, **kwargs) for tag in tags}

    @property
    def params(self):
        return (self.parent.params,) + tuple(child.params for child in self.children.values())

    @params.setter
    def params(self, value):
        self.parent.params = value[0]
        for child, child_params in zip(self.children.values(), value[1:]):
            child.params = child_params

    def permute(self, perm):
        self.parent.permute(perm)
        for child in self.children.values():
            child.permute(perm)


class HierarchicalTransitions(_Hierarchical):
    pass


class HierarchicalObservations(_Hierarchical):
    pass
