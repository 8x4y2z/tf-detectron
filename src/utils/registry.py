# -*- coding: utf-8 -*-

class Registry:
    def __init__(self,name):
        "docstring"
        self._name = name
        self._map = {}

    def register(self,obj=None):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                if name in self._map:
                    raise RuntimeError("Key already present")
                self._map[name] = func_or_class
                return func_or_class
            return deco
        if obj.__name__ in self._map:
            raise RuntimeError("Key already present")
        self._map[obj.__name__] = obj

    def get(self, name):
        ret = self._map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret
