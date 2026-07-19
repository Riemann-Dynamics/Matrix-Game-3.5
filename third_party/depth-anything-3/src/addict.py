"""Minimal ``addict.Dict`` compatibility used by the vendored DA3 runtime.

The upstream DA3 source imports ``addict`` but the public dependency list does
not declare it.  DA3 only relies on recursive mapping conversion and attribute
access, so keeping that tiny contract local avoids making plain depth inference
depend on an otherwise unrelated package.
"""

from __future__ import annotations


class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    @classmethod
    def _convert(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            value = type(self)()
            super().__setitem__(name, value)
            return value

    def __setattr__(self, name, value):
        if name.startswith("__"):
            object.__setattr__(self, name, value)
            return
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert(value))

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value

    def to_dict(self):
        def unwrap(value):
            if isinstance(value, Dict):
                return {key: unwrap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            if isinstance(value, tuple):
                return tuple(unwrap(item) for item in value)
            return value

        return unwrap(self)
