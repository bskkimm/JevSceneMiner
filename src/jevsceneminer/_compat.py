"""Compatibility bridges for the original public import paths."""
import importlib
import sys
from types import ModuleType


def alias_module(name, target):
    """Expose one module object through both names, including private helpers."""
    implementation = importlib.import_module(target)
    sys.modules[name] = implementation


class _ForwardingPackage(ModuleType):
    def __getattr__(self, name):
        return getattr(self._implementation, name)

    def __setattr__(self, name, value):
        if name in self._forwarded_names:
            setattr(self._implementation, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._forwarded_names:
            delattr(self._implementation, name)
        else:
            super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._implementation)))


def forward_package(name, target):
    """Keep a package's former module API and mutable settings."""
    package = sys.modules[name]
    package._implementation = importlib.import_module(target)
    # Remember names after deletion so mock.patch can restore the canonical value.
    package._forwarded_names = frozenset(key for key in vars(package._implementation)
                                        if not key.startswith("__"))
    package.__class__ = _ForwardingPackage
    package.__all__ = [key for key in vars(package._implementation) if not key.startswith("_")]
