"""Compatibility import; implementation: ``jevsceneminer.evidence.compact``."""
from ._compat import alias_module

if __name__ == "__main__":
    import runpy
    runpy.run_module("jevsceneminer.evidence.compact", run_name="__main__", alter_sys=True)
else:
    alias_module(__name__, "jevsceneminer.evidence.compact")
