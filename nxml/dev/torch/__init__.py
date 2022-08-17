from ...core import LazyModule


dist = LazyModule(".dist", "dist", globals(), __package__)
giung2 = LazyModule(".giung2", "giung2", globals(), __package__)
nn = LazyModule(".nn", "nn", globals(), __package__)
xla = LazyModule(".xla", "xla", globals(), __package__)


__all__ = [
    "dist",
    "giung2",
    "nn",
    "xla",
]
