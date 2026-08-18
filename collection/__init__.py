"""Recording with our runtime, and packaging any recording as a Minari dataset.

Both entry points are imported lazily, because they do not share a dependency:
packaging runs in a `minari` environment and recording in a torch one, and
neither should have to install the other's stack to be importable.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

__all__ = ["CollectionRunner", "build_dataset"]


def __getattr__(name):
    if name == "CollectionRunner":
        from .runner import CollectionRunner

        return CollectionRunner
    if name == "build_dataset":
        from .build_minari import build_dataset

        return build_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
