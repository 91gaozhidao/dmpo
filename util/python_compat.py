"""Small runtime compatibility helpers for third-party packages."""

import collections
from collections import abc as collections_abc


def apply_collections_abc_compat():
    """Backfill pre-3.10 ``collections`` aliases used by older deps."""

    aliases = {
        "Mapping": collections_abc.Mapping,
        "MutableMapping": collections_abc.MutableMapping,
        "Sequence": collections_abc.Sequence,
    }
    for name, value in aliases.items():
        if not hasattr(collections, name):
            setattr(collections, name, value)
