"""Disk cache for expensive intermediate results."""
import hashlib
import json
import os
import pickle
from pathlib import Path


def settings_fingerprint(info):
    """Short, stable hash of a (JSON-serialisable) dict of settings."""
    text = json.dumps(info, sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


class DiskCache:
    """
    Stores results as <name>__<hash>.pkl in the cache directory.

    The hash is derived from the settings that influence the result, so a
    changed setting automatically leads to a new file instead of silently
    reusing a stale one. With overwrite=True existing files are recomputed.
    """

    def __init__(self, cache_dir, overwrite=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

    def get_or_compute(self, name, key_info, compute_fn):
        path = self.cache_dir / f"{name}__{settings_fingerprint(key_info)}.pkl"

        if path.exists() and not self.overwrite:
            print(f"[cache] loading {path.name}")
            with open(path, "rb") as f:
                return pickle.load(f)

        print(f"[cache] computing {name} (stored as {path.name})")
        result = compute_fn()

        # Write to a temporary file first so an interrupted run leaves no
        # corrupt cache file behind.
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return result
