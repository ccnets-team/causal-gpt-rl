"""Compress a Minari 0.5.3 HDF5 file without changing its dataset contract.

Minari's buffer writer stores float32 spatial observations uncompressed.  For
binary occupancy grids this makes a million-transition tier tens of gigabytes.
This utility rewrites numeric arrays with lossless HDF5 gzip, checks every copied
array and attribute, then atomically swaps the file and verifies Minari load-back.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import minari
import numpy as np


def _copy_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _row_block(item, max_bytes=4 * 1024 * 1024):
    """Bound temporary arrays while keeping complete observation rows together."""
    row_bytes = int(np.prod(item.shape[1:])) * item.dtype.itemsize
    return max(1, min(item.shape[0], max_bytes // max(1, row_bytes)))


def _copy_group(source, target):
    _copy_attrs(source, target)
    for key, item in source.items():
        if isinstance(item, h5py.Group):
            _copy_group(item, target.create_group(key))
            continue
        if item.shape and item.dtype.kind in "biuf" and item.nbytes >= 4096:
            block = _row_block(item)
            copied = target.create_dataset(
                key,
                shape=item.shape,
                dtype=item.dtype,
                chunks=(block, *item.shape[1:]),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            for start in range(0, item.shape[0], block):
                stop = min(start + block, item.shape[0])
                copied[start:stop] = item[start:stop]
            _copy_attrs(item, copied)
        else:
            source.copy(item, target, name=key)


def _verify_group(source, target):
    if set(source.keys()) != set(target.keys()):
        raise ValueError(f"HDF5 keys changed at {source.name}")
    if set(source.attrs.keys()) != set(target.attrs.keys()):
        raise ValueError(f"HDF5 attributes changed at {source.name}")
    for key in source.attrs:
        if not np.array_equal(source.attrs[key], target.attrs[key]):
            raise ValueError(f"HDF5 attribute changed at {source.name}: {key}")
    for key, item in source.items():
        copied = target[key]
        if isinstance(item, h5py.Group):
            _verify_group(item, copied)
        elif item.shape != copied.shape or item.dtype != copied.dtype:
            raise ValueError(f"HDF5 shape/dtype changed at {item.name}")
        else:
            if item.shape:
                block = _row_block(item)
                for start in range(0, item.shape[0], block):
                    stop = min(start + block, item.shape[0])
                    if not np.array_equal(item[start:stop], copied[start:stop],
                                          equal_nan=item.dtype.kind in "f"):
                        raise ValueError(f"HDF5 values changed at {item.name}, row {start}")
            elif not np.array_equal(item[()], copied[()]):
                raise ValueError(f"HDF5 values changed at {item.name}")
            if set(item.attrs.keys()) != set(copied.attrs.keys()):
                raise ValueError(f"HDF5 attributes changed at {item.name}")
            for attr in item.attrs:
                if not np.array_equal(item.attrs[attr], copied.attrs[attr]):
                    raise ValueError(f"HDF5 attribute changed at {item.name}: {attr}")


def compress_dataset(dataset_id: str) -> dict:
    before = minari.load_dataset(dataset_id)
    expected = (before.total_episodes, before.total_steps,
                before.observation_space, before.action_space)
    from minari.storage.datasets_root_dir import get_dataset_path

    path = get_dataset_path(dataset_id) / "data" / "main_data.hdf5"
    if not path.is_file():
        raise FileNotFoundError(path)
    temporary = path.with_suffix(".compressed.hdf5")
    backup = path.with_suffix(".original.hdf5")
    if temporary.exists() or backup.exists():
        raise FileExistsError("compression temp/backup already exists; resolve it before retrying")
    try:
        with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
            _copy_group(source, target)
        with h5py.File(path, "r") as source, h5py.File(temporary, "r") as target:
            _verify_group(source, target)
        original_bytes = path.stat().st_size
        compressed_bytes = temporary.stat().st_size
        os.replace(path, backup)
        try:
            os.replace(temporary, path)
            after = minari.load_dataset(dataset_id)
            actual = (after.total_episodes, after.total_steps,
                      after.observation_space, after.action_space)
            if actual != expected:
                raise ValueError(f"Minari contract changed: {expected} -> {actual}")
        except Exception:
            if path.exists():
                os.replace(path, temporary)
            os.replace(backup, path)
            raise
        backup.unlink()
        return {"dataset_id": dataset_id, "original_bytes": original_bytes,
                "compressed_bytes": compressed_bytes,
                "episodes": expected[0], "transitions": expected[1]}
    finally:
        if temporary.exists() and path.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    args = parser.parse_args()
    print(compress_dataset(args.dataset_id), flush=True)


if __name__ == "__main__":
    main()
