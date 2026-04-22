#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as _dt
import math
from pprint import pformat
from typing import Any

import h5py
import numpy as np


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    for u in units:
        f /= 1024.0
        if f < 1024.0:
            return f"{f:.2f} {u}"
    return f"{f:.2f} EiB"


def _indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("utf-8", errors="replace")


def _fmt_scalar(v: Any) -> str:
    if isinstance(v, (np.generic,)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, bytes):
        s = _safe_decode(v)
        if len(s) > 200:
            s = s[:200] + "…"
        return repr(s)
    if isinstance(v, str):
        s = v
        if len(s) > 200:
            s = s[:200] + "…"
        return repr(s)
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return repr(v)


def _fmt_array(a: np.ndarray, *, max_elems: int = 32) -> str:
    # Keep output stable and short for big arrays.
    if a.size == 0:
        return f"array(shape={a.shape}, dtype={a.dtype}, empty)"

    flat = a.ravel()
    show = min(flat.size, max_elems)
    head = flat[:show]

    if flat.size > max_elems:
        suffix = f", … (+{flat.size - max_elems} more)"
    else:
        suffix = ""

    try:
        content = np.array2string(head, threshold=max_elems, edgeitems=math.inf)
    except Exception:
        content = repr(head)

    return f"array(shape={a.shape}, dtype={a.dtype}, head={content}{suffix})"


def _fmt_attr_value(v: Any) -> str:
    # h5py may return scalars, bytes, numpy arrays, or lists.
    if isinstance(v, np.ndarray):
        return _fmt_array(v)
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "[]"
        if len(v) <= 16:
            return pformat([_fmt_scalar(x) for x in v])
        head = [_fmt_scalar(x) for x in v[:16]]
        return pformat(head)[:-1] + f", … (+{len(v) - 16} more)]"
    return _fmt_scalar(v)


def _print_attrs(obj: h5py.Group | h5py.Dataset, *, indent: int) -> None:
    if len(obj.attrs) == 0:
        return
    print(" " * indent + "attrs:")
    for k in sorted(obj.attrs.keys()):
        try:
            v = obj.attrs[k]
        except Exception as e:
            print(" " * (indent + 2) + f"- {k!r}: <error reading attr: {e}>")
            continue
        print(" " * (indent + 2) + f"- {k!r}: {_fmt_attr_value(v)}")


def _describe_dataset(ds: h5py.Dataset) -> str:
    parts: list[str] = []
    parts.append(f"shape={ds.shape}")
    parts.append(f"dtype={ds.dtype}")

    try:
        if ds.chunks is not None:
            parts.append(f"chunks={ds.chunks}")
    except Exception:
        pass

    try:
        comp = ds.compression
        if comp is not None:
            parts.append(f"compression={comp!r}")
    except Exception:
        pass

    try:
        fill = ds.fillvalue
        if fill is not None:
            parts.append(f"fill={_fmt_scalar(fill)}")
    except Exception:
        pass

    try:
        nbytes = int(ds.size) * int(ds.dtype.itemsize)
        parts.append(f"approx_nbytes={_human_bytes(nbytes)}")
    except Exception:
        pass

    return ", ".join(parts)


def _walk(name: str, obj: h5py.Group | h5py.Dataset, *, indent: int) -> None:
    if isinstance(obj, h5py.Group):
        title = "/" if name == "" else name
        print(" " * indent + f"[group] {title}")
        _print_attrs(obj, indent=indent + 2)

        keys = sorted(list(obj.keys()))
        for k in keys:
            try:
                child = obj.get(k, getlink=True)
            except Exception:
                child = None

            # Show links explicitly (common in some HDF5 layouts).
            try:
                link = obj.get(k, getlink=True)
                if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                    print(
                        " " * (indent + 2) + f"[link] {title.rstrip('/')}/{k} -> {link}"
                    )
                    continue
            except Exception:
                pass

            child_obj = obj[k]
            child_name = (title.rstrip("/") + "/" + k) if title != "/" else ("/" + k)
            _walk(child_name, child_obj, indent=indent + 2)

    elif isinstance(obj, h5py.Dataset):
        print(" " * indent + f"[dataset] {name} ({_describe_dataset(obj)})")
        _print_attrs(obj, indent=indent + 2)
    else:
        print(" " * indent + f"[unknown] {name}: {type(obj)}")


def inspect_h5(path: str) -> None:
    with h5py.File(path, "r") as f:
        print(f"HDF5: {path}")
        breakpoint()
        _walk("", f, indent=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an HDF5 file: groups, datasets, shapes, dtypes, and attributes.",
    )
    parser.add_argument("path", help="Path to .h5/.hdf5 file")
    args = parser.parse_args(argv)

    inspect_h5(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
