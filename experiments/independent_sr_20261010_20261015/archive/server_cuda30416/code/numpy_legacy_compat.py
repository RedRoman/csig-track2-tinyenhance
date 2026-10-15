"""Restore the dtype catalog required by imgaug 0.4 / pyiqa 0.1.10 imports.

Import this module before pyiqa when running with NumPy 2. It restores only
the removed NumPy 1.x ``sctypes`` catalog. It does not replace numerical
functions, change dtype defaults, seed RNGs, or alter IQA/crop code. Existing
NumPy 1.x catalogs are left untouched. Original scoring sources stay intact.
"""
from __future__ import annotations

import numpy as np


def restore_dtype_catalog() -> dict:
    if "sctypes" in np.__dict__:
        return {"numpy_version": np.__version__, "sctypes_added": False}
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64, np.longdouble],
        "complex": [np.complex64, np.complex128, np.clongdouble],
        "others": [bool, object, bytes, str, np.void],
    }
    return {"numpy_version": np.__version__, "sctypes_added": True}


COMPATIBILITY = restore_dtype_catalog()
