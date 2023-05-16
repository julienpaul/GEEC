#!/usr/bin/env python3
# utils.py
"""
"""

# --- import -----------------------------------
# import from standard lib
# import from other lib
import numpy as np

# import from my project

vector_nan = np.empty(3)
vector_nan.fill(np.nan)


def cross_product(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """cross product of vectors V1 and V2

    return V1xV2
    """
    if len(V1) == 3 and len(V2) == 3:
        a1, a2, a3 = V1
        b1, b2, b3 = V2

        x = a2 * b3 - a3 * b2
        y = a3 * b1 - a1 * b3
        z = a1 * b2 - a2 * b1
        return np.array([x, y, z])
    else:
        raise TypeError("Input vectors must be 3D")
