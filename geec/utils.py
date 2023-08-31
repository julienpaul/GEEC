#!/usr/bin/env python3
# utils.py
"""
"""

# --- import -----------------------------------
# import from standard lib
# import from other lib
import glm
import numpy as np
import numpy.typing as npt
from loguru import logger

# import from my project

vector_nan = np.empty(3)
vector_nan.fill(np.nan)

minus_identity = glm.mat3(-1)  # np.diag(np.full(3, -1))
# epsilon = np.finfo(float).eps  # another way: 7./3 - 4./3 -1
epsilon = 1.0e-6  # using glm, we get lower precision


def outerCross(vec: glm.vec3, mat: glm.mat3x3) -> glm.mat3x3:
    """
    vec = [vx, vy, vz]
    mat = [mxx][mxy][mxz]
          [myx][myy][myz]
          [mzx][mzy][mzz]

    outerCross(vec, mat) =
        [vy*mzx - vz*myx][...][...]
        [vz*mxx - vx*mzx][...][...]
        [vx*myx - vy*mxx][...][...]

    """
    c1 = glm.cross(mat[0], vec)
    c2 = glm.cross(mat[1], vec)
    c3 = glm.cross(mat[2], vec)
    return glm.mat3(c1, c2, c3)


def outerDot(mat: glm.mat3x3, vec: glm.vec3) -> glm.vec3:
    """
    mat = [mxx][mxy][mxz]
          [myx][myy][myz]
          [mzx][mzy][mzz]

    vec = [vx]
          [vy]
          [vz]

    outerDot(vec, mat) =
        [mxx*vx + mxy*vy + mxz*vz][myx*vx + myy*vy +..][...]
    """
    _mat = glm.transpose(mat)
    v1 = glm.dot(_mat[0], vec)
    v2 = glm.dot(_mat[1], vec)
    v3 = glm.dot(_mat[2], vec)
    return glm.vec3(v1, v2, v3)


def cross_product(V1: npt.NDArray, V2: npt.NDArray) -> npt.NDArray:
    """cross product of vectors V1 and V2

    return V1xV2
    """
    if V1.ndim == 2:
        return np.array([cross_product(V, V2) for V in V1])
    elif V2.ndim == 2:
        return np.array([cross_product(V1, V) for V in V2])
    elif V1.size == 3 and V2.size == 3:
        # if V1.size == 3 and V2.size == 3:
        a1, a2, a3 = V1
        b1, b2, b3 = V2

        x = a2 * b3 - a3 * b2
        y = a3 * b1 - a1 * b3
        z = a1 * b2 - a2 * b1
        return np.array([x, y, z])
    else:
        msg = "Input vectors must be 3D"
        logger.error(msg)
        raise TypeError(msg)
