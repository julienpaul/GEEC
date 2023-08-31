"""
"""

# --- import -----------------------------------
# import from standard lib
import math

# import from other lib
import glm
import numpy as np
import numpy.typing as npt

# import from my project
from geec.utils import (
    cross_product,
    epsilon,
    minus_identity,
    outerCross,
    outerDot,
)


def outward_unit_vector(points: npt.NDArray, centroid: npt.NDArray) -> npt.NDArray:
    """Returns the outward unit vector of the face composed of the given points

    centroid is the centroid of the polyhedron
    """
    # TODO: voir https://stackoverflow.com/a/66141653
    A, B, C = points
    AB = B - A
    AC = C - A
    # normal to the plane containing (A,B,C)
    cross = cross_product(AB, AC)
    norm = np.linalg.norm(cross)

    AD = centroid - A
    dot = np.dot(cross, AD)

    ccw = not dot > 0

    # compute unit outward vector
    norm = np.linalg.norm(cross)
    # reverse vector if not ccw
    un = cross / norm if ccw else -cross

    return un


def get_sign(points: npt.NDArray, un: npt.NDArray, observer: npt.NDArray):
    _points = points - observer
    return np.sign(np.dot(un, _points[0]))


# @jit(fastmath=True, parallel=True)
def get_omega(
    points: list[glm.vec3],
    un: glm.vec3,
    gradient: bool = False,
):
    """Returns the solid angle and its derivatives subtended by the facet,
    at the observer position.

    Assume points are counter-clockwise
    """
    w = 0
    dw = glm.vec3()
    npoints = len(points)
    for i in range(npoints):
        angle = 0
        dangle = glm.vec3()
        #
        p1 = points[i]
        p2 = points[(i + 1) % npoints]
        p3 = points[(i + 2) % npoints]
        #
        dp1 = glm.dot(un, p1)
        if abs(dp1) > epsilon:
            if dp1 > 0:
                p1, p3 = p3, p1
            #
            # v1 = cross_product(p2, p1)
            # v2 = cross_product(p2, p3)
            v1 = glm.cross(p2, p1)
            v2 = glm.cross(p2, p3)

            norm1 = glm.l2Norm(v1)
            norm2 = glm.l2Norm(v2)

            A1 = v1 / norm1
            A2 = v2 / norm2

            B = glm.dot(A1, A2)
            # sign of perp is negative if points are clockwise
            perp = glm.dot(p3, A1)

            # if abs(B) - 1 > 0 and abs(B) - 1 < epsilon:
            #     B = glm.sign(B)
            angle = math.acos(B)
            if perp < 0:
                angle = 2 * np.pi - angle

            if gradient:
                denom = math.sqrt(1 - B * B)
                # [p1 x dpx, p1 x dpy, p1 x dpz]
                cross_dp1 = outerCross(p1, minus_identity)
                cross_dp2 = outerCross(p2, minus_identity)
                cross_dp3 = outerCross(p3, minus_identity)
                # cross_dp1 = cross_product(p1, minus_identity)
                # cross_dp2 = cross_product(p2, minus_identity)
                # cross_dp3 = cross_product(p3, minus_identity)

                dV1 = cross_dp2 - cross_dp1
                dV2 = cross_dp2 - cross_dp3

                dv1 = outerDot(dV1, A1)
                dv2 = outerDot(dV2, A2)

                dA1 = (dV1 * norm1 - glm.outerProduct(dv1, v1)) / (norm1 * norm1)
                dA2 = (dV2 * norm2 - glm.outerProduct(dv2, v2)) / (norm2 * norm2)

                dB = outerDot(dA1, A2) + outerDot(dA2, A1)

                dangle = glm.vec3()
                if denom != 0:
                    dangle = -dB / denom
                    if perp < 0:
                        dangle = -dangle

        dw += dangle
        w += angle

    sign = glm.sign(dp1)
    omega = (w - (npoints - 2) * np.pi) * -sign
    domega = dw * -sign

    return (omega, domega)


# # compute the solid angle
# get_omega = partial(_get_omega, gradient=False)
# # compute the solid angle and its derivatives
# get_domega = partial(_get_omega, gradient=True)


def get_faces_omega(points_list, un_list, gradient: bool = False):
    return [
        get_omega(p, u, gradient) for p, u in zip(points_list, un_list, strict=True)
    ]
