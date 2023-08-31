"""
"""

# --- import -----------------------------------
# import from standard lib
import math

# import from other lib
import glm

# import from my project
from geec.utils import epsilon


def _line_integral_along_edge(
    cj: float,
    Lj: glm.vec3,
    lj: float,
    Rj: glm.vec3,
    rj: float,
    b2: float,
    bj: float,
    gradient: bool = False,
) -> tuple[glm.vec3, glm.mat3x3 | None]:
    """Line integral along edge"""
    Ll = Lj / lj
    if abs(cj) > epsilon:  # cj != 0
        # observation point and edge are not in line
        l2 = lj * lj
        r2 = rj * rj
        _sqrt = math.sqrt(l2 + bj + r2)

        t = (_sqrt + lj + b2) / cj

        integral = math.log(t)

    else:
        # observation point and edge are in line
        t = abs(lj - rj) / rj

        integral = math.log(t)

    pqr = integral * Ll

    if gradient:
        Rr = Rj / rj

        if abs(cj) > epsilon:  # cj != 0
            u = t * cj
            v = cj
            DU = -((Lj + Rj) / _sqrt + Ll)
            DV = -(Rr + Ll)

        else:
            u = abs(lj - rj)
            v = rj
            DU = ((lj - rj) / u) * Rr
            DV = -Rr

        DT = (v * DU - u * DV) / (v * v)
        DPQR = glm.outerProduct((DT / t), Ll)
    else:
        DPQR = None

    return (pqr, DPQR)


def get_ccw_line_integrals(
    points: list[glm.vec3],
    gradient: bool = False,
    # points: list[glm.vec3], observer: glm.vec3, gradient: bool = False,
) -> tuple[glm.vec3, glm.mat3x3 | None]:
    """
    compute the line integral of vectors (i/r), (j/r), (k/r),
    taken around the egde of the polygon in a counterclockwise direction

    Note: observation points' coordinates are setup through Polyhedron instance
    """
    R1, R2 = points
    r1, r2 = (glm.l2Norm(R1), glm.l2Norm(R2))

    Lj = R2 - R1
    lj = glm.l2Norm(Lj)

    chsgn = 1  # if origin,p1 & p2 are on a st line
    if r1 > r2 and glm.dot(R1, R2) / (r1 * r2) == 1:  # p1 farther than p2
        R1 = R2  # interchange p1,p2
        r1 = r2
        chsgn = -1

    # bj = 2 Rj.Lj
    bj = 2 * glm.dot(R1, Lj)
    b2 = bj / lj / 2
    if abs(r1 + b2) < epsilon:  # r1 + b2 == 0
        # observation point and edge are in line
        Lj, bj = -Lj, -bj
        b2 = bj / lj / 2

    # cj = R1 + bj/(2*lj)
    cj = r1 + b2

    pqr, dpqr = _line_integral_along_edge(cj, Lj, lj, R1, r1, b2, bj, gradient)

    # change sign of I if p1,p2 were interchanged
    pqr = pqr * chsgn

    return (pqr, dpqr)


# # compute the solid angle
# get_ccw_line_integrals = partial(_get_ccw_line_integrals, gradient=False)
# # compute the solid angle and its derivatives
# get_ccw_line_integrals_derivatives = partial(_get_ccw_line_integrals, gradient=True)


def get_edges_pqr(points_list, gradient: bool = False):
    return [get_ccw_line_integrals(p, gradient) for p in points_list]
