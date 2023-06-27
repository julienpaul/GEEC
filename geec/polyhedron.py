#!/usr/bin/env python3
# polyhedron.py
"""
    The polyhedron module sets up the polyhedron's classes (Polyhedron, Face, Edge),
    and computes gravity and gravity gradients fields.

    Gravity fields are computed using Singh and Guptasarma method (2001)

    Example usage:
    from geec.polyhedron import Polyhedron
    p = Polyhedron(points)           # initialise Polyhedron object
    p.get_gravity(coord, d, Gc)      # compute gravity fields from the polyhedron with
                                     # density 'd' and gravitational constant 'Gc',
                                     # "seen" from the coordinates 'coord'
    p.plot()                         # plot 3D view of the polyhedron in your browser
"""

# --- import -----------------------------------
# import from standard lib
from itertools import pairwise
from math import acos, log, sqrt

# import from other lib
import numpy as np
from loguru import logger
from numba import jit
from rich.pretty import pprint as rprint
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# import from my project
from geec.utils import (
    cross_product,
    epsilon,
    minus_identity,
)


class Edge:
    """
    Edge object: line segments connecting certain pairs of vertices

    start: starting point's coordinates
    end: ending point's coordinates
    vector: edge's vector
    length: length of edge's vector
    points: points coordinates of the edge
    """

    def __init__(
        self,
        points: tuple[np.ndarray, np.ndarray],
        simplex: tuple[np.ndarray, np.ndarray],
        twinface: int,
    ) -> None:
        """
        polyhedron: Polyhedron instance
        simplex: points indices (from Polyhedron points list) of the Edge
        twinface: index of the face containing the edge's twin
        """
        logger.trace("initialise Edge")
        self._simplex = np.array(simplex)
        self.points = points
        self.start = self.points[0]
        self.end = self.points[1]
        self.vector = self.end - self.start  # Lj
        self.length = np.linalg.norm(self.vector)  # lj
        self._twinface = twinface
        self._twin = None
        # depending on the station, will be computed later
        self._pqr = None
        self._dpqr = None

    def reset(self):
        """reset value for future computing"""
        self._pqr = None
        self._dpqr = None

    def __rich_repr__(self):
        """rich pretty print fields"""
        yield "start", self.start
        yield "end", self.end
        yield "vector", self.vector
        yield "length", self.length
        yield "points", self.points

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    # def pqr(self) -> np.ndarray:
    @property
    def pqr(self) -> np.ndarray | None:
        return self._pqr

    @property
    def dpqr(self) -> np.ndarray | None:
        return self._dpqr

    def set_twin(self, twin) -> None:
        """set up the twin of the edge

        Each edge is part of two Faces, we call twin the edge of the other Face.
        Note that start and end point of the twin are reversed.
        """
        if isinstance(twin, Edge):
            self._twin = twin
        else:
            raise TypeError("'twin' must be Edge object")

    # def _set_pqr(self, pqr: np.ndarray) -> None:
    def _set_pqr(self, pqr: np.ndarray | None) -> None:
        if isinstance(pqr, np.ndarray):
            self._pqr = pqr
        else:
            raise TypeError("'pqr' must be numpy array")

    def _set_dpqr(self, dpqr: np.ndarray | None) -> None:
        if isinstance(dpqr, np.ndarray):
            self._dpqr = dpqr
        else:
            raise TypeError("'dpqr' must be numpy array")

    # def _set_twin_pqr(self, pqr: np.ndarray | None, dpqr: np.ndarray | None) -> None:
    def _set_twin_pqr(self) -> None:
        """assign value to twin edge

        Note: twins have opposite values.
        """
        if isinstance(self._twin, Edge):
            twin = self._twin
            if self._pqr is not None:
                twin._set_pqr(-self._pqr)
            if self._dpqr is not None:
                twin._set_dpqr(-self._dpqr)
        else:
            raise TypeError("'twin' undefined")

    @staticmethod
    @jit(nopython=True)
    def _line_integral_along_edge(
        cj: float,
        Lj: np.ndarray,
        lj: float,
        Rj: np.ndarray,
        rj: float,
        b2: float,
        bj: float,
        gradient: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Line integral along edge"""
        Ll = Lj / lj
        if abs(cj) > epsilon:  # cj != 0
            # observation point and edge are not in line
            l2 = lj * lj
            r2 = rj * rj
            _sqrt = sqrt(l2 + bj + r2)

            t = (_sqrt + lj + b2) / cj

            integral = log(t)

        else:
            # observation point and edge are in line
            t = abs(lj - rj) / rj

            integral = log(t)

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
            DPQR = Ll * (DT / t)[:, np.newaxis]
        else:
            DPQR = None

        return (pqr, DPQR)

    def _get_ccw_line_integrals(self, obs: np.ndarray, gradient: bool = False) -> None:
        """
        compute the line integral of vectors (i/r), (j/r), (k/r),
        taken around the egde of the polygon in a counterclockwise direction

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        if self._pqr is None:
            # use shifted coordinates, see Polyhedron.get_gravity
            R1, R2 = (self.points[0] - obs, self.points[1] - obs)
            r1, r2 = (np.linalg.norm(R1), np.linalg.norm(R2))

            chsgn = 1  # if origin,p1 & p2 are on a st line
            if r1 > r2 and np.dot(R1, R2) / (r1 * r2) == 1:  # p1 farther than p2
                R1 = R2  # interchange p1,p2
                r1 = r2
                chsgn = -1

            Lj = self.vector
            lj = self.length

            # bj = 2 Rj.Lj
            bj = 2 * np.dot(R1, Lj)
            b2 = bj / lj / 2
            if abs(r1 + b2) < epsilon:  # r1 + b2 == 0
                # observation point and edge are in line
                Lj, bj = -Lj, -bj
                b2 = bj / lj / 2

            # cj = R1 + bj/(2*lj)
            cj = r1 + b2

            (pqr, dpqr) = self._line_integral_along_edge(
                cj, Lj, lj, R1, r1, b2, bj, gradient
            )

            # change sign of I if p1,p2 were interchanged
            self._pqr = pqr * chsgn
            self._dpqr = dpqr
            # assign twin value
            self._set_twin_pqr()


class Face:
    """
    Face object: facet of the polyhedron

    points: list of vertices (corner points)
    neighbors: neighboring faces of the current face
    simplex: points indices of the face
    un: unit outward vector
    edges: list of Edges of the face
    """

    def __init__(
        self,
        # polyhedron: "Polyhedron",
        # polypoints: np.ndarray,  # "Polyhedron",
        # polynorm: np.ndarray,  # "Polyhedron",
        points: np.ndarray,
        simplex: np.ndarray,
        un: np.ndarray,
        neighbors: np.ndarray,
        simplices: np.ndarray,
    ) -> None:
        """
        polyhedron: Polyhedron instance
        simplex: points indices (from Polyhedron points list) of the Face
        un: unit outward vector
        neighbors: indices (from Polyhedron faces list) of the neighbors faces
        simplices: points indices (from Polyhedron points list) of the neighbors faces
        """
        logger.trace("initialise Face")
        self.points = points
        self.simplex = simplex
        self.npoints = len(self.simplex)
        self.un = un
        self.neighbors = neighbors
        self.neighbors_simplices = simplices

        self.edges = self._get_edges()
        self.nedges = len(self.edges)

        # depending on the station, will be computed later
        self._dp1 = None
        self._sign = None
        self._omega = None
        self._domega = None
        self._pqr = None
        self._dpqr = None
        # self._pqr = vector_nan
        self._g = None

    def reset(self):
        """reset values for future computing"""
        self._dp1 = None
        self._sign = None
        self._omega = None
        self._domega = None
        self._pqr = None
        self._dpqr = None
        for edge in self.edges:
            edge.reset()

    def __rich_repr__(self):
        """rich pretty print fields"""
        yield "points", self.points
        yield "simplex", self.simplex
        yield "un", self.un
        yield "neighbors", self.neighbors
        yield "edges", self.edges

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    # def __repr__(self):
    #     return (
    #         str(self.__class__.__name__)
    #         + "(\n"
    #         + f"\tpoints: {self.points}\n"
    #         + f"\tsimplex: {self.simplex}\n"
    #         + f"\tneighbors: {self.neighbors}\n"
    #         + f"\tun: {self.un}\n"
    #         # + f"\tedges: \n{self.edges}\n"
    #         + "\tedges: ["
    #         + "".join("\n%s" % e for e in self.edges)
    #         + "]"
    #         # + pformat(vars(self), indent=4, depth=2, sort_dicts=False)
    #         + ")"
    #     )
    #     "\n".join("%s:%s" % (k, v) for k, v in self.maze.items())

    def _add_pqr(self, pqr: np.ndarray | None) -> None:
        """ """
        if self._pqr is not None:
            self._pqr += pqr
        else:
            self._pqr = pqr

    def _add_dpqr(self, dpqr: np.ndarray | None) -> None:
        """ """
        if self._dpqr is not None:
            self._dpqr += dpqr
        else:
            self._dpqr = dpqr

    def _get_edges(self) -> list[Edge]:
        """set up edges of the face

        Warning points should already be sorted counterclockwise
        """
        _points = np.insert(self.points, self.npoints, self.points[0], axis=0)
        _simplex = np.insert(self.simplex, self.npoints, self.simplex[0], axis=0)
        return [
            Edge(p, s, self._get_twinface(s))
            for p, s in zip(pairwise(_points), pairwise(_simplex), strict=True)
        ]

        # return [
        #     Edge(self.polypoints, self.polynorm, p, s, self._get_twinface(s))
        #     for p, s in zip(pairwise(_points), pairwise(_simplex))
        # ]
        # _simplex = np.insert(self.simplex, self.npoints, self.simplex[0], axis=0)
        # return [
        #     Edge(self._polyhedron, s, self._get_twinface(s))
        #     for s in pairwise(_simplex)
        # ]

    def _get_twinface(self, pair: tuple[int, int]) -> int:
        """find index of the face of the edge's twin."""
        contained = [all(p in s for p in pair) for s in self.neighbors_simplices]
        return self.neighbors[contained][0]

    def _get_dot_point1(self, obs: np.ndarray) -> None:
        """scalar product of face's unit outward vector and vector OA,
        where
          O is the observation point
          A is the first corner of the face

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        # use shifted origin
        # p1 = self._polyhedron.points[self.simplex][0]
        # p1 = self.polypoints[self.simplex][0]
        p1 = self.points[0] - obs
        self._dp1 = np.dot(self.un, p1)

    def _get_sign(self) -> None:
        """sign of scalar product of face's unit outward vector and vector OA,
        where
          O is the observation point
          A is the first corner of the face

        sign > 0 : observation point is "above" of the face
        sign < 0 : observation point is "below" of the face
        """
        if self._dp1 is not None:
            self._sign = np.sign(self._dp1)
        else:
            raise NotImplementedError("'dp1' is undefined")

    # @profile
    def _get_omega(self, obs: np.ndarray, gradient: bool = False) -> None:
        """compute solid angle subtended by the face at the observation point

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        if self._dp1 is None:
            raise NotImplementedError("'dp1' is undefined")
        if self._sign is None:
            raise NotImplementedError("'sign' is undefined")

        self._omega = 0
        self._domega = np.zeros(3)
        if abs(self._dp1) > epsilon:
            points = self.points - obs  # [p1,p2,p3]
            dots = np.dot(self.un, points.T)  # [un.p1, un.p2, un.p3]
            cross = [
                cross_product(points[i], points[(i + 1) % self.npoints])
                for i in range(self.npoints)
            ]  # [p1xp2, p2xp3, p3xp1]
            norm = np.linalg.norm(cross, axis=1)
            # unit cross vector
            unitv = cross / norm[:, np.newaxis]  # [A1, A2, A3]

            if gradient:
                # [[p1 x dpx, p1 x dpy, p1 x dpz],[p2 x dpx,...],...]
                cross_dp = cross_product(points, minus_identity)
                # cross_dp = np.array(
                #     [
                #         [cross_product(points[i], minus_identity[j]) for j in range(3)]
                #         for i in range(3)
                #     ]
                # )

            w = 0
            dw = 0
            for i in range(self.npoints):
                inout = dots[i]
                (A1, A2) = self._get_A1A2(i, inout, self.npoints, unitv)
                perp = self._get_perp(i, inout, self.npoints, points, A1)
                b = self._get_b(A1, A2)

                angle = self._get_angle(i, inout, b, perp)
                w += angle
                if gradient:
                    dangle = self._get_dangle(
                        i,
                        inout,
                        b,
                        perp,
                        A1,
                        A2,
                        self.npoints,
                        cross_dp,
                        cross,
                        norm,
                        unitv,
                    )
                    dw += dangle

            w -= (self.npoints - 2) * np.pi
            self._omega = -self._sign * w
            self._domega = -self._sign * dw

    @staticmethod
    # @jit(nopython=True)
    def _get_A1A2(
        i: int, inout: float, npoints: int, unitv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ """
        A1, A2 = -unitv[i], unitv[(i + 1) % npoints]
        if inout > 0:
            A1, A2 = A2, A1
        return (A1, A2)

    @staticmethod
    # @jit(nopython=True)
    def _get_perp(
        i: int, inout: float, npoints: int, points: np.ndarray, A1: np.ndarray
    ) -> float:
        """ """
        # Check if face is seen from inside
        if inout == 0:
            return 1
        else:
            p3 = points[(i + 2) % npoints]  # p3
            # if inout>0: face seen from inside
            if inout > 0:
                p3 = points[i]  # p1

            # sign of perp is negative if points are clockwise
            return np.dot(p3, A1)

    @staticmethod
    # @jit(nopython=True)
    def _get_b(A1: np.ndarray, A2: np.ndarray) -> float:
        return np.dot(A1, A2)

    @staticmethod
    # @jit(nopython=True)
    def _get_angle(i: int, inout: float, b: float, perp: float) -> float:
        if inout == 0:
            angle = 0
        else:
            angle = acos(b)
            if perp < 0:
                angle = 2 * np.pi - angle
        return angle

    @staticmethod
    # @jit(nopython=True)
    def _get_dangle(
        i: int,
        inout: float,
        b: float,
        perp: float,
        A1: np.ndarray,
        A2: np.ndarray,
        npoints: int,
        cross_dp: np.ndarray,
        cross: list[np.ndarray],
        norm: np.ndarray,
        unitv: np.ndarray,
    ) -> float:
        dangle = np.zeros(3)
        # local variables are access much more efficiently
        npdot = np.dot

        cdp1 = -cross_dp[i]
        cdp2 = cross_dp[(i + 1) % npoints]
        cdp3 = -cross_dp[(i + 2) % npoints]

        V1, V2 = -cross[i], cross[(i + 1) % npoints]
        v1, v2 = norm[i], norm[(i + 1) % npoints]
        uV1, uV2 = -unitv[i], unitv[(i + 1) % npoints]

        if inout > 0:
            cdp1, cdp3 = cdp3, cdp1
            V1, V2 = V2, V1
            v1, v2 = v2, v1
            uV1, uV2 = uV2, uV1

        dV1 = cdp1 + cdp2
        dV2 = cdp3 + cdp2

        dv1 = npdot(dV1, uV1)
        dv2 = npdot(dV2, uV2)
        # dv1 = np.dot(dV1, V1) / v1
        # dv2 = np.dot(dV2, V2) / v2

        DA1 = (dV1 * v1 - V1 * dv1[:, np.newaxis]) / (v1 * v1)
        DA2 = (dV2 * v2 - V2 * dv2[:, np.newaxis]) / (v2 * v2)

        DB = npdot(DA1, A2) + npdot(DA2, A1)
        denom = sqrt(1 - b * b)

        if denom != 0:
            dangle = -DB / denom
            if perp < 0:
                dangle = -dangle

        return dangle

    # def _get_omega(self, obs: np.ndarray, gradient: bool = False) -> None:
    #     """compute solid angle 'omega' subtended by the face at the observation point

    #     Note: observation points' coordinates are setup through Polyhedron instance
    #     """
    #     if self._dp1 is None:
    #         raise NotImplementedError("'dp1' is undefined")
    #     if self._sign is None:
    #         raise NotImplementedError("'sign' is undefined")
    #     # if self._domega is None:
    #     self._omega = 0
    #     self._domega = np.zeros(3)

    #     # if abs(self._dp1) == 0:
    #     # if self._dp1 == 0:
    #     #     self._omega = 0
    #     #     # self._domega = np.zeros(3)
    #     # else:
    #     # if self._dp1 != 0:
    #     if abs(self._dp1) > epsilon:
    #         w = 0
    #         # use shifted origin
    #         # points = self._polyhedron.points[self.simplex]  # [p1,p2,p3]
    #         # points = self.polypoints[self.simplex]  # [p1,p2,p3]
    #         points = self.points - obs  # [p1,p2,p3]
    #         dots = np.dot(self.un, points.T)  # [un.p1, un.p2, un.p3]
    #         cross = [
    #             cross_product(points[i], points[(i + 1) % self.npoints])
    #             for i in range(self.npoints)
    #         ]  # [p1xp2, p2xp3, p3xp1]
    #         norm = np.linalg.norm(cross, axis=1)
    #         # norm = norm[np.newaxis, :]
    #         # unit cross vector
    #         # unitv = cross / norm.T  # [A1, A2, A3]
    #         unitv = cross / norm[:, np.newaxis]  # [A1, A2, A3]

    #         if gradient:
    #             # [[p1 x dpx, p1 x dpy, p1 x dpz],[p2 x dpx,...],...]
    #             cross_dp = cross_product(points, minus_identity)
    #             # cross_dp = np.array(
    #             #     [
    #             #         [cross_product(points[i], minus_identity[j]) for j in range(3)]
    #             #         for i in range(3)
    #             #     ]
    #             # )

    #         for i in range(self.npoints):
    #             # finds the angle between planes O-p1-p2 and O-p2-p3,
    #             # p1,p2,p3 are 3 points, taken in ccw order as seen from origin O

    #             # Check if face is seen from inside
    #             inout = dots[i]
    #             A1, A2 = -unitv[i], unitv[(i + 1) % self.npoints]
    #             b = np.dot(A1, A2)
    #             if inout == 0:
    #                 angle = 0
    #                 perp = 1
    #             else:
    #                 p3 = points[(i + 2) % self.npoints]  # p3
    #                 # if inout>0: face seen from inside
    #                 if inout > 0:
    #                     A1, A2 = A2, A1
    #                     p3 = points[i]  # p1
    #                     # p1, p3 = p3, p1

    #                 # sign of perp is negative if points are clockwise
    #                 perp = np.dot(p3, A1)
    #                 angle = math.acos(b)
    #                 # print(f"perp: {perp, p3, A1}")
    #                 if perp < 0:
    #                     angle = 2 * np.pi - angle
    #                 # print(f"angle: {angle}")
    #                 w += angle

    #             if gradient:
    #                 dw = 0

    #                 cdp1, cdp2, cdp3 = (
    #                     -cross_dp[i],
    #                     cross_dp[(i + 1) % self.npoints],
    #                     -cross_dp[(i + 2) % self.npoints],
    #                 )

    #                 V1, V2 = -cross[i], cross[(i + 1) % self.npoints]
    #                 v1, v2 = norm[i], norm[(i + 1) % self.npoints]

    #                 if inout > 0:
    #                     cdp1, cdp3 = cdp3, cdp1
    #                     V1, V2 = V2, V1
    #                     v1, v2 = v2, v1

    #                 # cdp1 = cross_product(minus_identity, p1)
    #                 # cdp2 = cross_product(minus_identity, p2)
    #                 # cdp3 = cross_product(minus_identity, p3)

    #                 dV1 = cdp1 + cdp2
    #                 dV2 = cdp3 + cdp2

    #                 dv1 = np.dot(dV1, V1) / v1
    #                 dv2 = np.dot(dV2, V2) / v2

    #                 DA1 = (dV1 * v1 - V1 * dv1[:, np.newaxis]) / (v1 * v1)
    #                 DA2 = (dV2 * v2 - V2 * dv2[:, np.newaxis]) / (v2 * v2)

    #                 DB = np.dot(DA1, A2) + np.dot(DA2, A1)
    #                 denom = math.sqrt(1 - b * b)

    #                 if denom != 0:
    #                     dw = -DB / denom
    #                     if perp < 0:
    #                         dw = -dw

    #                 self._domega += dw

    #         w -= (self.npoints - 2) * np.pi
    #         self._omega = -self._sign * w
    #         self._domega = -self._sign * self._domega

    def _get_ccw_line_integrals(self, obs: np.ndarray, gradient: bool = False) -> None:
        """compute pqr for the current face

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        for edge in self.edges:
            # edge._get_ccw_line_integrals(obs)
            edge._get_ccw_line_integrals(obs, gradient)
            self._add_pqr(edge.pqr)
            self._add_dpqr(edge.dpqr)

    def get_gravity(
        self, obs: np.ndarray, density: float, Gc: float, gradient: bool = False
    ) -> np.ndarray:
        """compute gravity from the current face with density 'density' and
        gravitational constant 'Gc' seen from the observation points

        calculate gravity in mGal unit and gravity gradient in E

        Note: observation points' coordinates are setup through Polyhedron instance

        density: []
        Gc: Gravitational constant []
        """
        # if distance to face is non-zero
        self._get_dot_point1(obs)
        self._get_sign()
        self._get_omega(obs, gradient)
        self._get_ccw_line_integrals(obs, gradient)

        if self._dp1 is None:
            raise NotImplementedError("'dp1' is undefined")
        if self._omega is None:
            raise NotImplementedError("'omega' is undefined")
        if self._domega is None:
            raise NotImplementedError("'domega' is undefined")

        # if np.array_equal(self._pqr, vector_nan, equal_nan=True):
        if self._pqr is None:
            raise NotImplementedError("'pqr' is undefined")
        if gradient and self._dpqr is None:
            raise NotImplementedError("'dpqr' is undefined")

        factor = -density * Gc * 1e5
        (L, M, N) = self.un
        (P, Q, R) = self._pqr

        # if self._dp1 != 0:
        _gx = L * self._omega + N * Q - M * R
        _gy = M * self._omega + L * R - N * P
        _gz = N * self._omega + M * P - L * Q
        gx = self._dp1 * factor * _gx
        gy = self._dp1 * factor * _gy
        gz = self._dp1 * factor * _gz
        # else:
        #     gx, gy, gz = 0, 0, 0

        self._g = np.array([gx, gy, gz])

        if gradient:
            px, qx, rx = self._dpqr[0]
            py, qy, ry = self._dpqr[1]
            pz, qz, rz = self._dpqr[2]

            # derivative value
            dux, duy, duz = -self.un

            factor = factor * 1e4  # -density * Gc * 1e9
            txx = factor * (
                self._dp1 * (L * self._domega[0] + N * qx - M * rx) + dux * _gx
            )
            txy = factor * (
                self._dp1 * (L * self._domega[1] + N * qy - M * ry) + duy * _gx
            )
            txz = factor * (
                self._dp1 * (L * self._domega[2] + N * qz - M * rz) + duz * _gx
            )

            # tyx = txy
            # tyx = factor * (
            #     self._dp1 * (M * self._domega[0] + L * rx - N * px) + dux * _gy
            # )
            tyy = factor * (
                self._dp1 * (M * self._domega[1] + L * ry - N * py) + duy * _gy
            )
            tyz = factor * (
                self._dp1 * (M * self._domega[2] + L * rz - N * pz) + duz * _gy
            )

            # tzx =txz
            # tzx = factor * (
            #     self._dp1 * (N * self._domega[0] + M * px - L * qx) + dux * _gz
            # )
            # tzy =tyz
            # tzy = factor * (
            #     self._dp1 * (N * self._domega[1] + M * py - L * qy) + duy * _gz
            # )
            tzz = factor * (
                self._dp1 * (N * self._domega[2] + M * pz - L * qz) + duz * _gz
            )

            self._tensor = np.array([[txx, txy, txz], [txy, tyy, tyz], [txz, tyz, tzz]])
        else:
            self._tensor = np.zeros(3)

        return (self._g, self._tensor)


class Polyhedron:
    """
    Polyhedron object

    points: list of vertices (corner points) [x,y,z]
    faces: list of Faces (two-dimensional polygons)

    Sharing variables between different instances of different classes:
    https://stackoverflow.com/a/51992639
    """

    def __init__(self, points: np.ndarray) -> None:
        """initialise Polyhedron object

        points: list of vertices (corner points) [x,y,z]
        """
        logger.trace("initialise Polyhedron")
        # check points list of coordinates
        # TODO: check np.ndarray shape (N,3)
        if not isinstance(points, np.ndarray):
            raise KeyError("'points' must be a numpy array")
        else:
            self.points = points

        self.npoints = len(points)
        self._norm = np.empty(self.npoints)
        self._norm.fill(np.nan)

        self.faces = self._get_faces()
        self.nfaces = len(self.faces)
        # self._G = np.zeros(3)
        # self.edges = [e for f in self.faces for e in f.edges]
        # self.nedges = len(self.edges)

    def __rich_repr__(self):
        """rich pretty print fields"""
        yield "faces", self.faces
        yield "points", self.points

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    def _get_faces(self) -> list[Face]:
        """Warning"""
        hull = ConvexHull(self.points)
        centroid = np.mean(self.points[hull.vertices, :], axis=0)
        npdot = np.dot

        faces = []
        # check simplex points are in counterclockwise
        for s, n in zip(hull.simplices, hull.neighbors, strict=True):
            # get 3 first points of simplex
            A, B, C = hull.points[s][0:3]
            AB = B - A
            AC = C - A
            # normal to the plane containing (A,B,C)
            cross = cross_product(AB, AC)

            # look for a point inside the hull,
            # and such that DA is not parallel to the normal of the ABC plane
            D = centroid
            D_in_hull = self._in_hull(hull.points, D)
            AD = D - A
            dot = npdot(cross, AD)
            _niter = 0
            while dot == 0 and not D_in_hull:
                _niter += 1
                # if centroid not ok, look around for another point
                _std = np.std(points, axis=0)
                _round = np.round(_std, 1) * 10
                _low, _high = -_round, _round
                _rand = np.random.randint(_low, _high, size=(3)) / 10

                D = centroid + _rand
                D_in_hull = self._in_hull(hull.points, D)
                AD = D - A
                dot = npdot(cross, AD)
                if _niter > 10:
                    raise RuntimeError("Too many iteration to find point inside hull")

            # dot > 0: A,B,C are not counterclockwise
            # dot < 0: A,B,C are already counterclockwise
            ccw = not dot > 0

            # compute unit outward vector
            norm = np.linalg.norm(cross)
            if ccw:
                un = cross / norm
            else:
                # reverse vector
                un = -cross / norm
                # reverse last points, to make them counterclockwise
                s[1::] = list(reversed(s[1::]))

            faces.append(
                Face(
                    # self,  # Polyhedron instance
                    # self.points,  # Polyhedron instance
                    # self._norm,  # Polyhedron instance
                    points=hull.points[s],
                    simplex=s,
                    un=un,
                    neighbors=n,
                    simplices=hull.simplices[n],
                )
            )

        self._get_edge_twin(faces)

        return faces

    def _in_hull(self, points, x) -> bool:
        """
        check if point 'x' in inside the convex hull of 'points'

        source: https://stackoverflow.com/a/43564754
        """
        n_points = len(points)
        c = np.zeros(n_points)
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success

    def _get_edge_twin(self, faces: list[Face]) -> None:
        """find all edges' twin"""
        for face in faces:
            for edge in face.edges:
                if edge._twin is None:
                    facetwin = faces[edge._twinface]
                    for twin in facetwin.edges:
                        if np.array_equal(edge.start, twin.end) and np.array_equal(
                            edge.end, twin.start
                        ):
                            # set up the twin of 'edge'
                            edge.set_twin(twin)
                            twin.set_twin(edge)
                            break

    # def shift_origin(self, coord):
    #     """change points origin to 'coord'

    #     return points = points - coord
    #     """
    #     self.points -= coord

    # def change_norm(self, nan=False):
    #     """compute norm of points' vectors.

    #     optionally, force all value to NaN
    #     """
    #     if nan:
    #         self._norm.fill(np.nan)
    #     else:
    #         self._norm[:] = np.linalg.norm(self.points, axis=1)

    # def set_origin(self, coord):
    #     """set up new cooridnates' origin 'coord'

    #     coord: new origin
    #     """
    #     # shift origin to new origin
    #     self.shift_origin(coord)
    #     # compute norm of each points on new axis
    #     self.change_norm()

    # def reset_origin(self, coord):
    #     """shift back to former origin, and assign nan to norm

    #     coord: origin to shift from
    #     """
    #     # shift origin back
    #     self.shift_origin(-coord)
    #     # assign nan to norm of each points
    #     self.change_norm(nan=True)

    # def setup(self, coord):
    #     """set up new origin 'coord', and assign G to zeros"""
    #     # shift origin
    #     self.set_origin(coord)
    #     # assign G
    #     G = np.zeros(3)
    #     return G

    def reset(self, coord):
        """reset origin and all parameters"""
        # shift back origin
        # self.reset_origin(coord)
        for face in self.faces:
            face.reset()

    def get_gravity(
        self, coord: np.ndarray, density: float, Gc: float, gradient: bool = False
    ) -> np.ndarray:
        """
        compute gravity from polyhedron with density 'density' and
        gravitational constant 'Gc' seen from the coordinates 'coord'

        coord: observation point's coordinates
        density: []
        Gc: Gravitational constant []
        gradient: optionaly compue gradient gravity fields
        """

        # setup G array, and shift back origin
        G = np.zeros(3)
        # self.set_origin(coord)
        T = np.zeros((3, 3))

        for face in self.faces:
            (g, t) = face.get_gravity(coord, density, Gc, gradient)
            G += g
            T += t

        # reset intermediate variables, and shift back origin
        # self.reset_origin(coord)
        self.reset(coord)

        return (G, T)

    def plot(self):
        """plot 3D view of the polyhedron in the browser"""
        import plotly.graph_objects as go

        edges = [e for f in self.faces for e in f.edges]

        fig = go.Figure()
        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[edge.start[0], edge.end[0]],
                    y=[edge.start[1], edge.end[1]],
                    z=[edge.start[2], edge.end[2]],
                    mode="lines+markers",
                )
            )
        fig.update_layout(showlegend=False)
        fig.show()


if __name__ == "__main__":
    cube = [
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
        (-0.5, -0.5, -1.0),
        (0.5, -0.5, -1.0),
        (0.5, 0.5, -1.0),
        (-0.5, 0.5, -1.0),
    ]

    points = np.array(cube)
    p = Polyhedron(points)
    p.plot()
    print(p)
