#!/usr/bin/env python3
# station.py
"""
    The station module sets up the Station's class,
    and calculates the resulting gravity fields of a polyhedron.

    Example usage:
    from geec.station import Station
    s = Station(points)         # initialise Station object
    s.compute_gravity(p, d, Gc) # compute gravity from polyhedron 'p' with density 'd',
                                # and gravitational constant Gc at the station point
    print(s.G)                  # print Gravity fields at the station point
"""
# TODO
# check if edge already computed !!!

# --- import -----------------------------------
# import from standard lib
import math

# import from other lib
import numpy as np
from loguru import logger
from rich.pretty import pprint as rprint

# import from my project
from geec.polyhedron import Edge, Face, Polyhedron


class Station:
    """
    Station object: gravity fields computation point

    """

    def __init__(self, coord: np.ndarray) -> None:
        """initialise Station object

        coord: station's coordinates
        """

        logger.trace("initialise Station")
        # check points list of coordinates
        if not isinstance(coord, np.ndarray):
            raise KeyError("'coord' must be a numpy array.")

        self.coord = coord
        self.G = np.full(3, 0)

    def __rich_repr__(self):
        yield "coord", self.coord
        yield "G", self.G

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    def compute_gravity(
        self, polyhedron: Polyhedron, density: float, Gc: float
    ) -> None:
        """
        compute gravity from polyhedron with density 'density' and
        gravitational constant 'Gc' at the station point

        density: []
        Gc: Gravitational constant []
        """
        logger.trace("compute gravity")
        # if self.obs:
        # self.obs = np.insert(self.obs, self.nobs, obs, axis=0)
        # self.nobs += 1
        # else:
        # self.obs = np.array([obs])
        # self.nobs = 1

        for face in polyhedron.faces:
            # logger.info(f"\nworking on Face: {face.simplex}")
            f = _Face(self.coord, face)
            f.get_dot_point1(self.coord)
            f.get_sign()
            f.get_omega(self.coord)
            f.get_ccw_line_integrals(self.coord)
            g = f.get_gravity(density, Gc)
            self.G = self.G + g


class _Face:
    """
    _Face object: extend Polyhedron's Face object with parameters link to station

    self.coord: station's coordinate
    self.obj: Polyhedron Face object
    self.dp1: scalar product of face's unit outward vector and P1
    self.sign: sign of dp1
    self.omega: solid angle subtended by the face at the observation point
    self.pqr: line integrals of vectors (i/r),(j/r),(k/r), around the egde in ccw
    """

    def __init__(self, coord: np.ndarray, obj: Face) -> None:
        """
        coord: observation points' coordinates
        obj: instance of Face object
        """
        logger.trace("initialise FaceStation")
        if not isinstance(obj, Face):
            raise KeyError("obj must be an instance of Face object")

        self.coord = coord
        self.obj = obj
        # derived from obj
        self._points = np.full(1, np.nan)
        self._un = np.full(3, np.nan)
        self._edges = []
        self._simplex = None
        # computed here after
        self.sign = 0
        self.dp1 = np.nan
        self.omega = np.nan
        self.pqr = np.full(3, 0)

    @property
    def points(self) -> np.ndarray:
        if self.obj is not None:
            self._points = self.obj.points
        return self._points

    @property
    def un(self) -> np.ndarray:
        if self.obj is not None:
            self._un = self.obj.un
        return self._un

    @property
    def simplex(self) -> np.ndarray:
        if self.obj is not None:
            self._simplex = self.obj.simplex
        return self._simplex

    @property
    def edges(self) -> list[Edge]:
        if self.obj is not None:
            self._edges = self.obj.edges
        return self._edges

    def get_dot_point1(self, coord: np.ndarray) -> None:
        """scalar product of face's unit outward vector and vector OA,
        where
          O is the observation point
          A is the first corner of the face

        coord: observation points' coordinates
        """
        # shift origin
        _ = self.points[0] - coord
        self.dp1 = np.dot(self.un, _)

    def get_sign(self) -> None:
        """sign of scalar product of face's unit outward vector and vector OA,
        where
          O is the observation point
          A is the first corner of the face

        sign > 0 : observation point is "above" of the face
        sign < 0 : observation point is "below" of the face
        """
        self.sign = np.sign(self.dp1)

    def get_omega(self, coord: np.ndarray) -> None:
        """compute solid angle 'omega' subtended by the face at the observation point

        coord: observation point's coordinates
        """
        npoints = len(self.points)
        if abs(self.dp1) == 0:
            self.omega = 0
        else:
            w = 0
            # shift origin
            points = self.points - coord
            _ = np.concatenate((points, points[0:2]))
            for corners in [_[i : i + 3] for i in range(npoints)]:
                p1, p2, p3 = corners
                # find the solid angle subtended by a polygon at the origin.
                w += self._angle(p1, p2, p3, self.un)
            w -= (npoints - 2) * np.pi
            self.omega = -self.sign * w

    def _angle(
        self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, un: np.ndarray
    ) -> float:
        """
        finds the angle between planes O-p1-p2 and O-p2-p3,
        where
          p1,p2,p3 are coordinates of three points, taken in ccw order
            as seen from origin O.
          un is the unit outward normal vector to the polygon.

        p1,p2,p3: coordinates of three points
        un: unit outward normal vector to the polygon
        """
        # Check if face is seen from inside
        inout = np.dot(un, p1)
        # if inout>0: face seen from inside, interchange p1 and p3
        if inout > 0:
            p1, p3 = p3, p1

        if inout == 0:
            angle = 0
            perp = 1
        else:
            n1 = np.cross(p2, p1)
            n1 = n1 / np.linalg.norm(n1)

            # n2 = np.cross(p3, p2)
            n2 = np.cross(p2, p3)
            n2 = n2 / np.linalg.norm(n2)

            # sign of perp is negative if points p1 p2 p3 are in cw order
            perp = np.sign(np.sum(np.dot(p3, n1)))
            r = np.sum(np.dot(n1, n2))
            angle = math.acos(r)
            if perp < 0:
                angle = 2 * np.pi - angle

        return angle

    def get_ccw_line_integrals(self, obs: np.ndarray) -> None:
        """compute pqr for the current face"""
        # for i, edge in enumerate(self.edges):
        #     logger.info(f"p[{i}]:{edge.coord}")
        for edge in self.edges:
            e = _Edge(self.coord, edge)
            e.get_ccw_line_integrals(obs)

            self.pqr = self.pqr + e.pqr
        #    logger.info(f"PQR: {self.pqr}")

    def get_gravity(self, density: float, Gc: float) -> np.ndarray:
        """compute gravity on the current face"""
        # if distance to face is non-zero
        if self.dp1 != 0:
            (L, M, N) = self.un
            (P, Q, R) = self.pqr

            factor = -density * Gc * self.dp1 * 1e5
            gx = (L * self.omega + N * Q - M * R) * factor
            gy = (M * self.omega + L * R - N * P) * factor
            gz = (N * self.omega + M * P - L * Q) * factor
        else:
            gx, gy, gz = 0, 0, 0

        return np.array([gx, gy, gz])


class _Edge:
    """
    _Edge object: extend Polyhedron's Edge object with parameters link to station

    self.coord: station's coordinate
    self.obj: Polyhedron Edge object
    self.pqr: line integrals of vectors (i/r),(j/r),(k/r), around the egde in ccw
    """

    def __init__(self, coord: np.ndarray, obj: Edge) -> None:
        """
        coord: point of observation coordinates
        obj: instance of Edge object
        """
        logger.trace("initialise _Edge")
        if not isinstance(obj, Edge):
            raise KeyError("obj must be an instance of Edge object")

        self.coord = coord
        self.obj = obj
        # derived from obj
        self._start = np.full(3, np.nan)
        self._end = np.full(3, np.nan)
        self._vector = np.full(3, np.nan)
        self._length = np.nan
        # computed here after
        self.pqr = np.full(3, np.nan)

    @property
    def start(self) -> np.ndarray:
        if self.obj is not None:
            self._start = self.obj.start
        return self._start

    @property
    def end(self) -> np.ndarray:
        if self.obj is not None:
            self._end = self.obj.end
        return self._end

    @property
    def vector(self) -> np.ndarray:
        if self.obj is not None:
            self._vector = self.obj.vector
        return self._vector

    @property
    def length(self) -> float:
        if self.obj is not None:
            self._length = self.obj.length
        return self._length

    def get_ccw_line_integrals(self, obs: np.ndarray) -> None:
        """
        compute the line integral of vectors (i/r), (j/r), (k/r),
        taken around the egde of the polygon in a counterclockwise direction
        """
        integral = 0
        p1, p2 = self.start - obs, self.end - obs
        n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
        chsgn = 1  # if origin,p1 & p2 are on a st line
        if np.dot(p1, p2) / (n1 * n2) == 1 and n1 > n2:  # p1 farther than p2
            chsgn = -1
            p1, p1 = p2, p1  # interchange p1,p2

        V = self.vector
        L = self.length

        L2 = L**2
        b = 2 * np.dot(V, p1)
        r1 = np.linalg.norm(p1)
        r12 = r1**2
        b2 = b / (2 * L)
        if r1 + b2 == 0:
            V = -V
            b = 2 * np.dot(V, p1)
            b2 = b / (2 * L)

        if r1 + b2 != 0:
            integral = (1 / L) * math.log(
                (math.sqrt(L2 + b + r12) + L + b2) / (r1 + b2)
            )

        # change sign of I if p1,p2 were interchanged
        self.pqr = integral * V * chsgn


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

    density = 1000
    Gc = 6.67408e-11
    obs = np.array([-1.05, -1.05, 0])

    s = Station(obs)
    s.compute_gravity(p, density, Gc)

    print(f"Gravity[{obs}]={s.G}")
