#!/usr/bin/env python3
# polyhedron.py
"""
    The polyhedron module sets up the polyhedron's classes (Polyhedron, Face, Edge).
    Example usage:
    from geec.polyhedron import Polyhedron
    p = Polyhedron(points)           # initialise Polyhedron object
"""

# --- import -----------------------------------
# import from standard lib
import math
from copy import deepcopy
from itertools import pairwise

# import from other lib
import numpy as np
from loguru import logger
from rich.pretty import pprint as rprint
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# import from my project


class Edge:
    """
    Edge object: line segments connecting certain pairs of vertices

    start: starting point's coordinates
    end: ending point's coordinates
    vector: edge's vector
    length: length of edge's vector
    coords: tuple of edge coordinates
    """

    def __init__(self, coords: tuple[np.ndarray, np.ndarray], twinface: int) -> None:
        """ """
        logger.trace("initialise Edge")
        self.start = coords[0]
        self.end = coords[1]
        self.vector = self.end - self.start
        self.length = np.linalg.norm(self.vector)
        self.coords = coords
        self._twinface = twinface
        self._twin = None
        # depending on the station, so compute later
        self._pqr = None

    def reset(self):
        """ """
        self._pqr = None

    @property
    def pqr(self) -> np.ndarray:
        return self._pqr

    def set_twin(self, twin) -> None:
        if isinstance(twin, Edge):
            self._twin = twin
        else:
            raise TypeError("'twin' must be Edge object")

    def set_pqr(self, pqr: np.ndarray) -> None:
        if isinstance(pqr, np.ndarray):
            self._pqr = pqr
        else:
            raise TypeError("'pqr' must be numpy array")

    def set_twin_pqr(self, pqr: np.ndarray) -> None:
        if isinstance(self._twin, Edge):
            twin = self._twin
            twin.set_pqr(pqr)
        else:
            raise TypeError("'twin' undefined")

    def __rich_repr__(self):
        yield "start", self.start
        yield "end", self.end
        yield "vector", self.vector
        yield "length", self.length
        yield "coords", self.coords

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    # def __repr__(self):
    #     return (
    #         str(self.__class__.__name__)
    #         + "(\n"
    #         + f"\tstart: {self.start}\n"
    #         + f"\tend: {self.end}\n"
    #         + f"\tvector: {self.vector}\n"
    #         + f"\tlength: {self.length}\n"
    #         + f"\tcoords: {self.coords}\n"
    #         # + pformat(vars(self), indent=4, depth=2, sort_dicts=False)
    #         + ")"
    #     )

    def _get_ccw_line_integrals(self, coord: np.ndarray) -> None:
        """
        compute the line integral of vectors (i/r), (j/r), (k/r),
        taken around the egde of the polygon in a counterclockwise direction

        coord: observation points' coordinates
        """
        if self._pqr is None:
            integral = 0
            p1, p2 = self.start - coord, self.end - coord
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
            self._pqr = integral * V * chsgn
            # assign twin value
            self.set_twin_pqr(-self._pqr)


class Face:
    """
    Face object

    points: list of vertices (corner points)
    neighbors: neighboring faces of the current face
    simplex: points indices of the face
    un: unit outward vector
    edges: list of Edges of the face
    """

    def __init__(
        self,
        points: np.ndarray,
        simplex: np.ndarray,
        un: np.ndarray,
        neighbors: np.ndarray,
        simplices: np.ndarray,
    ) -> None:
        """
        points: list of points
        un: unit outward vector
        simplex: points indices of the face
        neighbors: neighboring faces of the current face
        simplices: indices of the neighboring faces
        """
        logger.trace("initialise Face")
        self.points = points
        self.npoints = len(points)
        self.simplex = simplex
        self.un = un
        self.neighbors = neighbors
        self.simplices = simplices

        self.edges = self._get_edges()

        # depending on the station, so compute later
        self._dp1 = None
        self._sign = None
        self._omega = None
        self._pqr = None

    def reset(self):
        """ """
        self._dp1 = None
        self._sign = None
        self._omega = None
        self._pqr = None
        for edge in self.edges:
            edge.reset()

    def __rich_repr__(self):
        yield "points", self.points
        yield "simplex", self.simplex
        yield "un", self.un
        yield "neighbors", self.neighbors
        yield "edges", self.edges

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    def add_pqr(self, pqr: np.ndarray) -> None:
        """ """
        if self._pqr is not None:
            self._pqr += pqr
        else:
            self._pqr = deepcopy(pqr)

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

    def _get_edges(self) -> list[Edge]:
        """set up edges of the face

        Warning points should already be sorted counterclockwise
        """
        # _ = np.append(np.flip(points), points[-1])
        # return [Edge(pair) for pair in pairwise(_)]

        # edges = []
        # simplex_pairs = [p for p in pairwise(_simplex)]
        # point_pairs = [p for p in pairwise(_points)]
        # for x, y in zip(point_pairs, simplex_pairs):
        #     twin = self._get_twin(y)
        #     edges.append(Edge(x, twin))
        # return edges

        # _ = np.insert(self.points, self.npoints, self.points[0], axis=0)
        # return [Edge(p) for p in pairwise(_)]
        # return [Edge(_) for _ in combinations(points, 2)]

        _simplex = np.insert(self.simplex, self.npoints, self.simplex[0], axis=0)
        _points = np.insert(self.points, self.npoints, self.points[0], axis=0)
        return [
            Edge(p, self._get_twinface(s))
            for p, s in zip(pairwise(_points), pairwise(_simplex), strict=True)
        ]

    def _get_twinface(self, pair: tuple[int, int]) -> int:
        """find index of the face of the edge's twin."""
        contained = [all([p in s for p in pair]) for s in self.simplices]
        return self.neighbors[contained][0]

    def _get_dot_point1(self, coord: np.ndarray) -> None:
        """scalar product of face's unit outward vector and vector OA,
        where
          O is the observation point
          A is the first corner of the face

        coord: observation points' coordinates
        """
        # shift origin
        _ = self.points[0] - coord
        self._dp1 = np.dot(self.un, _)

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

    def _get_omega(self, coord: np.ndarray) -> None:
        """compute solid angle 'omega' subtended by the face at the observation point

        coord: observation point's coordinates
        """
        if self._dp1 is None:
            raise NotImplementedError("'dp1' is undefined")
        if self._sign is None:
            raise NotImplementedError("'sign' is undefined")

        if abs(self._dp1) == 0:
            self._omega = 0
        else:
            w = 0
            # shift origin
            points = self.points - coord
            _ = np.concatenate((points, points[0:2]))
            for corners in [_[i : i + 3] for i in range(self.npoints)]:
                p1, p2, p3 = corners
                # find the solid angle subtended by a polygon at the origin.
                w += self._angle(p1, p2, p3, self.un)
            w -= (self.npoints - 2) * np.pi
            self._omega = -self._sign * w

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

    def _get_ccw_line_integrals(self, coord: np.ndarray) -> None:
        """compute pqr for the current face

        coord: observation point's coordinates
        """
        # for i, edge in enumerate(self.edges):
        #     logger.info(f"p[{i}]:{edge.coord}")
        # print(f"face: {self.simplex}")
        for edge in self.edges:
            edge._get_ccw_line_integrals(coord)
            # print(f"edge pqr: {edge._pqr}")

            self.add_pqr(edge.pqr)

    def get_gravity(self, coord: np.ndarray, density: float, Gc: float) -> np.ndarray:
        """compute gravity from the current face with density 'density' and
        gravitational constant 'Gc' seen from the coordinates 'coord'

        coord: observation point's coordinates
        density: []
        Gc: Gravitational constant []
        """
        # if distance to face is non-zero
        self._get_dot_point1(coord)
        self._get_sign()
        self._get_omega(coord)
        self._get_ccw_line_integrals(coord)

        if self._dp1 is None:
            raise NotImplementedError("'dp1' is undefined")
        if self._omega is None:
            raise NotImplementedError("'omega' is undefined")
        if self._pqr is None:
            raise NotImplementedError("'pqr' is undefined")

        if self._dp1 != 0:
            (L, M, N) = self.un
            (P, Q, R) = self._pqr

            factor = -density * Gc * self._dp1 * 1e5
            gx = (L * self._omega + N * Q - M * R) * factor
            gy = (M * self._omega + L * R - N * P) * factor
            gz = (N * self._omega + M * P - L * Q) * factor
        else:
            gx, gy, gz = 0, 0, 0

        return np.array([gx, gy, gz])


class Polyhedron:
    """
    Polyhedron object

    points: list of vertices (corner points) [x,y,z]
    faces: list of Faces (two-dimensional polygons)
    """

    def __init__(self, points: np.ndarray) -> None:
        """initialise Polyhedron object

        points: list of vertices (corner points) [x,y,z]
        """
        logger.trace("initialise Polyhedron")
        # check points list of coordinates
        if not isinstance(points, np.ndarray):
            raise KeyError("'points' must be a numpy array.")

        self.points = points
        self.npoints = len(points)
        self.faces = self._get_faces()
        self.nfaces = len(self.faces)

    def __rich_repr__(self):
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

        faces = []
        # check simplex points are in counterclockwise
        for s, n in zip(hull.simplices, hull.neighbors, strict=True):
            # get 3 first points of simplex
            A, B, C = hull.points[s][0:3]
            AB = B - A
            AC = C - A
            # normal to the plane containing (A,B,C)
            cross = np.cross(AB, AC)

            # look for a point inside the hull,
            # and such that DA is not parallel to the normal of the ABC plane
            D = centroid
            D_in_hull = self._in_hull(hull.points, D)
            AD = D - A
            dot = np.dot(cross, AD)
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
                dot = np.dot(cross, AD)
                if _niter > 10:
                    raise RuntimeError("Too many iteration to find point inside hull")

            if dot > 0:
                # A,B,C are not counterclockwise
                ccw = False
            else:
                # A,B,C are already counterclockwise
                ccw = True

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
                    hull.points[s],
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
        """
        find edge's twin of all edges
        """
        for face in faces:
            for edge in face.edges:
                if edge._twin is None:
                    facetwin = faces[edge._twinface]
                    for twin in facetwin.edges:
                        if np.array_equal(edge.start, twin.end) and np.array_equal(
                            edge.end, twin.start
                        ):
                            # set the twin of 'edge'
                            edge.set_twin(twin)
                            twin.set_twin(edge)
                            break

    def get_gravity(self, coord: np.ndarray, density: float, Gc: float) -> np.ndarray:
        """
        compute gravity from polyhedron with density 'density' and
        gravitational constant 'Gc' seen from the coordinates 'coord'

        coord: observation point's coordinates
        density: []
        Gc: Gravitational constant []
        """
        # TODO: check the fastest
        # reset intermedaite variables
        self.reset()

        # G = np.sum(
        #     [face.get_gravity(coord, density, Gc) for face in self.faces], axis=0
        # )

        G = np.array([0.0, 0.0, 0.0])
        for face in self.faces:
            g = face.get_gravity(coord, density, Gc)
            G += g
        return G

    def reset(self):
        """ """
        for face in self.faces:
            face.reset()


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
    print(p)
