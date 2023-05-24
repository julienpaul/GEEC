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
import math
from itertools import pairwise

# import from other lib
import numpy as np
from loguru import logger
from rich.pretty import pprint as rprint
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# import from my project
from geec.utils import cross_product  # , vector_nan


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
        # polyhedron: "Polyhedron",
        # polypoints: np.ndarray,
        # polynorm: np.ndarray,
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
        # self._polyhedron = polyhedron
        # self.polypoints = polypoints
        # self.polynorm = polynorm
        self._simplex = np.array(simplex)
        # self.points = self._polyhedron.points[self._simplex]
        self.points = points
        # self._norm = self._polyhedron._norm[self._simplex]
        self.start = self.points[0]
        self.end = self.points[1]
        self.vector = self.end - self.start
        self.length = np.linalg.norm(self.vector)
        self._twinface = twinface
        self._twin = None
        # depending on the station, will be computed later
        # self._pqr = vector_nan
        self._pqr = None

    def reset(self):
        """reset value for future computing"""
        self._pqr = None
        # self._pqr.fill(np.nan)

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

    # def _set_twin_pqr(self, pqr: np.ndarray) -> None:
    def _set_twin_pqr(self, pqr: np.ndarray | None) -> None:
        if isinstance(self._twin, Edge):
            twin = self._twin
            twin._set_pqr(pqr)
        else:
            raise TypeError("'twin' undefined")

    def _get_ccw_line_integrals(self, obs: np.ndarray) -> None:
        """
        compute the line integral of vectors (i/r), (j/r), (k/r),
        taken around the egde of the polygon in a counterclockwise direction

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        # if np.array_equal(self._pqr, vector_nan, equal_nan=True):
        if self._pqr is None:
            integral = 0
            # use shifted coordinates, see Polyhedron.get_gravity
            p1, p2 = (self.points[0] - obs, self.points[1] - obs)
            n1, n2 = (np.linalg.norm(p1), np.linalg.norm(p2))
            # n1, n2 = (
            #     self.polynorm[self._simplex][0],
            #     self.polynorm[self._simplex][1],
            # )
            # p1, p2 = (
            #     self.polypoints[self._simplex][0],
            #     self.polypoints[self._simplex][1],
            # )
            # p1, p2 = (
            #     self._polyhedron.points[self._simplex][0],
            #     self._polyhedron.points[self._simplex][1],
            # )
            # n1, n2 = (
            #     self._polyhedron._norm[self._simplex][0],
            #     self._polyhedron._norm[self._simplex][1],
            # )
            chsgn = 1  # if origin,p1 & p2 are on a st line
            r1 = n1
            if n1 > n2 and np.dot(p1, p2) / (n1 * n2) == 1:  # p1 farther than p2
                p1, p2 = p2, p1  # interchange p1,p2
                chsgn = -1
                r1 = n2

            V = self.vector
            L = self.length

            L2 = L * L
            b = 2 * np.dot(V, p1)
            r12 = r1 * r1
            b2 = b / L / 2
            if r1 + b2 == 0:
                V, b = -V, -b
                b2 = b / L / 2

            if r1 + b2 != 0:
                integral = math.log((math.sqrt(L2 + b + r12) + L + b2) / (r1 + b2)) / L

            # change sign of I if p1,p2 were interchanged
            self._pqr = integral * chsgn * V
            # assign twin value
            self._set_twin_pqr(-self._pqr)


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
        # self._polyhedron = polyhedron
        # self.polypoints = polypoints
        # self.polynorm = polynorm
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
        self._pqr = None
        # self._pqr = vector_nan
        self._g = None

    def reset(self):
        """reset values for future computing"""
        self._dp1 = None
        self._sign = None
        self._omega = None
        # self._pqr.fill(np.nan)
        self._pqr = None
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

    # def _add_pqr(self, pqr: np.ndarray) -> None:
    def _add_pqr(self, pqr: np.ndarray | None) -> None:
        """ """
        # if not np.array_equal(self._pqr, vector_nan, equal_nan=True):
        if self._pqr is not None:
            self._pqr += pqr
        else:
            # self._pqr = deepcopy(pqr)
            self._pqr = pqr

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
        contained = [all([p in s for p in pair]) for s in self.neighbors_simplices]
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

    def _get_omega(self, obs: np.ndarray) -> None:
        """compute solid angle 'omega' subtended by the face at the observation point

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        if self._dp1 is None:
            raise NotImplementedError("'dp1' is undefined")
        if self._sign is None:
            raise NotImplementedError("'sign' is undefined")

        if abs(self._dp1) == 0:
            self._omega = 0
        else:
            w = 0
            # use shifted origin
            # points = self._polyhedron.points[self.simplex]  # [p1,p2,p3]
            # points = self.polypoints[self.simplex]  # [p1,p2,p3]
            points = self.points - obs  # [p1,p2,p3]
            dots = np.dot(self.un, points.T)  # [un.p1, un.p2, un.p3]
            cross = [
                cross_product(points[i], points[(i + 1) % self.npoints])
                for i in range(self.npoints)
            ]  # [p1xp2, p2xp3, p3xp1]
            norm = np.linalg.norm(cross, axis=1)
            norm = norm[np.newaxis, :]
            # unit cross vector
            unitv = cross / norm.T

            for i in range(self.npoints):
                # finds the angle between planes O-p1-p2 and O-p2-p3,
                # p1,p2,p3 are 3 points, taken in ccw order as seen from origin O

                # Check if face is seen from inside
                inout = dots[i]
                n1, n2 = -unitv[i], unitv[(i + 1) % self.npoints]
                if inout == 0:
                    angle = 0
                    perp = 1
                else:
                    p = points[(i + 2) % self.npoints]  # p3
                    # if inout>0: face seen from inside
                    if inout > 0:
                        n1, n2 = n2, n1
                        p = points[i]  # p1

                    # sign of perp is negative if points are clockwise
                    perp = np.dot(p, n1)
                    r = np.dot(n1, n2)
                    angle = math.acos(r)
                    if perp < 0:
                        angle = 2 * np.pi - angle
                    w += angle
            w -= (self.npoints - 2) * np.pi
            self._omega = -self._sign * w

    def _get_ccw_line_integrals(self, obs: np.ndarray) -> None:
        """compute pqr for the current face

        Note: observation points' coordinates are setup through Polyhedron instance
        """
        for edge in self.edges:
            edge._get_ccw_line_integrals(obs)

            self._add_pqr(edge.pqr)

    def get_gravity(self, obs: np.ndarray, density: float, Gc: float) -> np.ndarray:
        """compute gravity from the current face with density 'density' and
        gravitational constant 'Gc' seen from the observation points

        Note: observation points' coordinates are setup through Polyhedron instance

        density: []
        Gc: Gravitational constant []
        """
        # if distance to face is non-zero
        self._get_dot_point1(obs)
        self._get_sign()
        self._get_omega(obs)
        self._get_ccw_line_integrals(obs)

        if self._dp1 is None:
            raise NotImplementedError("'dp1' is undefined")
        if self._omega is None:
            raise NotImplementedError("'omega' is undefined")

        # if np.array_equal(self._pqr, vector_nan, equal_nan=True):
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

        self._g = np.array([gx, gy, gz])
        return self._g


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
        if not isinstance(points, np.ndarray):
            raise KeyError("'points' must be a numpy array.")

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

    def shift_origin(self, coord):
        """change points origin to 'coord'

        return points = points - coord
        """
        self.points -= coord

    def change_norm(self, nan=False):
        """compute norm of points' vectors.

        optionally, force all value to NaN
        """
        if nan:
            self._norm.fill(np.nan)
        else:
            self._norm[:] = np.linalg.norm(self.points, axis=1)

    def set_origin(self, coord):
        """set up new cooridnates' origin 'coord'

        coord: new origin
        """
        # shift origin to new origin
        self.shift_origin(coord)
        # compute norm of each points on new axis
        self.change_norm()

    def reset_origin(self, coord):
        """shift back to former origin, and assign nan to norm

        coord: origin to shift from
        """
        # shift origin back
        self.shift_origin(-coord)
        # assign nan to norm of each points
        self.change_norm(nan=True)

    def setup(self, coord):
        """set up new origin 'coord', and assign G to zeros"""
        # shift origin
        self.set_origin(coord)
        # assign G
        G = np.zeros(3)
        return G

    def reset(self, coord):
        """reset origin and all parameters"""
        # shift back origin
        # self.reset_origin(coord)
        for face in self.faces:
            face.reset()

    def get_gravity(self, coord: np.ndarray, density: float, Gc: float) -> np.ndarray:
        """
        compute gravity from polyhedron with density 'density' and
        gravitational constant 'Gc' seen from the coordinates 'coord'

        coord: observation point's coordinates
        density: []
        Gc: Gravitational constant []
        """

        # setup G array, and shift back origin
        G = np.zeros(3)
        # self.set_origin(coord)

        for face in self.faces:
            g = face.get_gravity(coord, density, Gc)
            G += g

        # reset intermediate variables, and shift back origin
        # self.reset_origin(coord)
        self.reset(coord)

        return G

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
