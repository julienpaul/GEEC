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

    def __init__(self, coords: tuple[np.ndarray, np.ndarray]) -> None:
        """ """
        logger.trace("initialise Edge")
        self.start = coords[0]
        self.end = coords[1]
        self.vector = self.end - self.start
        self.length = np.linalg.norm(self.vector)
        self.coords = coords

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
        simplex: list[int],
        un: np.ndarray,
        neighbors: list[int] = None,
    ) -> None:
        """
        points: list of points
        un: unit outward vector
        simplex: points indices of the face
        neighbors: neighboring faces of the current face
        """
        logger.trace("initialise Face")
        self.points = points
        self.simplex = simplex
        self.un = un
        self.neighbors = neighbors if neighbors else []

        self.edges = self._get_edges(self.points)

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

    def _get_edges(self, points: np.ndarray) -> list[Edge]:
        """Warning no ccw"""
        # _ = np.append(np.flip(points), points[-1])
        # return [Edge(pair) for pair in pairwise(_)]
        _ = np.insert(points, len(points), points[0], axis=0)
        return [Edge(p) for p in pairwise(_)]
        # return [Edge(_) for _ in combinations(points, 2)]


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

        for i, p in enumerate(points):
            logger.info(f"points[{i}]: {p}")

        self.points = points
        self.faces = self._get_faces()

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
        for s in hull.simplices:
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

            faces.append(Face(hull.points[s], simplex=s, un=un))

        return faces

    def _in_hull(self, points, x):
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
