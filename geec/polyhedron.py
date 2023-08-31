# --- import -----------------------------------
# import from standard lib
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial

# import from other lib
import glm
import numpy as np
from loguru import logger
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

# import from my project
import geec.edge
import geec.face
from geec.dataset import Dataset


@dataclass(slots=True, kw_only=True)
class Polyhedron:
    # data: npt.NDArray[np.float64]
    points: list[glm.vec3]  # = field(init=False)
    centroid: glm.vec3  # = field(init=False)
    # Edge
    edges: list[frozenset]  # = field(init=False)
    # Face
    fpoints: list[list[int]]  # = field(init=False)
    fedges: list[list[int]]  # = field(init=False)
    redges: list[list[bool]]  # = field(init=False)
    un: list[glm.vec3]
    # topo
    fland: list[bool]
    #
    nfaces: int = field(init=False)
    npoints: int = field(init=False)
    nedges: int = field(init=False)

    def __post_init__(self):
        self.nfaces = len(self.fpoints)
        self.npoints = len(self.points)
        self.nedges = len(self.edges)

    # def compute_gravity(self):
    #     faces_points = [[self.points[i] for i in f] for f in self.fpoints]
    #     edges_points = [[self.points[i] for i in e] for e in self.edges]

    #     omega_domega = geec.face2.get_faces_omega(faces_points, self.un)
    #     pqr_dpqr = geec.edge2.get_edges_pqr(edges_points)

    #     faces_omega = [tpl[0] for tpl in omega_domega]
    #     faces_domega = [tpl[1] for tpl in omega_domega]

    #     edges_pqr = [tpl[0] for tpl in pqr_dpqr]
    #     edges_dpqr = [tpl[1] for tpl in pqr_dpqr]
    #     faces_pqr = [
    #         sum([edges_pqr[i] if not b else -edges_pqr[i] for i, b in zip(f, r)])
    #         for f, r in zip(self.fedges, self.redges)
    #     ]
    #     faces_dpqr = [
    #         [
    #             edges_dpqr[i] if not b or not edges_dpqr[i] else -edges_dpqr[i]
    #             for i, b in zip(f, r)
    #         ]
    #         for f, r in zip(self.fedges, self.redges)
    #     ]

    #     return (faces_pqr, faces_dpqr, faces_omega, faces_domega)

    def compute_gravity(self, gradient: bool = False):
        faces_points = [[self.points[i] for i in f] for f in self.fpoints]
        edges_points = [[self.points[i] for i in e] for e in self.edges]

        omega_domega = geec.face.get_faces_omega(faces_points, self.un, gradient)
        pqr_dpqr = geec.edge.get_edges_pqr(edges_points, gradient)

        faces_omega = [tpl[0] for tpl in omega_domega]
        faces_domega = [tpl[1] for tpl in omega_domega]

        edges_pqr = [tpl[0] for tpl in pqr_dpqr]
        edges_dpqr = [tpl[1] for tpl in pqr_dpqr]
        # reverse pqr if edges turn upside down, and sum edges of each face
        faces_pqr = [
            sum(
                [
                    edges_pqr[i] if not b else -edges_pqr[i]
                    for i, b in zip(f, r, strict=True)
                ]
            )
            for f, r in zip(self.fedges, self.redges, strict=True)
        ]
        # reverse dpqr if edges turn upside down
        edges_order_dpqr = [
            [
                edges_dpqr[i] if not b or not edges_dpqr[i] else -edges_dpqr[i]
                for i, b in zip(f, r, strict=True)
            ]
            for f, r in zip(self.fedges, self.redges, strict=True)
        ]
        # sum edges of each face, if gradient
        faces_dpqr = [sum(dpqr) if gradient else None for dpqr in edges_order_dpqr]

        return (faces_pqr, faces_dpqr, faces_omega, faces_domega)

    # compute_gravity = partialmethod(_compute_gravity, gradient=False)
    # compute_gravity_and_gradient = partialmethod(_compute_gravity, gradient=True)


def get_polyhedron(dataset: Dataset, topo: bool = False):
    """ """
    # TODO move to __init__ of Polyhedron object
    # compute convex hull
    data = dataset.coords
    hull = ConvexHull(data)
    # points = [glm.vec3(vertex) for vertex in hull.points[hull.vertices, :]]
    # hull = ConvexHull(points)
    points = [glm.vec3(vertex) for vertex in hull.points]
    npoints = len(points)
    # faces as list of points
    fpoints = [list(s) for s in hull.simplices]
    # get centroid
    centroid = get_centroid(points, topo)
    # compute outward unit vector of each face
    un = get_outward_unit_vectors(points, fpoints)
    # check ccw
    ccw = check_face_ccw(points, fpoints, centroid, un)
    # force face to be counterclockwise (if need be)
    un, fpoints = reverse_face(ccw, un, fpoints)
    # edges as set (unordered list) of points
    edges = get_set_edges(fpoints)
    nedges = len(edges)

    fland = [True] * len(fpoints)
    if topo:
        fland, fwater = check_land_water(points, fpoints)
        # turn water face upside down from topography
        un, fpoints = reverse_face(fwater, un, fpoints)

        # use geoid height instead of orthometric height
        # data[:, 2] = dataset.geoh
        # hull = ConvexHull(data)
        # geoid_points = [glm.vec3(vertex) for vertex in hull.points]
        geoid_points = [
            glm.vec3(v.x, v.y, h) for v, h in zip(points, dataset.geoh, strict=True)
        ]
        # we use same indices as for topography but we reverse the order of points
        geoid_fpoints = [f[::-1] for f in fpoints]
        # compute outward unit vector of each face
        geoid_un = get_outward_unit_vectors(geoid_points, geoid_fpoints)
        # # check ccw
        # cw = check_face_cw(geoid_points, geoid_fpoints, centroid, geoid_un)
        # # force face to be clockwise (if need be)
        # geoid_un, geoid_fpoints = reverse_face(cw, geoid_un, geoid_fpoints)
        # edges as set (unordered list) of points
        # geoid_edges = get_set_edges(geoid_fpoints)
        # As we use same points indices, list of edges is the same
        geoid_edges = deepcopy(edges)

    # faces as list of edges
    fedges, redges = get_face_edges(fpoints, edges)

    if topo:
        # faces as list of edges
        geoid_fedges = deepcopy(fedges)
        # reverse edges are
        geoid_redges = [[not x for x in f] for f in redges]

        # concatenate points from topography and geoids
        points += geoid_points
        # concatenate edges from topography and geoids
        edges += geoid_edges

        # increment the indices of face points
        geoid_fpoints = [[i + npoints for i in f] for f in geoid_fpoints]
        # increment the indices of face edges
        geoid_fedges = [[i + nedges for i in f] for f in geoid_fedges]

        # concatenate face points from topography and geoids
        fpoints += geoid_fpoints
        # concatenate face edges from topography and geoids
        fedges += geoid_fedges
        # concatenate reverse edges from topography and geoids
        redges += geoid_redges
        # concatenate unit outward vector from topography and geoids
        un += geoid_un
        # concatenate face land from topography and geoids
        fland += fwater

    p = Polyhedron(
        points=points,
        centroid=centroid,
        edges=edges,
        fpoints=fpoints,
        fedges=fedges,
        redges=redges,
        un=un,
        fland=fland,
    )

    return p


def centroid_in_hull(points, x) -> bool:
    """
    check if point 'x' in inside the convex hull of 'points'

    source: https://stackoverflow.com/a/43564754
    """
    _points = np.array(points)
    n_points = len(_points)
    c = np.zeros(n_points)
    A = np.r_[_points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def get_centroid(points, topo: bool = False):
    if topo:
        centroid = glm.vec3()
    else:
        centroid = glm.vec3(np.mean(points, axis=0))
        if not centroid_in_hull(points, centroid):
            _msg = "Centroid not in hull. Polyhedron not Convex."
            logger.error(_msg)
            raise ValueError(_msg)
    return centroid


def get_outward_unit_vectors(points, fpoints):
    # list of normalized output vector for each face
    return [
        glm.normalize(glm.cross(points[i1] - points[i0], points[i2] - points[i0]))
        for i0, i1, i2 in fpoints
    ]


def _check_face(points, fpoints, centroid, un_list, ccw=True) -> list[bool]:
    """Check face's points are counterclockwise"""
    # vector from centroid to face's vertex
    AD_list = [centroid - points[p[0]] for p in fpoints]
    dot = [glm.dot(AD, un) for AD, un in zip(AD_list, un_list, strict=True)]
    # check if any dot is zero
    if not all(dot):
        BD_list = [centroid - points[p[1]] for p in fpoints]
        BD_dot_un = [glm.dot(BD, un) for BD, un in zip(BD_list, un_list, strict=True)]
        dot = [b if a == 0 else a for a, b in zip(dot, BD_dot_un, strict=True)]
    # dot < 0: A,B,C are     counterclockwise
    # dot > 0: A,B,C are not counterclockwise
    _ccw = [(d < 0) for d in dot] if ccw else [(d > 0) for d in dot]
    return _ccw


# check face are counterclockwise
check_face_ccw = partial(_check_face, ccw=True)
# check face are clockwise
check_face_cw = partial(_check_face, ccw=False)


def check_land_water(points, fpoints) -> tuple[list[bool], list[bool]]:
    """Check if most of points in a face are above/below water level

    Return list of boolean values indicating if face is land (or not).
    """
    land = [1 if ortho > 0 else -1 for x, y, ortho in points]
    fland = [sum([land[i] for i in f]) > 0 for f in fpoints]
    fwater = [not b for b in fland]
    return fland, fwater


def reverse_face(ccw: list[bool], un, fpoints):
    """Reverse face if not counterclockwise/clockwise.

    Turn face upside down, and reverse points order

    Return outward unit vector and face's points
    """
    # reverse vector
    un = [-v if not b else v for b, v in zip(ccw, un, strict=True)]
    # reverse last points, to make them counterclockwise/clockwise
    _ = [s.reverse() if not b else s for b, s in zip(ccw, fpoints, strict=True)]
    return un, fpoints


def _f0(x: list[int]):
    """Returns frozen set of the first edge of the simplice x"""
    return frozenset([x[0], x[1]])


def _f1(x: list[int]):
    """Returns frozen set of the second edge of the simplice x"""
    return frozenset([x[1], x[2]])


def _f2(x: list[int]):
    """Returns frozen set of the third edge of the simplice x"""
    return frozenset([x[2], x[0]])


def get_set_edges(fpoints):
    """edges as set (unordered list) of points"""
    return list({f(x) for x in fpoints for f in (_f0, _f1, _f2)})


def get_face_edges(fpoints, edges):
    """faces as list of edges"""
    # fedges = [
    #     [edges.index(_f0(x)), edges.index(_f1(x)), edges.index(_f2(x))] for x in fpoints
    # ]
    forder = [([x[0], x[1]], [x[1], x[2]], [x[2], x[0]]) for x in fpoints]
    # TODO: computing fedges is really slow
    fedges = [[edges.index(frozenset(x)) for x in f] for f in forder]
    redges = [[list(frozenset(x)) != x for x in f] for f in forder]
    return fedges, redges
