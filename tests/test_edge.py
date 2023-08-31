# --- import -----------------------------------
# import from standard lib
import glm
import numpy as np

# import from other lib
from pytest import approx

import geec.crs
from geec.dataset import Dataset
from geec.edge import (
    get_ccw_line_integrals,
    get_edges_pqr,
)

# import from my project
# from geec.polyhedre import Polyhedron
from geec.polyhedron import get_polyhedron
from tests.conftest import Cube, CubeExceptedGLM

# a == approx(b): absolute(a - b) <= (atol + rtol * expected(absolute(a-b)))
atol = 1.0e-6  # 1e-14  # absolute tolerance
rtol = 0  # relative tolerance


def minus_list(_list):
    return [-item if item else item for item in _list]


class TestCube:
    """Test gravity only"""

    def test_get_ccw_line_integrals(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        for i, face in enumerate(poly.fedges):
            for j, ie in enumerate(face):
                pts = [poly.points[i] for i in poly.edges[ie]]
                pqr, dpqr = get_ccw_line_integrals(pts)  # , s.coord)
                if poly.redges[i][j]:
                    pqr = -pqr
                assert list(pqr) == approx(expected2.edge_pqr[i][j], abs=atol)

    def test_get_edges_pqr(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        edges_points = [[poly.points[i] for i in e] for e in poly.edges]
        _ = get_edges_pqr(edges_points)  # , s.coord)
        edges_pqr = [tpl[0] for tpl in _]

        for i, face in enumerate(poly.fedges):
            for j, ie in enumerate(face):
                pqr = edges_pqr[ie]
                if poly.redges[i][j]:
                    pqr = -pqr
                assert list(pqr) == approx(expected2.edge_pqr[i][j], abs=atol)


class TestCubeGradient:
    """Test gradient only"""

    def test_get_ccw_line_integrals_derivatives(
        self, cube: Cube, expected2: CubeExceptedGLM
    ):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        for i, face in enumerate(poly.fedges):
            for j, ie in enumerate(face):
                pts = [poly.points[i] for i in poly.edges[ie]]
                pqr, dpqr = get_ccw_line_integrals(pts, gradient=True)  # , s.coord)
                if poly.redges[i][j]:
                    dpqr = -dpqr
                assert np.array(dpqr) == approx(
                    np.array(expected2.edge_dpqr[i][j]), abs=atol
                )

    def test_get_edges_dpqr(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        edges_points = [[poly.points[i] for i in e] for e in poly.edges]
        _ = get_edges_pqr(edges_points, gradient=True)  # , s.coord)
        edges_dpqr = [tpl[1] for tpl in _]
        for i, face in enumerate(poly.fedges):
            for j, ie in enumerate(face):
                dpqr = edges_dpqr[ie]
                if poly.redges[i][j]:
                    dpqr = -dpqr
                assert np.array(dpqr) == approx(
                    np.array(expected2.edge_dpqr[i][j]), abs=atol
                )
