#!/usr/bin/env python3
# test_station.py

# --- import -----------------------------------
# import from standard lib
import glm
import numpy as np

# import from other lib
from pytest import approx

import geec.crs
from geec.dataset import Dataset
from geec.face import get_faces_omega, get_omega

# import from my project
from geec.polyhedron import get_polyhedron
from tests.conftest import Cube, CubeExceptedGLM

# a == approx(b): absolute(a - b) <= (atol + rtol * expected(absolute(a-b)))
atol = 1.0e-6  # 1e-14  # absolute tolerance
rtol = 0  # relative tolerance


class TestCube:
    """Test gravity only"""

    def test_get_omega(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        for i, un in enumerate(poly.un):
            pts = [poly.points[j] for j in poly.fpoints[i]]
            omega, domega = get_omega(pts, un)
            assert omega == approx(expected2.omega[i], abs=atol)

    def test_get_faces_omega(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        faces_points = [[poly.points[i] for i in f] for f in poly.fpoints]
        _ = get_faces_omega(faces_points, poly.un)
        omega = [tpl[0] for tpl in _]
        for i, _un in enumerate(poly.un):
            assert omega[i] == approx(expected2.omega[i], abs=atol)


class TestCubeGradient:
    """Test gradient only"""

    def test_get_domega(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        for i, un in enumerate(poly.un):
            pts = [poly.points[j] for j in poly.fpoints[i]]
            omega, domega = get_omega(pts, un, gradient=True)
            assert np.array(domega) == approx(np.array(expected2.domega[i]), abs=atol)

    def test_get_faces_domega(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]

        faces_points = [[poly.points[i] for i in f] for f in poly.fpoints]
        _ = get_faces_omega(faces_points, poly.un, gradient=True)
        domega = [tpl[1] for tpl in _]
        for i, _un in enumerate(poly.un):
            assert np.array(domega[i]) == approx(
                np.array(expected2.domega[i]), abs=atol
            )
