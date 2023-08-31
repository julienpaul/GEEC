#!/usr/bin/env python3
# test_polyhedron.py

# --- import -----------------------------------
# import from standard lib
# import from other lib
import glm
import numpy as np
from pytest import approx

import geec.crs

# import from my project
from geec.dataset import Dataset
from geec.polyhedron import get_polyhedron
from tests.conftest import Cube, CubeExcepted, CubeExceptedGLM

# np.allclose => absolute(a - b) <= (atol + rtol * absolute(b))
atol = 1.0e-6  # 1e-14  # absolute tolerance
rtol = 0  # relative tolerance


class TestCube:
    def test_simplices(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        p = get_polyhedron(ds)

        assert p.nfaces == 12
        for i, simplex in enumerate(p.fpoints):
            assert list(simplex) == approx(expected.simplices2[i], abs=atol)

    def test_unit_outward(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        p = get_polyhedron(ds)

        for i, un in enumerate(p.un):
            assert list(un) == approx(expected.un[i], abs=atol)

    def test_compute_gravity(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]
        pqr, dpqr, omega, domega = poly.compute_gravity()

        for i, _face in enumerate(poly.fedges):
            assert omega[i] == approx(expected2.omega[i], abs=atol)
            assert np.array(pqr[i]) == approx(np.array(expected2.pqr[i]), abs=atol)

    def test_compute_gravity_and_gradient(self, cube: Cube, expected2: CubeExceptedGLM):
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        poly = get_polyhedron(ds)

        s = glm.vec3(cube.obs)
        poly.points = [p - s for p in poly.points]
        pqr, dpqr, omega, domega = poly.compute_gravity(gradient=True)

        for i, _face in enumerate(poly.fedges):
            assert omega[i] == approx(expected2.omega[i], abs=atol)
            assert np.array(domega[i]) == approx(
                np.array(expected2.domega[i]), abs=atol
            )
            assert np.array(pqr[i]) == approx(np.array(expected2.pqr[i]), abs=atol)
            assert np.array(dpqr[i]) == approx(np.array(expected2.dpqr[i]), abs=atol)
