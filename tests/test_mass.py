#!/usr/bin/env python3
# test_station.py

# --- import -----------------------------------
# import from standard lib
# import from other lib
import glm
import numpy as np
import pytest
from pytest import approx

import geec.crs
import geec.mass

# import from my project
from geec.dataset import Dataset
from geec.mass import Mass
from tests.conftest import Cube, CubeExcepted, CubeExceptedGLM

# np.allclose => absolute(a - b) <= (atol + rtol * absolute(b))
atol = 1.0e-6  # 1e-14  # absolute tolerance
atol_derivatives = 1.0e-4  # 1e-14  # absolute tolerance
rtol = 0  # relative tolerance

np.set_printoptions(precision=20)


class TestCube:
    """Test gravity only"""

    def test_faces_gravity(self, cube: Cube, expected2: CubeExceptedGLM):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        mass = Mass(density=cube.density, gravity_constant=cube.Gc, dataset=ds)
        mass.to_polyhedron()

        s = glm.vec3(cube.obs)
        G, T = mass.compute_gravity(s, faces=True)

        for i in range(mass.polyhedron.npoints):
            assert np.array(G[i]) == approx(np.array(expected2.faces_g[i]), abs=atol)

    def test_gravity(self, cube: Cube, expected2: CubeExceptedGLM):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        mass = Mass(density=cube.density, gravity_constant=cube.Gc, dataset=ds)
        mass.to_polyhedron()

        s = glm.vec3(cube.obs)
        G, T = mass.compute_gravity(s)

        assert np.array(G) == approx(np.array(expected2.G), abs=atol)

    def test_faces_gradient(self, cube: Cube, expected2: CubeExceptedGLM):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        mass = Mass(density=cube.density, gravity_constant=cube.Gc, dataset=ds)
        mass.to_polyhedron()

        s = glm.vec3(cube.obs)
        G, T = mass.compute_gravity(s, gradient=True, faces=True)

        for i in range(mass.polyhedron.npoints):
            assert np.array(T[i]) == approx(
                np.array(expected2.faces_T[i]), abs=atol_derivatives
            )

    def test_gradient(self, cube: Cube, expected2: CubeExceptedGLM):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        mass = Mass(density=cube.density, gravity_constant=cube.Gc, dataset=ds)
        mass.to_polyhedron()

        s = glm.vec3(cube.obs)
        G, T = mass.compute_gravity(s, gradient=True)

        assert np.array(T) == approx(np.array(expected2.T), abs=atol_derivatives)

    def test_gradient_diagonal(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
        ds = Dataset(coords=points, crs=crs)
        mass = Mass(density=cube.density, gravity_constant=cube.Gc, dataset=ds)
        mass.to_polyhedron()

        s = glm.vec3(cube.obs)
        G, T = mass.compute_gravity(s, gradient=True)

        assert T[0][0] + T[1][1] + T[2][2] == approx(0, abs=atol_derivatives)


if __name__ == "__main__":
    pytest.main(["test_mass.py", "-s"])
