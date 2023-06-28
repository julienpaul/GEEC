#!/usr/bin/env python3
# test_polyhedron.py

# --- import -----------------------------------
# import from standard lib
import numpy as np

# import from other lib
# import from my project
from geec.polyhedron import Polyhedron
from tests.conftest import Cube, CubeExcepted

# np.allclose => absolute(a - b) <= (atol + rtol * absolute(b))
atol = 1.0e-10  # 1e-14  # absolute tolerance
rtol = 0  # relative tolerance


class TestCube:
    def test_simplices(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        assert p.nfaces == 12
        for i, face in enumerate(p.faces):
            assert np.array_equal(face.simplex, np.array(expected.simplices[i]))

    def test_unit_outward(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        for i, face in enumerate(p.faces):
            assert np.array_equal(face.un, np.array(expected.un[i]))

    # def test_edges_array(self, cube: Cube):
    #     """ """
    #     points = np.array(cube.points)
    #     p = Polyhedron(points)

    #     for i, face in enumerate(p.faces):
    #         for j, edge in enumerate(face.edges):
    #             assert edge == p.edges[i * face.nedges + j]
