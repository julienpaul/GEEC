#!/usr/bin/env python3
# test_station.py

# --- import -----------------------------------
# import from standard lib
import numpy as np

# import from other lib
# import from my project
from geec.polyhedron import Polyhedron
from geec.station import Station
from tests.conftest import Cube, CubeExcepted

# np.allclose => absolute(a - b) <= (atol + rtol * absolute(b))
atol = 1e-14  # absolute tolerance
rtol = 0  # relative tolerance


class TestCube:
    def test_edges(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        # poly = deepcopy(p)
        poly = p
        # shift origin to cube.obs
        poly.setup(s.coord)
        for i, face in enumerate(poly.faces):
            # if i != 2:
            #     continue
            for j, edge in enumerate(face.edges):
                edge._get_ccw_line_integrals(s.coord)
                # assert np.array_equal(edge.pqr, np.array(expected.edge_pqr[i][j]))
                assert np.allclose(
                    edge.pqr,  # type: ignore[attr-defined]
                    np.array(expected.edge_pqr[i][j]),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                )

    def test_faces(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        # poly = deepcopy(p)
        poly = p
        # shift origin to cube.obs
        poly.setup(s.coord)
        for i, face in enumerate(poly.faces):
            # if i != 2:
            #     continue
            face._get_dot_point1(s.coord)
            assert face._dp1 == expected.dp1[i]
            face._get_sign()
            assert face._sign == expected.sign[i]
            face._get_omega(s.coord)
            assert face._omega == expected.omega[i]
            face._get_ccw_line_integrals(s.coord)
            # assert np.array_equal(face._pqr, np.array(expected.pqr[i]))
            assert np.allclose(
                face._pqr,  # type: ignore[attr-defined]
                np.array(expected.pqr[i]),
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

    def test_faces_gravity(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        s.compute_gravity(p, cube.density, cube.Gc)

        for i, face in enumerate(p.faces):
            # assert np.array_equal(g, np.array(expected.g[i]))
            assert np.allclose(
                face._g,  # type: ignore[attr-defined]
                np.array(expected.g[i]),
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

    def test_gravity(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        s.compute_gravity(p, cube.density, cube.Gc)

        # assert np.array_equal(s.G, expected.G)
        assert np.allclose(
            s.G,
            expected.G,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )

    def test_gravity_listobs(self, cube: Cube, expected: CubeExcepted):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        for i, obs in enumerate(cube.listobs):
            s = Station(obs)
            s.compute_gravity(p, cube.density, cube.Gc)

            # assert np.array_equal(s.G, expected.G)
            assert np.allclose(
                s.G,
                expected.Glistobs[i],
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
