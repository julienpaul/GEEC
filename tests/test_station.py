#!/usr/bin/env python3
# test_geec.py

# --- import -----------------------------------
# import from standard lib
import numpy as np

# import from other lib
# import from my project
from geec.polyhedron import Polyhedron
from geec.station import Station, _Edge, _Face


class Testcube:
    def test_simplices(self, cube, expected):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        for i, face in enumerate(p.faces):
            assert np.array_equal(face.simplex, np.array(expected.simplices[i]))

    def test_edges(self, cube, expected):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        for i, face in enumerate(p.faces):
            for j, edge in enumerate(face.edges):
                e = _Edge(s.coord, edge)
                e.get_ccw_line_integrals(cube.obs)
                assert np.array_equal(e.pqr, np.array(expected.edge_pqr[i][j]))

    def test_faces(self, cube, expected):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        for i, face in enumerate(p.faces):
            f = _Face(s.coord, face)
            f.get_dot_point1(s.coord)
            assert f.dp1 == expected.dp1[i]
            f.get_sign()
            assert f.sign == expected.sign[i]
            f.get_omega(s.coord)
            assert f.omega == expected.omega[i]
            f.get_ccw_line_integrals(s.coord)
            assert np.array_equal(f.pqr, np.array(expected.pqr[i]))

    def test_faces_gravity(self, cube, expected):
        """ """

        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        for i, face in enumerate(p.faces):
            # logger.info(f"\nworking on Face: {face.simplex}")
            f = _Face(s.coord, face)
            f.get_dot_point1(s.coord)
            f.get_sign()
            f.get_omega(s.coord)
            f.get_ccw_line_integrals(s.coord)
            g = f.get_gravity(cube.density, cube.Gc)
            assert np.array_equal(g, np.array(expected.g[i]))

    def test_gravity(self, cube, expected):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        s = Station(cube.obs)
        s.compute_gravity(p, cube.density, cube.Gc)

        assert np.array_equal(s.G, expected.G)

    def test_gravity_listobs(self, cube, expected):
        """ """
        points = np.array(cube.points)
        p = Polyhedron(points)

        for i, obs in enumerate(cube.listobs):
            s = Station(obs)
            s.compute_gravity(p, cube.density, cube.Gc)

            assert np.array_equal(s.G, expected.Glistobs[i])
