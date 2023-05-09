#!/usr/bin/env python3
# station.py
"""
    The station module sets up the Station's class,
    and calculates the resulting gravity fields of a polyhedron.

    Example usage:
    from geec.station import Station
    s = Station(points)         # initialise Station object
    s.compute_gravity(p, d, Gc) # compute gravity from polyhedron 'p' with density 'd',
                                # and gravitational constant Gc at the station point
    print(s.G)                  # print Gravity fields at the station point
"""
# --- import -----------------------------------
# import from standard lib
# import from other lib
import numpy as np
from loguru import logger
from rich.pretty import pprint as rprint

# import from my project
from geec.polyhedron import Polyhedron


class Station:
    """
    Station object: gravity fields computation point

    """

    def __init__(self, coord: np.ndarray) -> None:
        """initialise Station object

        coord: station's coordinates
        """

        logger.trace("initialise Station")
        # check points list of coordinates
        if not isinstance(coord, np.ndarray):
            raise KeyError("'coord' must be a numpy array.")

        self.coord = coord
        self.G = np.full(3, 0.0)

    def __rich_repr__(self):
        yield "coord", self.coord
        yield "G", self.G

    def __repr__(self):
        """force to use rich pretty print"""
        rprint(self)
        return ""

    def compute_gravity(
        self, polyhedrons: list[Polyhedron] | Polyhedron, density: float, Gc: float
    ) -> None:
        """
        compute gravity from polyhedrons with density 'density' and
        gravitational constant 'Gc' at the station point

        density: []
        Gc: Gravitational constant []
        """
        logger.trace("compute gravity")
        # if self.obs:
        # self.obs = np.insert(self.obs, self.nobs, obs, axis=0)
        # self.nobs += 1
        # else:
        # self.obs = np.array([obs])
        # self.nobs = 1
        if not isinstance(polyhedrons, list):
            polyhedrons = [polyhedrons]

        if not all(isinstance(x, Polyhedron) for x in polyhedrons):
            raise TypeError("polyhedrons should be of type Polyhedron")

        # TODO: check the fastest
        for poly in polyhedrons:
            # poly = deepcopy(polyhedron)
            self.G += poly.get_gravity(self.coord, density, Gc)


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

    density = 1000
    Gc = 6.67408e-11
    obs = np.array([-1.05, -1.05, 0])

    s = Station(obs)
    s.compute_gravity(p, density, Gc)

    print(f"Gravity[{obs}]={s.G}")
