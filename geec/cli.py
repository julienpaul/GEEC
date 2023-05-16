#!/usr/bin/env python3
# cli.py
"""
"""

# --- import -----------------------------------
# import from standard lib
from pathlib import Path

# import from other lib
import numpy as np
import pandas as pd
import typer
from loguru import logger

# import from my project
from geec.polyhedron import Polyhedron
from geec.station import Station

# setup log directory
_ = Path(__file__)
logdir = _.parent.parent / "log"
logdir.mkdir(parents=True, exist_ok=True)
logger.add(logdir / "geec_{time}.log")
app = typer.Typer()


@app.command()
def poly():
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
    Polyhedron(points)


@app.command()
def test():
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
    # obs = np.array([-1.05, -1.05, 0])

    # Start, End and Step
    x_start, x_end, x_step = -1.05, 1.06, 0.1
    y_start, y_end, y_step = -1.05, 1.06, 0.1
    z_start, z_end, z_step = 0, 1, 1

    g = np.mgrid[x_start:x_end:x_step, y_start:y_end:y_step, z_start:z_end:z_step]
    listObs = np.transpose(g.reshape(len(g), -1))

    def add_gravity(row):
        s = Station(np.array(row))
        s.compute_gravity(p, density, Gc)
        return s.G

    # create dataframe
    df = pd.DataFrame(listObs, columns=["x_mes", "y_mes", "z_mes"])
    # df[["Gx", "Gy", "Gz"]] = df.apply(add_gravity, axis=1, result_type="expand")

    listG = np.apply_along_axis(add_gravity, axis=1, arr=listObs)
    df[["Gx", "Gy", "Gz"]] = listG

    # Save result in csv file
    filepath = Path(__file__)
    filepath = filepath.parent.parent / "output" / "out.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def main():
    from geec import timing  # noqa: F401

    app()


if __name__ == "__main__":
    main()
