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

logger.add(Path("logs") / "geec_{time}.log")
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

    # Start, End ans Step
    x_start, x_end, x_step = -1.05, 1.05, 0.1
    y_start, y_end, y_step = -1.05, 1.05, 0.1
    z_start, z_end, z_step = 0, 1, 1

    g = np.mgrid[x_start:x_end:x_step, y_start:y_end:y_step, z_start:z_end:z_step]

    listobs = np.transpose(g.reshape(len(g), -1))
    # for i in range(listobs.shape[-1]):
    #     obs = listobs[..., i]
    # add empty column

    def add_gravity(row):
        s = Station(np.array(row))
        s.compute_gravity(p, density, Gc)
        return s.G

    df = pd.DataFrame(listobs, columns=["x_mes", "y_mes", "z_mes"])
    df[["Gx", "Gy", "Gz"]] = df.apply(add_gravity, axis=1, result_type="expand")

    # df["G"] = df.apply(add_gravity, axis=1)

    # df[["Gx", "Gy", "Gz"]] = 0
    # for index, row in df[["x_mes", "y_mes", "z_mes"]].iterrows():
    #     s = Station(np.array(row))
    #     s.compute_gravtiy(p, density, Gc)
    #     df.loc[index, ["Gx", "Gy", "Gz"]] = s.G
    #     # print(f"Gravity at points {np.array(row)} is {s.G}")

    filepath = Path(__file__)
    filepath = filepath.parent.parent / "output" / "out.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("done")


def main():
    from geec import timing  # noqa: F401

    app()


if __name__ == "__main__":
    main()
