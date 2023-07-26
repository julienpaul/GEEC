#!/usr/bin/env python3
# cli.py
"""
"""

# --- import -----------------------------------
# import from standard lib
import sys
from pathlib import Path
from typing import Annotated

# import from other lib
import numpy as np
import pandas as pd
import typer
from loguru import logger

# import from my project
import geec
import geec.config
import geec.crs
import geec.dataset
import geec.mass
import geec.observer
from geec.polyhedron import Polyhedron
from geec.station import Station

# setup log directory
_ = Path(__file__)
logdir = _.parent.parent / "log"
logdir.mkdir(parents=True, exist_ok=True)
# removes the default (0ᵗʰ) handler
logger.remove(0)
# add handler to stderr
logger.add(sys.stderr, level="SUCCESS", format="{message}")
# add handler to file
logger.add(logdir / "geec_{time}.log", level="DEBUG")

# setup typer
app = typer.Typer(add_completion=False)


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
def test_grav():
    """
    Test computing gravity fields [mGal] from cube mass body
    """
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

    density = 1000
    Gc = 6.67408e-11
    # obs = np.array([-1.05, -1.05, 0])

    crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
    dataset = geec.dataset.Dataset(coords=points, crs=crs)
    masses = [geec.mass.Mass(density=density, gravity_constant=Gc, dataset=dataset)]

    # transform mass bodies points
    geec.mass.to_lon180(masses)
    geec.mass.to_ellipsoid_height(masses)
    geec.mass.to_ecef(masses)
    # assess Polyhedron of each mass bodies
    p = [Polyhedron(mass.dataset.coords) for mass in masses]

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
    file_path = Path(__file__)
    file_path = file_path.parent.parent / "output" / "out.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


@app.command()
def test_grad():
    """
    Test computing gravity fields [mGal] and
    gradient gravity fields [E] from cube mass body
    """
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

    density = 1000
    Gc = 6.67408e-11
    # obs = np.array([-1.05, -1.05, 0])

    crs = geec.crs.CRS(name=geec.crs.CRSEnum.CART)
    dataset = geec.dataset.Dataset(coords=points, crs=crs)
    masses = [geec.mass.Mass(density=density, gravity_constant=Gc, dataset=dataset)]

    # transform mass bodies points
    geec.mass.to_lon180(masses)
    geec.mass.to_ellipsoid_height(masses)
    geec.mass.to_ecef(masses)
    # assess Polyhedron of each mass bodies
    p = [Polyhedron(mass.dataset.coords) for mass in masses]

    # Start, End and Step
    x_start, x_end, x_step = -1.05, 1.06, 0.1
    y_start, y_end, y_step = -1.05, 1.06, 0.1
    z_start, z_end, z_step = 0, 1, 1

    g = np.mgrid[x_start:x_end:x_step, y_start:y_end:y_step, z_start:z_end:z_step]
    listObs = np.transpose(g.reshape(len(g), -1))

    def add_gravity(row):
        s = Station(np.array(row))
        s.compute_gravity(p, density, Gc, gradient=True)
        listG = s.G
        listT = s.T.flatten()[
            [0, 1, 2, 4, 5, 8]
        ]  # "txx", "txy", "txz", "tyy", "tyz", "tzz"
        return np.concatenate([listG, listT])

    # create dataframe
    df = pd.DataFrame(listObs, columns=["x_mes", "y_mes", "z_mes"])
    # df[["Gx", "Gy", "Gz"]] = df.apply(add_gravity, axis=1, result_type="expand")

    listGT = np.apply_along_axis(add_gravity, axis=1, arr=listObs)
    df[["Gx", "Gy", "Gz", "txx", "txy", "txz", "tyy", "tyz", "tzz"]] = listGT

    # Save result in csv file
    file_path = Path(__file__)
    file_path = file_path.parent.parent / "output" / "out_gradient.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


@app.command()
def run(
    output: Annotated[str, typer.Argument(help="Output file path")],
    usercfg: Annotated[
        str,
        typer.Option(
            "--config",
            help="Configuration file path",
            rich_help_panel="Customization and Utils",
        ),
    ] = "",
    gradient: Annotated[
        bool,
        typer.Option(
            help="Compute gradient gravity fields",
            rich_help_panel="Customization and Utils",
        ),
    ] = False,
):
    """
    Compute gravity fields [mGal] from mass bodies at some observation points.

    Optionally compute gradient gravity fields [E]
    """
    config = geec.config.setup(usercfg)

    geec.config.show(config)
    logger.success(f"Note: logfiles are stored in {logdir}")

    # read mass bodies
    masses = geec.mass.get_masses(config)
    # transform mass bodies points
    geec.mass.to_lon180(masses)
    geec.mass.to_ellipsoid_height(masses)
    geec.mass.to_ecef(masses)
    # assess Polyhedron of each mass bodies
    p = [Polyhedron(mass.dataset.coords) for mass in masses]

    density = masses[0].density
    Gc = masses[0].gravity_constant

    # observation points
    observer = geec.observer.get_observer(config)
    observer.to_wgs84()

    obs_points = observer.dataset.coords
    obs_name = observer.coords_name
    obs_unit = observer.coords_unit

    def add_gravity(row):
        s = Station(np.array(row))
        s.compute_gravity(p, density, Gc, gradient=gradient)
        listG = s.G
        listT = s.T.flatten()[
            [0, 1, 2, 4, 5, 8]
        ]  # "txx", "txy", "txz", "tyy", "tyz", "tzz"
        return np.concatenate([listG, listT])

    listGT = np.apply_along_axis(add_gravity, axis=1, arr=obs_points)
    G_name = ["Gx", "Gy", "Gz"]
    G_unit = ["(mGal)", "(mGal)", "(mGal)"]
    T_name = ["txx", "txy", "txz", "tyy", "tyz", "tzz"]
    T_unit = ["(E)", "(E)", "(E)", "(E)", "(E)", "(E)"]

    # create dataframe
    data = np.concatenate([obs_points, listGT], axis=1)
    name = obs_name + G_name + T_name
    unit = obs_unit + G_unit + T_unit
    columns = pd.MultiIndex.from_tuples(zip(name, unit, strict=True))
    df = pd.DataFrame(data, columns=columns)

    # df = pd.DataFrame(obs_points, columns=["x_mes", "y_mes", "z_mes"])
    # df[["Gx", "Gy", "Gz", "txx", "txy", "txz", "tyy", "tyz", "tzz"]] = listGT

    # Save result in csv file
    file_path = Path(output).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path.with_suffix(".csv"), index=False)
    logger.success(f"\nResults are saved in csv file {file_path.with_suffix('.csv')}")


@app.command()
def config(
    output: Annotated[
        str,
        typer.Option(
            help="output file path",
            rich_help_panel="Customization and Utils",
        ),
    ] = "",
):
    """
    Create a template of the configuration file.
    """
    if output:
        file_path = Path(output).expanduser().resolve().with_suffix(".yaml")
    else:
        file_path = Path("./config_template.yaml").resolve()

    config = geec.config.setup()
    pkg_path = Path(str(config._package_path))
    template = pkg_path / "config_template.yaml"

    # copy template file
    file_path.write_text(template.read_text())
    logger.success(f"Save configuration template in {file_path}")


# def topo(
# topo: Annotated[
#     bool,
#     typer.Option(
#         help=(
#             "Mass body is a Topography. Mass will be split into water and land"
#             " bodies."
#         ),
#         rich_help_panel="Customization and Utils",
#     ),
# ] = False,


def _version_callback(value: bool):
    if value:
        print(f"{__package__} version: {geec.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Show version",
            is_eager=True,
        ),
    ] = False
):
    """
    Awesome Geec program
    """
    # from geec import timing
    # app()


# if __name__ == "__main__":
#     app()
