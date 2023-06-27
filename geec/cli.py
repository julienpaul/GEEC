#!/usr/bin/env python3
# cli.py
"""
"""

# --- import -----------------------------------
# import from standard lib
from pathlib import Path
from typing import Annotated

# import from other lib
import confuse  # Initialize config with your app
import numpy as np
import pandas as pd
import typer
from loguru import logger

# import from my project
import geec
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
def test_grav():
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


@app.command()
def test_grad():
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
    filepath = Path(__file__)
    filepath = filepath.parent.parent / "output" / "out_gradient.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def _version_callback(value: bool):
    if value:
        print(f"version: {geec.__version__}")
        raise typer.Exit()


def _setup_cfg():
    """set up from configuration file(s)

    read parameters from
    ~/.config/geec/config.yaml
    otherwise from
    /path/to/package/cfg/config_default.yaml
    """
    # set up configuration file
    try:
        # Read configuration file
        config = confuse.LazyConfig(
            "geec", modname=geec.__pkg_cfg__
        )  # Get a value from your YAML file

        # TODO check use of templates,
        #  cf examples in https://github.com/beetbox/confuse/tree/c244db70c6c2e92b001ce02951cf60e1c8793f75

        # set up default configuration file path
        pkg_path = Path(config._package_path)
        config.default_config_path = pkg_path / confuse.DEFAULT_FILENAME

    except Exception:
        logger.error("Something goes wrong when loading config file.")
        raise  # Throw exception again so calling code knows it happened
    else:
        return config


def read_mass_points(mass) -> np.ndarray:
    if mass["points"]:
        points = mass["points"].get(list)
        return np.array(points)
    elif mass["file_path"]:
        file_path = Path(mass["file_path"].get(str)).expanduser().resolve()
        if file_path.is_file():
            df = pd.read_csv(file_path, sep=",", header=None)
            return df.values
        else:
            raise FileNotFoundError(f"File {file_path} not found")
    else:
        raise TypeError("Mass body points must be a list of points or a file")


def read_obs_points(obs) -> np.ndarray:
    if obs["points"]:
        points = obs["points"].get()
        return np.array(points)
    elif obs["file_path"]:
        file_path = obs["file_path"].get(str)
        if Path(file_path).is_file():
            df = pd.read_csv(file_path, sep=",", header=None)
            return df.values
        else:
            raise FileNotFoundError(f"File {file_path} not found")
    elif obs["grid"]:
        grid = obs["grid"]
        # Start, End and Step
        x_start, x_end, x_step = grid["xstart_xend_xstep"].get(list)
        y_start, y_end, y_step = grid["ystart_yend_ystep"].get(list)
        z_start, z_end, z_step = grid["zstart_zend_zstep"].get(list)

        g = np.mgrid[x_start:x_end:x_step, y_start:y_end:y_step, z_start:z_end:z_step]
        return np.transpose(g.reshape(len(g), -1))
    else:
        raise TypeError("Mass body points must be a list of points or a file")


@app.command()
def run(
    output: Annotated[str, typer.Argument(help="output file path")],
    # config: str = "",
    config: Annotated[str, typer.Option(help="configuration file path")] = "",
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            # rich_help_panel="Customization and Utils",
        ),
    ] = False,
    gradient: Annotated[bool, typer.Option()] = False,
):
    """
    Compute gravity fields from a mass body at some observation points.
    """
    cfg = _setup_cfg()
    if config:
        cfg.set_file(config)

    # work on mass body
    mass = cfg["mass"]
    mass_points = read_mass_points(mass)
    p = Polyhedron(mass_points)

    density = mass["density"].get(float)
    Gc = mass["gravity_constant"].get(float)

    # observation points
    obs = cfg["obs"]
    obs_points = read_obs_points(obs)

    def add_gravity(row):
        s = Station(np.array(row))
        s.compute_gravity(p, density, Gc, gradient=gradient)
        listG = s.G
        listT = s.T.flatten()[
            [0, 1, 2, 4, 5, 8]
        ]  # "txx", "txy", "txz", "tyy", "tyz", "tzz"
        return np.concatenate([listG, listT])

    # create dataframe
    df = pd.DataFrame(obs_points, columns=["x_mes", "y_mes", "z_mes"])

    listGT = np.apply_along_axis(add_gravity, axis=1, arr=obs_points)
    df[["Gx", "Gy", "Gz", "txx", "txy", "txz", "tyy", "tyz", "tzz"]] = listGT

    # Save result in csv file
    filepath = Path(output).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath.with_suffix(".csv"), index=False)
    logger.info(f"Save result in csv file {filepath.with_suffix('.csv')}")


def main():
    from geec import timing  # noqa: F401

    app()


if __name__ == "__main__":
    main()
