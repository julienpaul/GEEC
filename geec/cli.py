"""
Command Line Interface
"""

# --- import -----------------------------------
# import from standard lib
from pathlib import Path
from typing import Annotated

# import from other lib
import typer
from loguru import logger

# import from my project
import geec
import geec.api
import geec.config
import geec.crs
import geec.dataset
import geec.mass
import geec.observer

# setup typer
app = typer.Typer(add_completion=False)


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

    # read mass body
    masses = geec.mass.get_masses(config)
    mass = masses[0]
    # transform mass bodies points
    mass.to_orthometric_height()
    mass.to_ecef()
    mass.to_polyhedron()

    # observation points
    observer = geec.observer.get_observer(config)
    observer.to_wgs84()

    observer.compute_gravity(mass, gradient=gradient)

    df = observer.create_dataframe()

    # Save result in csv file
    geec.api.write_file(df, output)


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


# @app.command()
# def topo(
#     input: Annotated[str, typer.Argument(help="Topography file path")],
#     output: Annotated[
#         str,
#         typer.Option(
#             help="output file path",
#             rich_help_panel="Customization and Utils",
#         ),
#     ] = "",
# ):
#     """
#     Create a configuration file from topography.
#
#     Masses will be split into water and land mass.
#     """
#     if output:
#         file_path = Path(output).expanduser().resolve().with_suffix(".yaml")
#     else:
#         file_path = Path("./config_topo.yaml").resolve()
#
#     geec.topography.get_config(input)
#     # copy template file
#     file_path.write_text(template.read_text())
#     logger.success(f"Save configuration topography in {file_path}")


def _version_callback(value: bool):
    if value:
        print(f"{__package__} version: {geec.__version__}")
        raise typer.Exit()


def _logdir_callback(value: bool):
    if value:
        logdir = geec.api.setup_logdir()
        print(f"log directory: {logdir}")
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
    ] = False,
    logdir: Annotated[
        bool,
        typer.Option(
            "--log",
            callback=_logdir_callback,
            help="Show log storage directory",
            is_eager=True,
        ),
    ] = False,
):
    """
    Awesome Geec program
    """
    # setup logger
    geec.api.setup_logger()

    # from geec import timing
    # app()


# if __name__ == "__main__":
#     app()
