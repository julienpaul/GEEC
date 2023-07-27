# --- import -----------------------------------
# import from standard lib
import importlib.resources as resources

# import from other lib
from typer.testing import CliRunner

# import from my project
from geec import __version__
from geec.cli import app


def test_version():
    assert __version__ == "0.0.3"


def test_geec_cli_grav():
    pkg_path = resources.files("geec")
    output = pkg_path.parent / "output/out.csv"
    runner = CliRunner()
    result = runner.invoke(app, f"run {output} --config geec/cfg/config_test.yaml")
    assert result.exit_code == 0


def test_geec_cli_grad():
    pkg_path = resources.files("geec")
    output = pkg_path.parent / "output/out_gradient.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        f"run {output} --config geec/cfg/config_test.yaml --gradient",
    )
    assert result.exit_code == 0
