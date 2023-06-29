#!/usr/bin/env python3
# test_geec.py

from typer.testing import CliRunner

from geec import __version__
from geec.cli import app


def test_version():
    assert __version__ == "0.0.3"


setup = """
from typer.testing import CliRunner
from geec.cli import app

def test_grav():
    runner = CliRunner()
    result = runner.invoke(app, "test-grav")
def test_grad():
    runner = CliRunner()
    result = runner.invoke(app, "test-grad")
"""


def test_geec_cli_grav():
    runner = CliRunner()
    result = runner.invoke(app, "test-grav")
    assert result.exit_code == 0


def test_geec_cli_grad():
    runner = CliRunner()
    result = runner.invoke(app, "test-grad")
    assert result.exit_code == 0
