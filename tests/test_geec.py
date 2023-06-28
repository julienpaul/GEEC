#!/usr/bin/env python3
# test_geec.py

import timeit

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
    # start_time = time.time()
    result = runner.invoke(app, "test-grav")
    # elapse = time.time() - start_time
    assert result.exit_code == 0
    # assert elapse <= 0.5  # sec
    elapse = 1000 * timeit.timeit("test_grav()", setup=setup, number=100) / 100
    assert elapse <= 500  # usec/pass


def test_geec_cli_grad():
    runner = CliRunner()
    # start_time = time.time()
    result = runner.invoke(app, "test-grad")
    # elapse = time.time() - start_time
    assert result.exit_code == 0
    # elapse =1000 * timeit.timeit("test_grad()", setup=setup, number=100) / 100
    # assert elapse <= 1000  # usec/pass
