#!/usr/bin/env python3
# test_geec.py
import time

from typer.testing import CliRunner

from geec import __version__
from geec.cli import app


def test_version():
    assert __version__ == "0.0.1"


def test_geec_cli():
    runner = CliRunner()
    start_time = time.time()
    result = runner.invoke(app, "test")
    elapse = time.time() - start_time
    assert result.exit_code == 0
    assert elapse <= 0.5
