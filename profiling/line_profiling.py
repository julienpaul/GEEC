from typer.testing import CliRunner

from geec.cli import app


def test_geec():
    runner = CliRunner()
    runner.invoke(app, "test-grad")


test_geec()


# add decorator '@profile' to function to be checked
# then run 'kernprof.py -lv line_profiling.py'
