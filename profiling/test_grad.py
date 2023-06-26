import timeit

setup = """
from typer.testing import CliRunner
from geec.cli import app

def f1():
    runner = CliRunner()
    result = runner.invoke(app, "test_grav")
def f2():
    runner = CliRunner()
    result = runner.invoke(app, "test-grad")
"""
# timeit.timeit("f1()", setup=setup, number=100)
if __name__ == "__main__":
    print(
        "%.2f usec/pass" % (1000 * timeit.timeit("f2()", setup=setup, number=100) / 100)
    )
