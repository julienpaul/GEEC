# --- import -----------------------------------
# import from standard lib
import timeit

# import from other lib
# import from my project

setup = """
import importlib.resources as resources
from pathlib import Path
from geec.cli import app
import geec.config
import geec.mass
import geec.observer

pkg_path = Path(resources.files("geec"))
output = pkg_path.parent / "output" / "out.csv"
cfgfile = pkg_path / "cfg" / "config_test_cube.yaml"
config = geec.config.setup(cfgfile)

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

def test_grav():
    observer.compute_gravity(mass)

def test_grad():
    observer.compute_gravity(mass, gradient=True)
"""


def test_speed_geec_cli_grav():
    """Test execution speed of the computation of gravity fields"""
    # number of executions
    npass = 100
    # timeit gives result in seconds
    to_usec = 1000
    elapse = to_usec * timeit.timeit("test_grav()", setup=setup, number=npass) / npass
    assert elapse <= 150  # usec/pass


def test_speed_geec_cli_grad():
    """Test execution speed of the computation of gravity and gradient fields"""
    # number of executions
    npass = 100
    # timeit gives result in seconds
    to_usec = 1000
    elapse = to_usec * timeit.timeit("test_grad()", setup=setup, number=npass) / npass
    assert elapse <= 300  # usec/pass
