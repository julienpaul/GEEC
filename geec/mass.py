"""
"""

# --- import -----------------------------------
# import from standard lib
from dataclasses import dataclass

# import from other lib
from confuse import LazyConfig

# import from my project
import geec.dataset
from geec.dataset import Dataset


@dataclass(slots=True, kw_only=True)
class Mass:
    density: float  # density of the mass body [kg m-3]
    gravity_constant: float  # gravity constant [m3 kg-1 s-2]
    dataset: Dataset

    def to_lon180(self):
        """Convert longitudes to -180°..180° on mass dataset."""
        self.dataset.to_lon180()

    def to_ellipsoid_height(self):
        """Convert altitudes to ellipsoid heights on mass dataset."""
        self.dataset.to_ellipsoid_height()

    def to_ecef(self):
        """Convert points to cartesian coordinates on mass dataset."""
        self.dataset.to_ecef()

    def to_wgs84(self):
        """Convert points to WGS84 ellipsoid coordinates on mass dataset."""
        self.dataset.to_wgs84()

    def to_enu(self, lon0, lat0, alt0):
        """Convert points to local ENU coordinates on mass dataset."""
        self.dataset.to_enu(lon0, lat0, alt0)


def get_masses(config: LazyConfig) -> list[Mass]:
    masses = []
    for masscfg in config["masses"]:
        dataset = geec.dataset.get_dataset(masscfg)
        density = masscfg["density"].get(float)
        gravity_constant = masscfg["gravity_constant"].get(float)

        mass = Mass(density=density, gravity_constant=gravity_constant, dataset=dataset)
        masses.append(mass)

    return masses


def to_lon180(masses: list[Mass]) -> None:
    """Convert longitudes to -180°..180° on each mass datasets."""
    [mass.dataset.to_lon180() for mass in masses]


def to_ellipsoid_height(masses: list[Mass]) -> None:
    """Convert altitudes to ellipsoid heights on each mass datasets."""
    [mass.dataset.to_ellipsoid_height() for mass in masses]


def to_ecef(masses: list[Mass]) -> None:
    """Convert points to cartesian coordinates on each mass datasets."""
    [mass.dataset.to_ecef() for mass in masses]


def to_wgs84(masses: list[Mass]) -> None:
    """Convert points to WGS84 ellipsoid coordinates on each mass datasets."""
    [mass.dataset.to_wgs84() for mass in masses]


def to_enu(masses: list[Mass], lon0, lat0, alt0) -> None:
    """Convert points to local ENU coordinates on each mass bdatasets."""
    [mass.dataset.to_enu(lon0, lat0, alt0) for mass in masses]


if __name__ == "__main__":
    pass
