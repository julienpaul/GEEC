"""
"""

# --- import -----------------------------------
# import from standard lib
from dataclasses import dataclass, field

# import from other lib
from confuse import LazyConfig

# import from my project
import geec.crs
import geec.dataset
from geec.dataset import Dataset

_OBERVER = {
    "cart": {"name": ["x_mes", "y_mes", "z_mes"], "unit": ["(m)", "(m)", "(m)"]},
    "geo": {
        "name": ["lon_mes", "lat_mes", "alt_mes"],
        "unit": ["(degree_east)", "(degree_north)", "(m)"],
    },
}

_OBSTYPE = {
    geec.crs.CRSEnum.CART: "cart",
    geec.crs.CRSEnum.ELLPS: "geo",
    geec.crs.CRSEnum.ENU: "geo",
    geec.crs.CRSEnum.ECEF: "geo",
}


@dataclass(slots=True, kw_only=True)
class Observer:
    dataset: Dataset
    coords_name: list[str] = field(init=False)
    coords_unit: list[str] = field(init=False)

    def __post_init__(self):
        obstype = _OBSTYPE[self.dataset.crs.name]
        self.coords_name = _OBERVER[obstype]["name"]
        self.coords_unit = _OBERVER[obstype]["unit"]

    def to_lon180(self):
        """Convert longitudes to -180°..180° on observers dataset."""
        self.dataset.to_lon180()

    def to_ellipsoid_height(self):
        """Convert altitudes to ellipsoid heights on observers dataset."""
        self.dataset.to_ellipsoid_height()

    def to_ecef(self):
        """Convert points to cartesian coordinates on observers dataset."""
        self.dataset.to_ecef()

    def to_wgs84(self):
        """Convert points to WGS84 ellipsoid coordinates on observers dataset."""
        self.dataset.to_wgs84()

    def to_enu(self, lon0, lat0, alt0):
        """Convert points to local ENU coordinates on observers dataset."""
        self.dataset.to_enu(lon0, lat0, alt0)


def get_observer(config: LazyConfig) -> Observer:
    obscfg = config["observers"]
    dataset = geec.dataset.get_dataset(obscfg)

    return Observer(dataset=dataset)


if __name__ == "__main__":
    pass
