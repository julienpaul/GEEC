"""
"""

# --- import -----------------------------------
# import from standard lib
from dataclasses import dataclass, field
from operator import itemgetter

# import from other lib
import glm
import numpy as np
import numpy.typing as npt
import pandas as pd
from confuse import LazyConfig
from loguru import logger

# import from my project
import geec.crs
import geec.dataset
import geec.mass
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
    GT: npt.NDArray = field(init=False)

    def __post_init__(self) -> None:
        obstype = _OBSTYPE[self.dataset.crs.name]
        self.coords_name = _OBERVER[obstype]["name"]
        self.coords_unit = _OBERVER[obstype]["unit"]
        self.GT = np.array([])

    def to_ellipsoid_height(self) -> None:
        """Convert altitudes to ellipsoid heights on observers dataset."""
        self.dataset.to_ellipsoid_height()

    def to_ecef(self) -> None:
        """Convert points to cartesian coordinates on observers dataset."""
        self.dataset.to_ecef()

    def to_wgs84(self) -> None:
        """Convert points to WGS84 ellipsoid coordinates on observers dataset."""
        self.dataset.to_wgs84()

    def to_enu(self, lon0, lat0, alt0) -> None:
        """Convert points to local ENU coordinates on observers dataset."""
        self.dataset.to_enu(lon0, lat0, alt0)

    def compute_gravity(self, mass, gradient: bool = False) -> None:
        def add_gravity(row: npt.NDArray) -> npt.NDArray:
            logger.trace(f"compute gravity for observer: {row}")
            try:
                obs = glm.vec3(row)
            except Exception:
                obs = glm.vec3(list(row))
            # mass.change_polyhedron()
            # change projection
            mass.to_enu(obs.x, obs.y, obs.z)
            # no need to shift point for computation in ENU coordinates
            if mass.dataset.crs.name == geec.crs.CRSEnum.ENU:
                obs = glm.vec3()

            G, T = mass.compute_gravity(obs, gradient=gradient)
            # keep only "txx", "txy", "txz", "tyy", "tyz", "tzz"
            T = itemgetter(0, 1, 2, 4, 5, 8)(np.array(T).flatten())
            return np.concatenate([G, T])

        self.GT = np.apply_along_axis(add_gravity, axis=1, arr=self.dataset.coords)

    def create_dataframe(self) -> pd.DataFrame:
        obs_points = self.dataset.coords
        obs_name = self.coords_name
        obs_unit = self.coords_unit
        G_name = ["Gx", "Gy", "Gz"]
        G_unit = ["(mGal)", "(mGal)", "(mGal)"]
        T_name = ["txx", "txy", "txz", "tyy", "tyz", "tzz"]
        T_unit = ["(E)", "(E)", "(E)", "(E)", "(E)", "(E)"]

        # create dataframe
        data = np.concatenate([obs_points, self.GT], axis=1)
        name = obs_name + G_name + T_name
        unit = obs_unit + G_unit + T_unit
        columns = pd.MultiIndex.from_tuples(zip(name, unit, strict=True))
        return pd.DataFrame(data, columns=columns)


def get_observer(config: LazyConfig) -> Observer:
    obscfg = config["observers"]
    dataset = geec.dataset.get_dataset(obscfg)

    return Observer(dataset=dataset)


# def compute_gravity(mass, gradient=False):
#     poly = mass.polyhedron
#     poly.points = mass.dataset.coords - observer.dataset.coords
#
#     faces_omega = geec.face.get_faces_omega(mass.fpoints, mass.un)
#     faces_points = [[points[i] for i in f] for f in fpoints]
#     edges_points = [[points[i] for i in f] for f in edges]
#     faces_omega = [ geec.face.get_omega(pts, un) for pts, un in zip(faces_points, un_list)]
#     edges_pqr = [ geec.edge.get_pqr(pts) for pts in edges_points]
#     faces_pqr = [[edges_pqr[i] if b else -edges_pqr[i] for i,b in zip(f,r)] for f,r in zip(fedges, reverse_edges)]]


if __name__ == "__main__":
    pass
