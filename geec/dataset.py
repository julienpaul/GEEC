"""
    The body module sets up the Body class,

    Example usage:
"""

# --- import -----------------------------------
# import from standard lib
from dataclasses import dataclass
from pathlib import Path

# import from other lib
import numpy as np
import numpy.typing as npt
import pandas as pd
from confuse import Subview
from loguru import logger

# import from my project
import geec.crs


@dataclass(slots=True, kw_only=True)
class Dataset:
    coords: npt.NDArray[np.float64]
    crs: geec.crs.CRS  # Coordinate Reference System

    @property
    def lon(self):
        return self.coords.T[0]

    @property
    def lat(self):
        return self.coords.T[1]

    @property
    def alt(self):
        return self.coords.T[2]

    def to_lon180(self) -> None:
        """Convert longitude to -180°..180°,
        depending on coordinates reference system.

        Note: for Cartesian or ECEF coordinate system, do nothing.
        """
        CRS_LON180 = {
            geec.crs.CRSEnum.ENU: geec.crs.wrap180,
            geec.crs.CRSEnum.ELLPS: geec.crs.wrap180,
            geec.crs.CRSEnum.ECEF: lambda x: x,
            geec.crs.CRSEnum.CART: lambda x: x,
        }

        self.coords = CRS_LON180[self.crs.name](self.coords)

    def to_ellipsoid_height(self) -> None:
        """Convert altitude to ellipsoid height,
        depending on coordinates reference system.

        Note: for Cartesian or ECEF coordinate system, do nothing.
        """
        CRS_ELLPSH = {
            geec.crs.CRSEnum.ENU: geec.crs.ellipsoid_height,
            geec.crs.CRSEnum.ELLPS: geec.crs.ellipsoid_height,
            geec.crs.CRSEnum.ECEF: lambda x, y: (x, y),
            geec.crs.CRSEnum.CART: lambda x, y: (x, y),
        }

        self.coords, self.crs = CRS_ELLPSH[self.crs.name](self.coords, self.crs)

    def to_ecef(self):
        """Convert points to cartesian coordinates system,
        depending on coordinates reference system.

        Note: for Cartesian or ECEF coordinate system, do nothing.
        """
        CRS_TOECEF = {
            geec.crs.CRSEnum.ENU: geec.crs.enu_to_ecef,
            geec.crs.CRSEnum.ELLPS: geec.crs.ellps_to_ecef,
            geec.crs.CRSEnum.ECEF: lambda x, y: (x, y),
            geec.crs.CRSEnum.CART: lambda x, y: (x, y),
        }

        self.coords, self.crs = CRS_TOECEF[self.crs.name](self.coords, self.crs)

    def to_wgs84(self):
        """Convert points to wgs84 ellipsoid,
        depending on coordinates reference system.

        Note: for Cartesian coordinates system, do nothing.
        """
        CRS_TOWGS84 = {
            geec.crs.CRSEnum.ENU: geec.crs.enu_to_wgs84,
            geec.crs.CRSEnum.ELLPS: geec.crs.ellps_to_wgs84,
            geec.crs.CRSEnum.ECEF: geec.crs.ecef_to_wgs84,
            geec.crs.CRSEnum.CART: lambda x, y: (x, y),
        }

        self.coords, self.crs = CRS_TOWGS84[self.crs.name](self.coords, self.crs)

    def to_enu(self, lon0, lat0, alt0) -> None:
        """Convert points from WGS84 ellipsoid to ENU coordinate system,
        depending on coordinates reference system.

        ENU origin: lat0, lon0, alt0

        Note: for Cartesian coordinates reference system, do nothing.
        """
        CRS_TOENU = {
            geec.crs.CRSEnum.ENU: lambda x, y, *args: (x, y),
            geec.crs.CRSEnum.ELLPS: geec.crs.wgs84_to_enu,
            geec.crs.CRSEnum.ECEF: geec.crs.ecef_to_enu,
            geec.crs.CRSEnum.CART: lambda x, y, *args: (x, y),
        }

        self.coords, self.crs = CRS_TOENU[self.crs.name](
            self.coords, self.crs, lon0, lat0, alt0
        )


def _read_coords(config: Subview) -> npt.NDArray[np.float64]:
    if config["points"]:
        coords = config["points"].get()
        return np.array(coords)
    elif config["file_path"]:
        file_path = Path(str(config["file_path"].get(str)))
        if file_path.is_file():
            df = pd.read_csv(file_path, sep=",", header=None)
            return df.values
        else:
            msg = f"File {file_path} not found"
            logger.error(msg)
            raise FileNotFoundError(msg)
    elif config["grid"]:
        grid = config["grid"]
        # Start, End and Step
        x_start, x_end, x_step = grid["xstart_xend_xstep"].get([float])
        y_start, y_end, y_step = grid["ystart_yend_ystep"].get([float])
        z_start, z_end, z_step = grid["zstart_zend_zstep"].get([float])

        g = np.mgrid[x_start:x_end:x_step, y_start:y_end:y_step, z_start:z_end:z_step]
        return np.transpose(g.reshape(len(g), -1))
    else:
        msg = "Points must be a list of points or a file"
        logger.error(msg)
        raise TypeError(msg)


def get_dataset(config: Subview) -> Dataset:
    ds_coords = _read_coords(config)
    ds_crs = geec.crs.get_crs(config["crs"])

    return Dataset(coords=ds_coords, crs=ds_crs)
