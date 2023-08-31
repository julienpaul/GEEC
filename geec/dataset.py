"""
    The dataset module sets up the Dataset class,

    Example usage:
"""

# --- import -----------------------------------
# import from standard lib
from dataclasses import dataclass, field
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
    geoh: npt.NDArray[np.float64] | None = field(init=False)

    @property
    def lon(self):
        return self.coords.T[0]

    @property
    def lat(self):
        return self.coords.T[1]

    @property
    def alt(self):
        return self.coords.T[2]

    def __post_init__(self):
        # Convert longitude to -180°..180°
        self._to_lon180()
        # Get geoid height
        self.geoh = self._get_geoh()

    def _to_lon180(self) -> None:
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

    def _get_geoh(self) -> npt.NDArray[np.float64] | None:
        """Get geoid height at each coords location,
        depending on coordinates reference system.

        Note: for Cartesian or ECEF coordinate system, do nothing.
        """
        CRS_GEOH = {
            geec.crs.CRSEnum.ENU: geec.crs.geoid_heights,
            geec.crs.CRSEnum.ELLPS: geec.crs.geoid_heights,
            geec.crs.CRSEnum.ECEF: lambda x, y: None,
            geec.crs.CRSEnum.CART: lambda x, y: None,
        }

        return CRS_GEOH[self.crs.name](self.coords, self.crs)

    def to_orthometric_height(self) -> None:
        """Convert altitude to orthometric height,
        depending on coordinates reference system.

        Note: for Cartesian or ECEF coordinate system, do nothing.
        """
        VCRS_ORTHO = {
            geec.crs.VDatumEnum.ORTHO: lambda x, y, z: (x, z),
            geec.crs.VDatumEnum.ELLPS: geec.crs.ellps_to_ortho_height,
        }

        CRS_ORTHO = {
            geec.crs.CRSEnum.ENU: VCRS_ORTHO[self.crs.vdatum],
            geec.crs.CRSEnum.ELLPS: VCRS_ORTHO[self.crs.vdatum],
            geec.crs.CRSEnum.ECEF: lambda x, y, z: (x, z),
            geec.crs.CRSEnum.CART: lambda x, y, z: (x, z),
        }

        self.coords, self.crs = CRS_ORTHO[self.crs.name](
            self.coords, self.geoh, self.crs
        )

    def to_ellipsoid_height(self) -> None:
        """Convert altitude to ellipsoid height,
        depending on coordinates reference system.

        Note: for Cartesian or ECEF coordinate system, do nothing.
        """
        VCRS_ELLPSH = {
            geec.crs.VDatumEnum.ORTHO: geec.crs.ortho_to_ellps_height,
            geec.crs.VDatumEnum.ELLPS: lambda x, y, z: (x, z),
        }

        CRS_ELLPSH = {
            geec.crs.CRSEnum.ENU: VCRS_ELLPSH[self.crs.vdatum],
            geec.crs.CRSEnum.ELLPS: VCRS_ELLPSH[self.crs.vdatum],
            geec.crs.CRSEnum.ECEF: lambda x, y, z: (x, z),
            geec.crs.CRSEnum.CART: lambda x, y, z: (x, z),
        }

        self.coords, self.crs = CRS_ELLPSH[self.crs.name](
            self.coords, self.geoh, self.crs
        )

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
        x_start, x_end, x_step = grid["xstart_xend_xstep"].get(list)
        y_start, y_end, y_step = grid["ystart_yend_ystep"].get(list)
        z_start, z_end, z_step = grid["zstart_zend_zstep"].get(list)

        g = np.mgrid[x_start:x_end:x_step, y_start:y_end:y_step, z_start:z_end:z_step]
        return np.transpose(g.reshape(len(g), -1))
    else:
        msg = "Points must be a list of points or a file"
        logger.error(msg)
        raise TypeError(msg)


# # https://gis.stackexchange.com/a/278636/227256
# import rasterio
# with rasterio.open("dem.tif") as dem:
#     height = 0
#     contours = measure.find_contours(dem.read(1), height)
#         for contour in contours: # find_contours returns "an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour"
#             # Convert each contour found at this height to LineString or GeoJSON and store elevation as required


def get_dataset(config: Subview) -> Dataset:
    coords = _read_coords(config)
    crs = geec.crs.get_crs(config["crs"])

    return Dataset(coords=coords, crs=crs)
