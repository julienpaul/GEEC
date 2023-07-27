"""
    The crs module sets up the Coordinates Reference sysytem class,

    Example usage:
"""

# --- import -----------------------------------
# import from standard lib
import importlib.resources as resources
from dataclasses import dataclass, field
from enum import Enum, StrEnum

# import from other lib
import confuse
import numpy as np
import numpy.typing as npt
import pygeodesy as pgeo
import pymap3d as pm
from confuse import Subview
from loguru import logger

# import from my project


# Set up enumerators for CRS, VDatum, and Ellipsoid model
class CRSEnum(StrEnum):
    CART = "cart"
    ELLPS = "ellps"
    ENU = "enu"
    ECEF = "ecef"


class VDatumEnum(StrEnum):
    ELLPS = "ellps"
    EGM84 = "egm84"
    EGM96 = "egm96"
    EGM08 = "egm2008"


EllModelEnum = Enum("EllModelEnum", pm.Ellipsoid.models)


# Set up class for Ellipsoid
class Ellipsoid(pm.Ellipsoid):
    """Ellipsoid coordinates reference system class"""

    def __init__(
        self,
        name: str | None = None,
        semimajor_axis: float | None = None,
        semiminor_axis: float | None = None,
        model: EllModelEnum = EllModelEnum["wgs84"],
    ):
        if name:
            # set up reference ellipsoid
            super().__init__(
                semimajor_axis=super().models[name]["a"],
                semiminor_axis=super().models[name]["b"],
                model=super().models[name]["name"],
                name=name,
            )
        elif semimajor_axis and semiminor_axis:
            # set up reference ellipsoid
            super().__init__(
                semimajor_axis=semimajor_axis,
                semiminor_axis=semiminor_axis,
                model="",
                name="",
            )
        else:
            # set up default reference ellipsoid
            name = "wgs84"
            super().__init__(
                semimajor_axis=super().models[name]["a"],
                semiminor_axis=super().models[name]["b"],
                model=super().models[name]["name"],
                name=name,
            )


# default ellipsoid
_WGS84_ = Ellipsoid()


# Set up data classes CRS, ENU
@dataclass(frozen=True, slots=True, kw_only=True)
class ENU:
    """Local East North Up coordinates reference system class"""

    # Local ENU (ref = enu)
    longitude_origin: float  # longitude express in ellipsoid reference
    latitude_origin: float  # latitude express in ellipsoid reference
    altitude_origin: float  # height above ellipsoid reference
    ellps: pm.Ellipsoid


@dataclass(frozen=False, slots=True, kw_only=True)
class CRS:
    """Coordinates reference system class"""

    name: CRSEnum = field(default=CRSEnum.ELLPS)
    ellps: Ellipsoid | None = None
    enu: ENU | None = None

    # Vertical Datum (default Ellipsoid, mean no change in vertical position)
    vdatum: VDatumEnum = field(default=VDatumEnum.ELLPS)

    def __post_init__(self):
        # if self.name not in CRSEnum:
        #     raise KeyError(f"Invalid name. Expected one of: {CRSEnum.__members__}")
        # if self.vdatum not in VDatumEnum:
        #     raise KeyError(f"Invalid vdatum. Expected one of: {VDatumEnum.__members__}")

        if self.name == CRSEnum.ENU and self.enu.ellps != self.ellps:
            msg = (
                f"Reference ellipsoid in ENU '{self.enu.ellps.name}' must be the"
                f" same as in CRS '{self.ellps.name}'"
            )
            logger.error(msg)
            raise ValueError(msg)


# functions for getting parameters from configuration file
def read_ellipsoid_parameters(config: Subview) -> tuple:
    """Read ellipsoid parameters from 'crs' subview (from configuration file)"""
    name = config["name"].get([str, None])
    semimajor_axis = config["semimajor_axis"].get([float, None])
    semiminor_axis = config["semiminor_axis"].get([float, None])
    if name and name not in EllModelEnum._member_names_:
        msg = (
            f"Invalid Ellipsoid name: {name}. It must be one of"
            f" {EllModelEnum._member_names_}."
        )
        logger.error(msg)
        raise confuse.ConfigValueError(msg)

    return (name, semimajor_axis, semiminor_axis)


def read_enu_parameters(config: Subview) -> tuple:
    """Read local ENU parameters from 'crs' subview (from configuration file)"""
    longitude_org = config["longitude_origin"].get(float)
    latitude_org = config["latitude_origin"].get(float)
    altitude_org = config["altitude_origin"].get(float)

    return (longitude_org, latitude_org, altitude_org)


def read_ref(config: Subview) -> CRSEnum:
    """Read the coordinates reference system from 'crs' subview (from configuration file)
    """
    ref = config["ref"].get(confuse.Optional(confuse.String("ellps"))).upper()
    if ref not in CRSEnum.__members__:
        msg = (
            f"Invalid Coordinates Refence System ref: {ref}. It must be one of"
            f" {CRSEnum._member_names_}."
        )
        logger.error(msg)
        raise confuse.ConfigValueError(msg)

    return CRSEnum[ref]


def read_vdatum(config: Subview) -> VDatumEnum:
    """Read the vertical datum from 'crs' subview (from configuration file)"""
    vdatum = config["vdatum"].get(confuse.Optional(confuse.String("ellps"))).upper()
    if vdatum not in VDatumEnum.__members__:
        msg = (
            f"Invalid Earth Gravitational Model: {vdatum}. It must be one of"
            f" {VDatumEnum._member_names_}"
        )
        logger.error(msg)
        raise confuse.ConfigValueError(msg)

    return VDatumEnum[vdatum]


# functions for creating CRS object instance
def create_cartesian_crs(config: Subview) -> CRS:
    """Create a cartesian CRS object from 'crs' subview (from configuration file)"""
    return CRS(name=CRSEnum.CART)


def create_ecef_crs(config: Subview) -> CRS:
    """Create a ECEF CRS object from 'crs' subview (from configuration file)"""
    return CRS(name=CRSEnum.ECEF)


def create_ellipsoid_crs(config: Subview) -> CRS:
    """Create an ellipsoid CRS object from 'crs' subview (from configuration file)"""
    name, semimajor_axis, semiminor_axis = read_ellipsoid_parameters(config)
    vdatum = read_vdatum(config)
    ellps = Ellipsoid(
        name=name, semimajor_axis=semimajor_axis, semiminor_axis=semiminor_axis
    )
    return CRS(name=CRSEnum.ELLPS, ellps=ellps, vdatum=vdatum)


def create_local_enu_crs(config: Subview) -> CRS:
    """Create a local ENU CRS object from 'crs' subview (from configuration file)"""
    name, semimajor_axis, semiminor_axis = read_ellipsoid_parameters(config)
    lon_org, lat_org, alt_org = read_enu_parameters(config)
    vdatum = read_vdatum(config)
    ellps = Ellipsoid(
        name=name, semimajor_axis=semimajor_axis, semiminor_axis=semiminor_axis
    )
    enu = ENU(
        ellps=ellps,
        longitude_origin=lon_org,
        latitude_origin=lat_org,
        altitude_origin=alt_org,
    )
    return CRS(name=CRSEnum.ENU, ellps=ellps, enu=enu, vdatum=vdatum)


CRSREF = {
    "cart": create_cartesian_crs,
    "ellps": create_ellipsoid_crs,
    "enu": create_local_enu_crs,
    "ecef": create_ecef_crs,
}


def get_crs(config: Subview) -> CRS:
    """Create a CRS object from 'crs' subview (from configuration file)"""
    ref = read_ref(config)
    f = CRSREF[ref]
    return f(config)


# functions for transforming CRS objects
def wrap180(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """wrap longitude coordinates to -180..180"""
    logger.info("Convert points longitude to -180..180")
    result = (x - 360 if x > 180 else x for x in points[:, 0])
    points[:, 0] = list(result)
    return points


data_path = resources.files("geec.data")
egm08geoid = pgeo.GeoidKarney(data_path / "geoids/egm2008-2_5.pgm")
egm96geoid = pgeo.GeoidKarney(data_path / "geoids/egm96-5.pgm")
egm84geoid = pgeo.GeoidKarney(data_path / "geoids/egm84-15.pgm")


def ellpsgeoid(x: pgeo.ellipsoidalKarney.LatLon):
    return 0


VDATUMREF = {
    VDatumEnum.ELLPS: ellpsgeoid,
    VDatumEnum.EGM84: egm84geoid,
    VDatumEnum.EGM96: egm96geoid,
    VDatumEnum.EGM08: egm08geoid,
}


def ellipsoid_height(
    points: npt.NDArray[np.float64], crs: CRS
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Compute ellipsoid height and overwrite height values with it.

    Height value assume to be orthometric height
    """
    logger.info("Convert points altitude to ellipsoid height")
    func = VDATUMREF[crs.vdatum]
    positions = [pgeo.ellipsoidalKarney.LatLon(lt, ln) for ln, lt, h in points]
    geoh = func(positions)

    # ellipsoid height (from GPS) = orthometric height + geoid height
    points.T[2] += geoh

    # update coordinates reference system
    crs.vdatum = VDatumEnum.ELLPS

    return points, crs


def ellps_to_ecef(
    points: npt.NDArray[np.float64], crs: CRS
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Convert points from ellipsoid to ECEF coordinates system"""
    logger.info("Convert points cooridnates from ellipsoid to ECEF")
    lon, lat, alt = points.T
    x, y, z = pm.geodetic2ecef(lat, lon, alt, ell=crs.ellps)

    points = np.array([x, y, z]).T

    # update coordinates reference system
    crs.name = CRSEnum.ECEF
    crs.ellps = None
    crs.enu = None

    return points, crs


def enu_to_ecef(
    points: npt.NDArray[np.float64], crs: CRS
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Convert points from ENU coordinates system to ECEF coordinates system

    ENU origin from crs
    """
    logger.info("Convert points cooridnates from ENU to WGS84 ellipsoid")
    lat0 = crs.enu.latitude_origin
    lon0 = crs.enu.longitude_origin
    h0 = crs.enu.altitude_origin

    e, n, u = points.T
    x, y, z = pm.enu2ecef(e, n, u, lat0, lon0, h0, ell=crs.ellps)

    points = np.array([x, y, z]).T

    # update coordinates reference system
    crs.name = CRSEnum.ECEF
    crs.ellps = None
    crs.enu = None

    return points, crs


def ecef_to_wgs84(
    points: npt.NDArray[np.float64], crs: CRS
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Convert points from ECEF coordinates system to WGS84 ellipsoid"""
    # if crs.enu is not None or crs.ellps is not None:
    #     msg = (
    #         "Body points not in in ECEF coordinates. Got crs.ellps: {crs.ellps} and"
    #         " crs.enu: {crs.enu}."
    #     )
    #     logger.error(msg)
    #     raise ValueError(msg)

    logger.info("Convert points cooridnates from ECEF to WGS84 ellipsoid")
    x, y, z = points.T
    lat, lon, alt = pm.ecef2geodetic(x, y, z)  # by default to 'wgs84'

    points = np.array([lon, lat, alt]).T

    # update coordinates reference system
    crs.name = CRSEnum.ELLPS
    crs.ellps = _WGS84_
    crs.enu = None

    return points, crs


def ellps_to_wgs84(
    points: npt.NDArray[np.float64], crs: CRS
) -> tuple[npt.NDArray[np.float64], CRS]:
    """convert points from ellipsoid to WGS84 ellipsoid"""
    if crs.enu is None and crs.ellps == _WGS84_:
        logger.warning("Body points already in WGS84 ellipsoid.")
    else:
        logger.info("Convert points cooridnates from ellipsoid to WGS84 ellipsoid")
        points, crs = ellps_to_ecef(points, crs)
        points, crs = ecef_to_wgs84(points, crs)

    return points, crs


def enu_to_wgs84(
    points: npt.NDArray[np.float64], crs: CRS
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Convert points from ENU coordinates system to WGS84 ellipsoid

    ENU origin from crs
    """
    logger.info("Convert points cooridnates from ENU to WGS84 ellipsoid")
    points, crs = enu_to_ecef(points, crs)
    points, crs = ecef_to_wgs84(points, crs)

    return points, crs


def ecef_to_enu(
    points: npt.NDArray[np.float64], crs: CRS, lon0: float, lat0: float, alt0: float
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Transform and overwrite points from ECEF to ENU coordinates.

    ENU origin: lon0, lat0, alt0
    """
    # if crs.enu is not None or crs.ellps is not None:
    #     msg = (
    #         "Body points not in in ECEF coordinates. Got crs.ellps: {crs.ellps} and"
    #         " crs.enu: {crs.enu}."
    #     )
    #     logger.error(msg)
    #     raise ValueError(msg)

    logger.info("Convert points from ECEF to ENU coorindates.")
    x, y, z = points.T
    e, n, u = pm.ecef2enu(x, y, z, lat0, lon0, alt0)

    points = np.array([e, n, u]).T

    # update coordinates reference system
    crs.name = CRSEnum.ENU
    crs.enu = ENU(
        ellps=_WGS84_,
        longitude_origin=lon0,
        latitude_origin=lat0,
        altitude_origin=alt0,
    )

    return points, crs


def wgs84_to_enu(
    points: npt.NDArray[np.float64], crs: CRS, lon0: float, lat0: float, alt0: float
) -> tuple[npt.NDArray[np.float64], CRS]:
    """Transform and overwrite points from WGS84 ellipsoid to ENU coordinates.

    ENU origin: lon0, lat0, alt0
    """
    # if crs.enu is not None or crs.ellps != _WGS84_:
    #     msg = (
    #         "Body points must be in WGS84 ellipsoid to be transformed to ENU"
    #         f" coordinates. Got crs.ellps: {crs.ellps} and crs.enu:"
    #         f" {crs.enu}."
    #     )
    #     logger.error(msg)
    #     raise ValueError(msg)

    logger.info("Convert points from WGS84 ellipsoid to ENU coorindates.")
    lon, lat, alt = points.T
    e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)

    points = np.array([e, n, u]).T

    # update coordinates reference system
    crs.name = CRSEnum.ENU
    crs.enu = ENU(
        ellps=_WGS84_,
        longitude_origin=lon0,
        latitude_origin=lat0,
        altitude_origin=alt0,
    )

    return points, crs
