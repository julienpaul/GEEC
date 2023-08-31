"""
"""

# --- import -----------------------------------
# import from standard lib
from dataclasses import dataclass, field
from typing import Self

# import from other lib
import glm
from confuse import LazyConfig

import geec.crs

# import from my project
import geec.dataset
import geec.polyhedron
import geec.topo
from geec.dataset import Dataset
from geec.polyhedron import Polyhedron
from geec.utils import outerCross


@dataclass(slots=True, kw_only=True)
class Mass:
    density: float  # density of the mass body [kg m-3]
    gravity_constant: float  # gravity constant [m3 kg-1 s-2]
    dataset: Dataset
    topo: bool = False
    extension: float = 180
    wdensity: float = 1025
    polyhedron: Polyhedron = field(init=False)

    def to_orthometric_height(self):
        """Convert altitudes to orthometric heights on mass dataset."""
        self.dataset.to_orthometric_height()

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

    def to_polyhedron(self):
        self.polyhedron = geec.polyhedron.get_polyhedron(self.dataset, self.topo)

    def change_polyhedron(self) -> Self:
        return geec.polyhedron.get_polyhedron_topo(self.polyhedron, self.dataset.geoh)

    def compute_gravity(
        self, observer: glm.vec3, gradient: bool = False, faces: bool = False
    ):
        """
        faces: Only use for testing purposes
        """
        # deepcopy ?
        poly = self.change_polyhedron() if self.topo else self.polyhedron
        # shift points
        poly.points = [p - observer for p in poly.points]

        PQR, DPQR, OMEGA, DOMEGA = poly.compute_gravity(gradient=gradient)

        faces_p1 = [poly.points[i] for i in [ip[0] for ip in poly.fpoints]]
        DP1 = [glm.dot(u, p) for u, p in zip(poly.un, faces_p1, strict=True)]
        density = [self.density if land else self.wdensity for land in poly.fland]
        factor = [-d * self.gravity_constant * 1e5 for d in density]

        # u = [l,m,n]
        # [[l * omega, m * omega, n * omega],..]
        X = [o * u for o, u in zip(OMEGA, poly.un, strict=True)]
        # [[ n*q - m*r, l*r - n*p, m*p - l*q],..]
        Y = [glm.cross(pqr, u) for pqr, u in zip(PQR, poly.un, strict=True)]

        XY = [(x + y) for x, y in zip(X, Y, strict=True)]
        faces_G = [xy * f * dp1 for f, xy, dp1 in zip(factor, XY, DP1, strict=True)]
        G = sum(faces_G)

        if gradient:
            factor = [f * 1e4 for f in factor]  # -density * Gc * 1e9

            # derivative value
            DU = [-u for u in poly.un]

            # [((L*domega[0], L*domega[1], L*domega[2]), (M*domega[0], M*domega[1),..)),..]
            DX = [
                # glm.mat3x3(u.x * do, u.y * do, u.z * do)
                glm.outerProduct(do, u)
                for do, u in zip(DOMEGA, poly.un, strict=True)
            ]
            # [(N * qx - M * rx, L * rx - N * px, M * px - L * qx),(...)]
            DY = [
                glm.transpose(outerCross(u, glm.transpose(dpqr)))
                for dpqr, u in zip(DPQR, poly.un, strict=True)
            ]

            DXY = [(dx + dy) for dx, dy in zip(DX, DY, strict=True)]
            DG = [glm.outerProduct(du, xy) for du, xy in zip(DU, XY, strict=True)]

            # dpqr2= glm.transpose(dpqr)
            # Tx = factor * ( dp1 * (un.x * domega + un.z * dpqr2[1] - un.y * dpqr2[2]) + DU[0] * XY[0].x)
            # Ty = factor * ( dp1 * (un.y * domega + un.x * dpqr2[2] - un.z * dpqr2[0]) + DU[0] * XY[0].y)
            # Tz = factor * ( dp1 * (un.z * domega + un.y * dpqr2[0] - un.x * dpqr2[1]) + DU[0] * XY[0].z)

            # Tx = factor * ( dp1 * (un.x * domega + glm.transpose(DY[0])[0]) + DU[0] * XY[0].x)
            # Ty = factor * ( dp1 * (un.y * domega + glm.transpose(DY[0])[1]) + DU[0] * XY[0].y)
            # Tz = factor * ( dp1 * (un.z * domega + glm.transpose(DY[0])[2]) + DU[0] * XY[0].z)

            # Tx = factor * ( dp1 * (DXY[0]) + DU[0] * XY[0].x)
            # Ty = factor * ( dp1 * (DXY[1]) + DU[0] * XY[0].y)
            # Tz = factor * ( dp1 * (DXY[2]) + DU[0] * XY[0].z)

            # Tx = factor * ( dp1 * (DXY[0]) + DG[0])
            # Ty = factor * ( dp1 * (DXY[1]) + DG[1])
            # Tz = factor * ( dp1 * (DXY[2]) + DG[2])

            # WARNING: txy != tyx and tzy != tyz on faces_T

            faces_T = [
                f * (dp1 * dxy + dg)
                for f, dp1, dxy, dg in zip(factor, DP1, DXY, DG, strict=True)
            ]
            T = sum(faces_T)

        else:
            faces_T = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
            T = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

        if faces:
            return (faces_G, faces_T)
        else:
            return (G, T)


# def get_mass(config: LazyConfig) -> Mass:
#     """Read mass body parameters from config file"""
#     masscfg = config["mass"]
#     dataset = geec.dataset.get_dataset(masscfg)
#     density = masscfg["density"].get(float)
#     gravity_constant = masscfg["gravity_constant"].get(float)
#
#     mass = Mass(density=density, gravity_constant=gravity_constant, dataset=dataset)
#
#     return mass


def get_masses(config: LazyConfig) -> list[Mass]:
    masses = []
    for masscfg in config["masses"]:
        dataset = geec.dataset.get_dataset(masscfg)
        density = masscfg["density"].as_number()  # get(float)
        gravity_constant = masscfg["gravity_constant"].as_number()  # get(float)
        topo, ext, wdensity = geec.topo.get_topo(masscfg["topo"])
        mass = Mass(
            density=density,
            gravity_constant=gravity_constant,
            dataset=dataset,
            topo=topo,
            extension=ext,
            wdensity=wdensity,
        )
        masses.append(mass)

    return masses


def to_orthometric_height(masses: list[Mass]) -> None:
    """Convert altitudes to orhometric heights on each mass datasets."""
    [mass.dataset.to_orthometric_height() for mass in masses]


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


def get_polyhedron(masses: list[Mass]) -> None:
    [mass.to_polyhedron() for mass in masses]


# # compute the solid angle
# compute_gravity = partial(_compute_gravity, gradient=False)
# # compute the solid angle and its derivatives
# compute_gravity_and_gradient = partial(_compute_gravity, gradient=True)

if __name__ == "__main__":
    pass
