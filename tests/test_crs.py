# --- import -----------------------------------
# import from standard lib
import numpy as np
import pymap3d as pm

# import from other lib
import pytest
from pytest import approx

# import from my project
import geec.crs

# np.allclose => absolute(a - b) <= (atol + rtol * absolute(b))
atol = 1.0e-2  # 1e-14  # absolute tolerance
rtol = 0  # relative tolerance


class TestEllipsoid:
    def test_default(self):
        expected = pm.Ellipsoid.from_name("wgs84")
        ellps = geec.crs.Ellipsoid()
        assert ellps.name == expected.model
        assert ellps.semimajor_axis == expected.semimajor_axis
        assert ellps.semiminor_axis == expected.semiminor_axis
        assert ellps.model == expected.name

    def test_from_name(self):
        name = "jupyter"
        expected = pm.Ellipsoid.from_name(name)
        ellps = geec.crs.Ellipsoid(name=name)
        assert ellps.name == expected.model
        assert ellps.semimajor_axis == expected.semimajor_axis
        assert ellps.semiminor_axis == expected.semiminor_axis
        assert ellps.model == expected.name

    def test_from_axis(self):
        a = 6378160.0
        b = 6356774.516
        ellps = geec.crs.Ellipsoid(semimajor_axis=a, semiminor_axis=b)
        assert ellps.name == ""
        assert ellps.semimajor_axis == a
        assert ellps.semiminor_axis == b
        assert ellps.model == ""

    def test_missing_one_axis(self):
        """missing one axis, reutrn default"""
        a = 6378160.0
        expected = pm.Ellipsoid.from_name("wgs84")
        ellps = geec.crs.Ellipsoid(semimajor_axis=a)
        assert ellps.name == expected.model
        assert ellps.semimajor_axis == expected.semimajor_axis
        assert ellps.semiminor_axis == expected.semiminor_axis
        assert ellps.model == expected.name


class TestENU:
    def test_no_keyword(self):
        with pytest.raises(Exception) as exc_info:
            geec.crs.ENU()
        assert exc_info.match(r"missing 4 required keyword-only arguments")


class TestCRS:
    def test_default(self):
        crs = geec.crs.CRS()
        assert crs.name == geec.crs.CRSEnum.ELLPS
        assert crs.ellps is None
        assert crs.enu is None
        assert crs.vdatum == geec.crs.VDatumEnum.ELLPS

    def test_ellispoid_is_uniq(self):
        ellps1 = geec.crs._WGS84_
        ellps2 = geec.crs.Ellipsoid("jupyter")
        enu = geec.crs.ENU(
            longitude_origin=0.0,
            latitude_origin=0.0,
            altitude_origin=0.0,
            ellps=ellps1,
        )
        with pytest.raises(ValueError) as exc_info:
            geec.crs.CRS(name=geec.crs.CRSEnum.ENU, ellps=ellps2, enu=enu)
        assert exc_info.match(r"Reference ellipsoid in ENU")

    # def test_invalid_name(self):
    #     with pytest.raises(KeyError) as exc_info:
    #         crs = geec.crs.CRS(name="toto")
    #     assert exc_info.match(r"Invalid name. Expected one of:")

    # def test_invalid_vdatum(self):
    #     with pytest.raises(Exception) as exc_info:
    #         crs = geec.crs.CRS(name=geec.crs.CRSEnum.ELLPS, vdatum="toto")
    #     assert exc_info.match(r"Invalid name. Expected one of:")


class TestTransform:
    def test_wrap180(self):
        points = np.array(
            [
                [0.0, 90.0, -4138.95],
                [137.50, 88.82, -2246.88],
                [275.015, 88.33, -2114.86],
            ]
        )
        expected = np.array(
            [
                [0.0, 90.0, -4138.95],
                [137.50, 88.82, -2246.88],
                [-84.985, 88.33, -2114.86],
            ]
        )
        result = geec.crs.wrap180(points)
        assert np.allclose(
            result,  # type: ignore[attr-defined]
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )

    def test_geoid_heights(self):
        points = np.array(
            [
                [-62.2138562081, 16.7408514345, 82.2349],
                [-69.6365451, -31.2084545, 2419.1],
                [-119.371255098, 34.408092589, 162.510695809],
            ]
        )
        expected = np.array(
            [
                -41.48,
                30.52,
                -34.45,
            ],
        )
        egm96 = geec.crs.GeoidDatumEnum.EGM96
        crs = geec.crs.CRS(gdatum=egm96)

        result = geec.crs.geoid_heights(points, crs)
        print(f"geoid: {result}")
        assert result == approx(expected, abs=atol)

    def test_ellps_to_ortho_height(self):
        points = np.array(
            [
                [-62.2138562081, 16.7408514345, 82.2349],
                [-69.6365451, -31.2084545, 2419.1],
                [-119.371255098, 34.408092589, 162.510695809],
            ]
        )
        expected = np.array(
            [
                [-62.2138562081, 16.7408514345, 123.72],
                [-69.6365451, -31.2084545, 2388.58],
                [-119.371255098, 34.408092589, 196.96],
            ]
        )
        egm96 = geec.crs.GeoidDatumEnum.EGM96
        crs = geec.crs.CRS(gdatum=egm96)
        geoh = geec.crs.geoid_heights(points, crs)

        result, result_crs = geec.crs.ellps_to_ortho_height(points, geoh, crs)
        assert result == approx(expected, abs=atol)
        assert result_crs.vdatum == geec.crs.VDatumEnum.ORTHO

    def test_ortho_to_ellps_height(self):
        points = np.array(
            [
                [-62.2138562081, 16.7408514345, 123.72],
                [-69.6365451, -31.2084545, 2388.58],
                [-119.371255098, 34.408092589, 196.96],
            ]
        )
        expected = np.array(
            [
                [-62.2138562081, 16.7408514345, 82.2349],
                [-69.6365451, -31.2084545, 2419.1],
                [-119.371255098, 34.408092589, 162.510695809],
            ]
        )
        egm96 = geec.crs.GeoidDatumEnum.EGM96
        crs = geec.crs.CRS(gdatum=egm96)
        geoh = geec.crs.geoid_heights(points, crs)

        result, result_crs = geec.crs.ortho_to_ellps_height(points, geoh, crs)
        print(f"result: {result}")
        assert result == approx(expected, abs=atol)
        assert result_crs.vdatum == geec.crs.VDatumEnum.ELLPS

    def test_wgs84_to_ecef_to_wgs84(self):
        points = np.array(
            [
                [0.0, 90.0, -4138.95],
                [137.50, 88.82, -2246.88],
                [-84.985, 88.33, -2114.86],
            ]
        )
        crs = geec.crs.CRS(ellps=geec.crs._WGS84_)

        ecef_pts, ecef_crs = geec.crs.ellps_to_ecef(points, crs)
        result, result_crs = geec.crs.ecef_to_wgs84(ecef_pts, ecef_crs)

        assert np.allclose(
            result,  # type: ignore[attr-defined]
            points,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        assert result_crs.name == geec.crs.CRSEnum.ELLPS

    def test_enu_to_wgs84_to_enu(self):
        points = np.array(
            [
                [3.9160865488798058e-10, 0.0000000000000000e00, 6.3526133642451800e06],
                [-9.7131216441411307e04, 8.9004361595498893e04, 6.3531487700508945e06],
                [1.6297972541943214e04, -1.8572664693843544e05, 6.3519201771788709e06],
            ]
        )
        lon0, lat0, alt0 = 0.0, 0.0, 0.0
        ellps = geec.crs.Ellipsoid()
        enu = geec.crs.ENU(
            ellps=ellps,
            latitude_origin=lat0,
            longitude_origin=lon0,
            altitude_origin=alt0,
        )
        vdatum = geec.crs.VDatumEnum.ELLPS

        crs = geec.crs.CRS(
            name=geec.crs.CRSEnum.ENU, ellps=ellps, enu=enu, vdatum=vdatum
        )

        wgs84_pts, wgs84_crs = geec.crs.enu_to_wgs84(points, crs)
        result, result_crs = geec.crs.wgs84_to_enu(
            wgs84_pts, wgs84_crs, lon0, lat0, alt0
        )

        assert np.allclose(
            result,  # type: ignore[attr-defined]
            points,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        assert result_crs.name == geec.crs.CRSEnum.ENU
