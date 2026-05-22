# fargopy/tests/test___interp.py
import os
import numpy as np
import pytest
import fargopy as fp


@pytest.fixture(scope="session")
def sim():
    # Load the local test simulation only once per test session
    fp.Simulation.download_precomputed("p3disoj")
    return fp.Simulation(output_dir=f"/tmp/p3disoj")


@pytest.fixture(scope="session")
def sim2d():
    # 2D Fargo3D simulation used to validate the no-slice loading path
    fp.Simulation.download_precomputed("fargo")
    return fp.Simulation(output_dir="/tmp/fargo")


def test_interpolacion_1d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="phi=0,theta=1.56",
        snapshot=(1, 2),
    )
    x = 1.2
    valor = data.evaluate(var1=x, snapshot=1.2)
    # If it returns a 0-d array or a scalar, both are acceptable; it must be finite
    assert np.isfinite(valor).all(), "1D interpolation must return a finite value"


def test_interpolacion_1d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="phi=0,theta=1.56",
        snapshot=(1, 2),
    )
    x = np.array([1.2, 1.3, 1.4])
    valor = data.evaluate(var1=x, snapshot=1.2)
    assert np.asarray(valor).shape == x.shape, (
        "1D interpolation must preserve the input shape"
    )
    assert np.isfinite(valor).any(), (
        "At least some points should be finite if they lie inside the domain"
    )


def test_interpolacion_2d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="theta=1.56",
        snapshot=(1, 2),
    )
    # Conservative point (typically inside the domain)
    x = 1.2
    y = 0.14
    valor = data.evaluate(
        var1=x, var2=y, snapshot=1.2, interpolator="griddata", method="nearest"
    )
    assert np.isscalar(valor) or np.asarray(valor).shape == (), (
        "2D point interpolation must return a scalar/0-d value"
    )
    assert np.isfinite(valor), (
        "2D point interpolation must return a finite value (nearest)"
    )


def test_interpolacion_2d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        slice="theta=1.56",
        snapshot=(1, 2),
    )

    # Your y=[1.3,1.4,1.5] values are very likely outside the domain -> NaNs (linear) or spurious values.
    # Here we use "safe" points and, additionally, nearest-neighbor interpolation to avoid NaNs
    # due to the convex hull limitation.
    x = np.array([0.9, 1.0, 1.1])
    y = np.array([0.05, 0.10, 0.15])

    valor = data.evaluate(
        var1=x, var2=y, snapshot=1.2, interpolator="griddata", method="nearest"
    )
    valor = np.asarray(valor)

    assert valor.shape == x.shape, (
        "2D interpolation must return an array with the same shape as the input"
    )
    assert np.isfinite(valor).all(), (
        "With nearest and in-domain points, NaNs should not appear"
    )


# FAILING
def test_interpolacion_3d_point(sim):
    data = sim.load_field(
        fields="gasdens",
        snapshot=(1, 2),
    )
    x, y, z = 1.2, 1.3, 1.4
    valor = data.evaluate(
        var1=x, var2=y, var3=z, snapshot=1.2, interpolator="griddata", method="nearest"
    )
    assert np.isscalar(valor) or np.asarray(valor).shape == (), (
        "3D point interpolation must return a scalar/0-d value"
    )
    assert np.isfinite(valor), (
        "3D point interpolation must return a finite value (nearest)"
    )


def test_interpolacion_3d_array(sim):
    data = sim.load_field(
        fields="gasdens",
        snapshot=(1, 2),
    )
    x = np.array([1.2, 1.3, 1.4])
    y = np.array([1.3, 1.4, 1.5])
    z = np.array([0.024, 0.14, 0.2])

    valor = data.evaluate(
        var1=x, var2=y, var3=z, snapshot=1.2, interpolator="griddata", method="nearest"
    )
    valor = np.asarray(valor)

    assert valor.shape == x.shape, (
        "3D interpolation must return an array with the same shape as the input"
    )
    assert np.isfinite(valor).all(), (
        "With nearest, NaNs should not appear except for extremely out-of-domain points"
    )


def test_interpolacion_2d_no_slice_polar(sim2d):
    data = sim2d.load_field(
        fields="gasdens",
        snapshot=0,
        coords="polar",
    )

    assert data.dim == 2
    assert data._original_coords == "polar"

    row = data.df.iloc[0]
    phi_mesh = np.asarray(row.var1_mesh)
    r_mesh = np.asarray(row.var2_mesh)

    assert phi_mesh.shape == r_mesh.shape
    assert np.asarray(row.var3_mesh).shape == phi_mesh.shape

    phi = float(phi_mesh[phi_mesh.shape[0] // 2, phi_mesh.shape[1] // 2])
    r = float(r_mesh[r_mesh.shape[0] // 2, r_mesh.shape[1] // 2])
    valor = data.evaluate(var1=phi, var2=r, snapshot=0, interpolator="griddata", method="nearest")

    assert np.isfinite(valor), "2D polar no-slice interpolation must stay finite"


def test_interpolacion_2d_no_slice_cartesian(sim2d):
    data = sim2d.load_field(
        fields="gasdens",
        snapshot=0,
        coords="cartesian",
    )

    assert data.dim == 2
    assert data._original_coords == "cartesian"

    row = data.df.iloc[0]
    x_mesh = np.asarray(row.var1_mesh)
    y_mesh = np.asarray(row.var2_mesh)

    assert x_mesh.shape == y_mesh.shape
    assert np.asarray(row.var3_mesh).shape == x_mesh.shape

    x = float(x_mesh[x_mesh.shape[0] // 2, x_mesh.shape[1] // 2])
    y = float(y_mesh[y_mesh.shape[0] // 2, y_mesh.shape[1] // 2])
    valor = data.evaluate(var1=x, var2=y, snapshot=0, interpolator="griddata", method="nearest")

    assert np.isfinite(valor), "2D cartesian no-slice interpolation must stay finite"


def test_interpolacion_2d_no_snapshot_uses_latest(sim2d):
    data = sim2d.load_field(
        fields="gasdens",
        coords="cartesian",
    )

    assert data.snapshot[0] == sim2d.nsnaps - 1

    row = data.df.iloc[0]
    arr = np.asarray(row.gasdens_mesh)
    assert np.nanstd(arr) > 0, "Default 2D load should not return a uniform initial state"


def test_interpolacion_2d_cartesian_full_mesh_no_nan_seam(sim2d):
    data = sim2d.load_field(
        fields="gasdens",
        snapshot=10,
        coords="cartesian",
    )

    row = data.df.iloc[0]
    x_mesh = np.asarray(row.var1_mesh)
    y_mesh = np.asarray(row.var2_mesh)

    valor = data.evaluate(
        var1=x_mesh,
        var2=y_mesh,
        snapshot=10,
        interpolator="griddata",
        method="linear",
    )

    assert np.isfinite(valor).all(), "The full 2D cartesian mesh must not leave a NaN seam"
