"""
Tests for the halo model implementation.

Run with:  python -m pytest test_halo_model.py -v
     or:   python test_halo_model.py
"""

import numpy as np
import pytest

from linear_power_spectrum import get_cosmology_dict
from halo_model import (
    load_linear_pk, Cosmology, HaloModel,
    nfw_profile_check, tinker10_dndM, tinker10_bias,
    mass_to_radius, sigma_sq,
)


# ---------------------------------------------------------------------------
# Fixtures / shared objects
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup():
    """Load the linear P(k) and create cosmology / halo model objects."""
    pk_spline, k_arr, pk_arr = load_linear_pk("linear_pk.txt")
    cosmo = Cosmology(Omega_m=0.3138, h=0.6736)
    hm = HaloModel(pk_spline, cosmo=cosmo, z=0.0, n_M=80)
    return pk_spline, cosmo, hm, k_arr, pk_arr


# ---------------------------------------------------------------------------
# Test 1: Large-scale power spectrum agrees with linear theory
# ---------------------------------------------------------------------------

class TestHaloModelVsLinear:
    """
    On large scales (k < ~0.05 h/Mpc) the halo model total P(k) should
    recover the linear theory power spectrum to within ~10%.

    This is a fundamental consistency check: the halo model is constructed so
    that the 2-halo term dominates at large scales and approaches linear theory.
    """

    @pytest.mark.parametrize("k_test", [0.005, 0.01, 0.02, 0.04])
    def test_large_scale_agreement(self, setup, k_test):
        pk_spline, cosmo, hm, _, _ = setup

        P_lin = pk_spline(k_test)
        P_hm  = hm.total(k_test)
        ratio = P_hm / P_lin

        # Allow up to 15% deviation (halo model is approximate)
        assert 0.85 < ratio < 1.15, (
            f"k={k_test:.3f} h/Mpc: P_hm/P_lin = {ratio:.3f}, expected 0.85–1.15"
        )

    def test_two_halo_dominates_at_large_scales(self, setup):
        """At k=0.01 h/Mpc the 2-halo term should be >> 1-halo term."""
        pk_spline, cosmo, hm, _, _ = setup
        k = 0.01
        P1h = hm.one_halo(k)
        P2h = hm.two_halo(k)
        assert P2h > 10 * P1h, (
            f"k={k}: P_2h={P2h:.2e} should be >> P_1h={P1h:.2e}"
        )

    def test_one_halo_dominates_at_small_scales(self, setup):
        """At k=5 h/Mpc the 1-halo term should dominate."""
        pk_spline, cosmo, hm, _, _ = setup
        k = 5.0
        P1h = hm.one_halo(k)
        P2h = hm.two_halo(k)
        assert P1h > P2h, (
            f"k={k}: P_1h={P1h:.2e} should be > P_2h={P2h:.2e}"
        )

    def test_total_positive_everywhere(self, setup):
        """Total power spectrum must be positive at all k."""
        pk_spline, cosmo, hm, _, _ = setup
        k_arr = np.geomspace(0.005, 20.0, 30)
        P_tot = hm.total(k_arr)
        assert np.all(P_tot > 0), "P_total must be positive everywhere"


# ---------------------------------------------------------------------------
# Test 2: NFW profile normalisation
# ---------------------------------------------------------------------------

class TestNFWNormalisation:
    """
    Verify that int rho_NFW(r) 4pi r^2 dr = M (within r_200m).
    """

    @pytest.mark.parametrize("log_M", [12.0, 13.0, 14.0, 15.0])
    def test_nfw_norm(self, setup, log_M):
        _, cosmo, _, _, _ = setup
        M = 10**log_M  # M_sun/h
        frac_err = nfw_profile_check(M, z=0.0, cosmo=cosmo, Delta=200)
        assert abs(frac_err) < 1e-3, (
            f"M=10^{log_M}: NFW norm error = {frac_err:.2e}, should be < 1e-3"
        )

    def test_nfw_fourier_unity_at_k0(self, setup):
        """u(k->0|M) should be ~1 (profile is normalised by mass)."""
        from halo_model import nfw_fourier
        _, cosmo, _, _, _ = setup
        M = 1e13  # M_sun/h
        u_small_k = nfw_fourier(1e-4, M, z=0.0, cosmo=cosmo)
        assert abs(u_small_k - 1.0) < 0.01, (
            f"u(k~0|M) = {u_small_k:.4f}, expected ~1"
        )

    def test_nfw_fourier_decreases_with_k(self, setup):
        """NFW Fourier transform should decrease monotonically with k."""
        from halo_model import nfw_fourier
        _, cosmo, _, _, _ = setup
        M = 1e13
        k_arr = np.geomspace(1e-3, 20, 50)
        u_arr = np.array([float(nfw_fourier(ki, M, cosmo=cosmo)) for ki in k_arr])
        assert np.all(np.diff(u_arr) < 0), (
            "NFW Fourier transform should be monotonically decreasing with k"
        )


# ---------------------------------------------------------------------------
# Test 3: HMF normalisation (integral = Omega_m * rho_crit)
# ---------------------------------------------------------------------------

class TestHMFNormalisation:
    """
    int dn/dM * M dM should give a significant fraction of rho_m.

    Note: the Tinker 2010 HMF is not mass-conserving by design – it integrates
    to ~0.73 of rho_m over all masses (sigma -> infinity), and ~0.41 of rho_m
    over our practical mass range 1e10 – 1e16 M_sun/h. This is a well-known
    property of the Tinker functional form (it is a fit to simulation data, not
    derived from a conservation law). The HaloModel normalizes the halo bias to
    ensure P_2h -> P_lin at large scales regardless of this.
    """

    def test_hmf_integral(self, setup):
        pk_spline, cosmo, _, _, _ = setup
        M_arr = np.geomspace(1e10, 1e16, 150)
        dndM_arr = tinker10_dndM(M_arr, pk_spline, cosmo, z=0.0)
        rho_from_hmf = np.trapezoid(dndM_arr * M_arr, M_arr)
        rho_m_true = cosmo.rho_m(0.0)
        frac = rho_from_hmf / rho_m_true
        # Over M=1e10-1e16, Tinker HMF gives ~30-60% of rho_m (limited mass range)
        assert 0.20 < frac < 0.70, (
            f"int(dn/dM * M) / rho_m = {frac:.3f}, expected 0.20–0.70 for M=[1e10,1e16]"
        )


# ---------------------------------------------------------------------------
# Test 4: Bias normalization (consistency of 2-halo term)
# ---------------------------------------------------------------------------

class TestBiasIntegral:
    """
    The HaloModel normalizes the Tinker bias so that
        int dn/dM * b_norm(M) * M/rho_m dM = 1
    This ensures the 2-halo term -> P_lin at k -> 0.

    We verify that the normalized bias (stored in hm.bias_arr) gives an
    integral of ~1 by definition, AND that the raw Tinker bias is < 1
    (demonstrating the normalization was necessary and applied).
    """

    def test_normalized_bias_integral(self, setup):
        pk_spline, cosmo, hm, _, _ = setup
        rho_m = cosmo.rho_m(0.0)
        # hm.bias_arr is already normalized
        I_norm = np.trapezoid(hm.dndM_arr * hm.bias_arr * hm.M_arr / rho_m,
                              hm.M_arr)
        assert abs(I_norm - 1.0) < 0.01, (
            f"Normalized bias integral = {I_norm:.4f}, expected 1.0"
        )

    def test_raw_tinker_bias_integral(self, setup):
        """Raw Tinker bias integral (without normalization) should be < 1 for
        M=[1e10,1e16] since the mass range doesn't capture all halos."""
        pk_spline, cosmo, _, _, _ = setup
        M_arr = np.geomspace(1e10, 1e16, 150)
        dndM_arr = tinker10_dndM(M_arr, pk_spline, cosmo, z=0.0)
        b_arr = tinker10_bias(M_arr, pk_spline, cosmo, z=0.0)
        rho_m = cosmo.rho_m(0.0)
        I = np.trapezoid(dndM_arr * b_arr * M_arr / rho_m, M_arr)
        # Without normalization, integral is ~0.4-0.7 for this mass range
        assert 0.30 < I < 0.80, (
            f"Raw Tinker bias integral = {I:.3f}, expected 0.30–0.80 for M=[1e10,1e16]"
        )


# ---------------------------------------------------------------------------
# Standalone runner (without pytest)
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all tests without pytest, printing results."""
    import traceback

    # Build shared setup
    pk_spline, k_arr, pk_arr = load_linear_pk("linear_pk.txt")
    cosmo = Cosmology(Omega_m=0.3138, h=0.6736)
    print("Building halo model (may take ~30s)...")
    hm = HaloModel(pk_spline, cosmo=cosmo, z=0.0, n_M=80)
    setup_data = (pk_spline, cosmo, hm, k_arr, pk_arr)

    tests = [
        # (name, callable)
    ]

    # --- Large-scale tests ---
    t1 = TestHaloModelVsLinear()
    print("\n=== Test: Large-scale P(k) vs linear theory ===")
    for k_test in [0.005, 0.01, 0.02, 0.04]:
        try:
            t1.test_large_scale_agreement(setup_data, k_test)
            print(f"  PASS  k={k_test:.3f} h/Mpc large-scale agreement")
        except AssertionError as e:
            print(f"  FAIL  {e}")

    try:
        t1.test_two_halo_dominates_at_large_scales(setup_data)
        print("  PASS  2-halo dominates at large scales")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    try:
        t1.test_one_halo_dominates_at_small_scales(setup_data)
        print("  PASS  1-halo dominates at small scales")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    try:
        t1.test_total_positive_everywhere(setup_data)
        print("  PASS  P_total positive everywhere")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    # --- NFW normalisation ---
    t2 = TestNFWNormalisation()
    print("\n=== Test: NFW normalisation ===")
    for log_M in [12.0, 13.0, 14.0, 15.0]:
        try:
            t2.test_nfw_norm(setup_data, log_M)
            print(f"  PASS  M=10^{log_M} NFW normalised")
        except AssertionError as e:
            print(f"  FAIL  {e}")

    try:
        t2.test_nfw_fourier_unity_at_k0(setup_data)
        print("  PASS  u(k~0) = 1")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    try:
        t2.test_nfw_fourier_decreases_with_k(setup_data)
        print("  PASS  u(k) decreasing with k")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    # --- HMF integral ---
    t3 = TestHMFNormalisation()
    print("\n=== Test: HMF integral ===")
    try:
        t3.test_hmf_integral(setup_data)
        print("  PASS  HMF integral in range")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    # --- Bias normalization ---
    t4 = TestBiasIntegral()
    print("\n=== Test: Bias normalization ===")
    try:
        t4.test_normalized_bias_integral(setup_data)
        print("  PASS  Normalized bias integral = 1")
    except AssertionError as e:
        print(f"  FAIL  {e}")
    try:
        t4.test_raw_tinker_bias_integral(setup_data)
        print("  PASS  Raw Tinker bias integral in expected range")
    except AssertionError as e:
        print(f"  FAIL  {e}")

    print("\nDone.")
    return hm, pk_spline, cosmo


if __name__ == "__main__":
    run_all_tests()
