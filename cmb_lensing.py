"""
CMB Lensing Power Spectrum via Limber Approximation
=====================================================
Computes C_L^{kappa kappa} using the halo model matter power spectrum
and the Limber approximation.

The CMB lensing convergence kernel is:
    W^kappa(chi) = (3/2) Omega_m H0^2/c^2 * (chi/a) * (chi_* - chi)/(chi_*)
                 = (3/2) Omega_m (H0/c)^2 * chi * (1+z) * (chi_* - chi)/chi_*

Under the Limber approximation:
    C_L^{kappa kappa} = int_0^{chi_*} dchi  [W^kappa(chi)]^2 / chi^2
                         * P_m( (L+0.5)/chi, z(chi) )

For simplicity, and because we pre-compute the halo model at a single redshift,
we evaluate P_m at z=0 (appropriate at low redshifts).  For full accuracy one
would compute P_m(k,z) = D(z)^2 P_m(k, z=0) or tabulate it over redshift.

Units: C_L is dimensionless (convergence is dimensionless).
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


# Speed of light in km/s (for H0 in km/s/Mpc)
_C_LIGHT_KM_S = 2.99792458e5


def cmb_lensing_kernel(chi, chi_star, cosmo):
    """
    CMB lensing convergence weight function W^kappa(chi).

    W(chi) = (3/2) (H0/c)^2 Omega_m * chi * (1+z) * (chi_star - chi) / chi_star

    Parameters
    ----------
    chi : float or array  [Mpc/h]
        Comoving distance.
    chi_star : float  [Mpc/h]
        Comoving distance to last scattering surface.
    cosmo : Cosmology

    Returns
    -------
    W : float or array  [h/Mpc]  (such that W^2/chi^2 * P has units of [(Mpc/h)^0])
    """
    # H0 / c in h/Mpc:  H0 = 100 h km/s/Mpc => H0/(c km/s/Mpc) = 100h/c => h/Mpc * 100/c
    H0_over_c = cosmo.h * 100.0 / _C_LIGHT_KM_S  # h * (100 km/s/Mpc) / (c km/s) = h/Mpc  ??? let me be careful
    # H0 = 100 h km/s/Mpc
    # c  = 2.998e5 km/s
    # H0/c = 100 h / 2.998e5  Mpc^{-1}
    # In h/Mpc units, distances have factor 1/h absorbed, so H0/c = 100/c h/Mpc? No:
    # Let us be explicit. Using comoving distances in Mpc/h and k in h/Mpc:
    #   H0/c = 100 h km/s/Mpc / (2.998e5 km/s) = h / 2998 Mpc^{-1}
    #   In (h/Mpc) units, 1 h/Mpc = 1 h/Mpc, so H0/c = h/2998 Mpc^{-1} = 1/2998 h/Mpc
    H0_over_c = 1.0 / 2997.92458  # [h/Mpc]^{} -> actually [Mpc/h]^{-1} when chi in Mpc/h

    # z(chi): invert chi(z) numerically; approximate here using linear growth
    # For the kernel we need (1+z); we get z from chi
    # Simple inversion: not computed inline; caller provides z or we use z~0 approx
    # Instead we store z_of_chi as a spline (built in cl_kappakappa)
    # Here chi is passed alongside z information from the caller.
    # This function computes just the geometric part:
    W = (1.5 * cosmo.Omega_m * H0_over_c**2
         * chi * (chi_star - chi) / chi_star)
    # The (1+z) factor from the kernel: W *= (1+z), handled by caller
    return W


def cl_kappakappa(ell, halo_model_pk_spline, cosmo,
                  z_max=5.0, n_chi=300, z_star=1100.0,
                  delta_pk_fn=None):
    """
    CMB lensing convergence power spectrum C_L^{kappa kappa} via Limber integral.

    C_L = int_0^{chi_*} dchi [W^kappa(chi, z)]^2 / chi^2
                              * P_hm((L + 0.5)/chi, z)

    where the Limber approximation maps (L, chi) -> k = (L+0.5)/chi.

    We approximate P_hm(k, z) = D(z)^2 / D(0)^2 * P_hm(k, z=0) using the
    growth factor to include redshift evolution.

    Parameters
    ----------
    ell : float or array
        Multipole moments.
    halo_model_pk_spline : callable
        P_hm(k) spline at z=0 [units: (Mpc/h)^3], k in h/Mpc.
    cosmo : Cosmology
    z_max : float
        Maximum redshift for integration (CMB is at z~1100, but most signal is
        from z < 5).
    n_chi : int
        Number of integration steps in comoving distance.
    z_star : float
        Redshift of last scattering.
    delta_pk_fn : callable or None
        Optional extra power spectrum contribution from a modified HMF,
        with signature delta_pk_fn(k_arr, z_arr) -> array of shape len(k_arr).
        This is added to P_hm(k)*D(z)^2 at each z in the integral.
        Built by make_delta_pk_callable().

    Returns
    -------
    C_L : float or array  (dimensionless)
    """
    ell = np.atleast_1d(np.asarray(ell, dtype=float))

    chi_star = cosmo.comoving_distance(z_star)

    # Build z(chi) mapping
    z_arr = np.linspace(0.001, z_max, n_chi)
    chi_arr = cosmo.comoving_distance(z_arr)

    # Growth factor array
    Dz_arr = np.array([cosmo.growth_factor(zi) for zi in z_arr])
    # D(z=0) = 1 by our normalisation convention
    D0 = 1.0

    # Lensing kernel W^kappa(chi)  [h/Mpc]
    H0_over_c = 1.0 / 2997.92458  # h/Mpc
    W_arr = (1.5 * cosmo.Omega_m * H0_over_c**2
             * chi_arr * (1 + z_arr)
             * (chi_star - chi_arr) / chi_star)

    # dchi/dz for change of variables
    dchi_dz_arr = np.array([cosmo.dchi_dz(zi) for zi in z_arr])

    # Limber integral for each ell
    CL = np.empty(len(ell))
    for i, L in enumerate(ell):
        # k = (L + 0.5) / chi at each chi
        k_limber = (L + 0.5) / chi_arr

        # Only keep chi where k is in our spline range (1e-4, 50) h/Mpc
        k_lo, k_hi = 1e-4, 49.9
        valid = (k_limber >= k_lo) & (k_limber <= k_hi) & (chi_arr > 0)

        if valid.sum() < 2:
            CL[i] = 0.0
            continue

        chi_v = chi_arr[valid]
        W_v = W_arr[valid]
        Dz_v = Dz_arr[valid]
        k_v = k_limber[valid]
        dchi_dz_v = dchi_dz_arr[valid]
        z_v = z_arr[valid]

        Pk_v = halo_model_pk_spline(k_v) * (Dz_v / D0)**2

        # Add extra power from modified HMF (already carries its own z-dependence)
        if delta_pk_fn is not None:
            Pk_v = Pk_v + delta_pk_fn(k_v, z_v)

        # integrand in dz: [W(chi)]^2 / chi^2 * P(k) * dchi/dz
        integrand = W_v**2 / chi_v**2 * Pk_v * dchi_dz_v

        CL[i] = np.trapezoid(integrand, z_v)

    return CL.squeeze() if CL.size == 1 else CL


def cl_kappakappa_linear(ell, pk_lin_spline, cosmo,
                         z_max=5.0, n_chi=300, z_star=1100.0):
    """
    Same as cl_kappakappa but using the linear power spectrum.
    Useful for comparison.
    """
    return cl_kappakappa(ell, pk_lin_spline, cosmo,
                         z_max=z_max, n_chi=n_chi, z_star=z_star)


def make_delta_pk_callable(dndM_extra_fn, cosmo,
                           k_lo=5e-3, k_hi=50.0, n_k=60,
                           z_lo=0.02, z_max=5.0, n_z=50,
                           M_lo=1e10, M_hi=1e16, n_M=60):
    """
    Precompute the extra 1-halo power ΔP(k,z) on a (k, z) grid arising
    from an additional HMF component, then return a fast interpolating callable.

    The extra power is the 1-halo contribution only:

        ΔP_1h(k, z) = ∫ dM (dn_extra/dM)(M, z) * (M / rho_m(z))^2 * |u(k|M,z)|^2

    Parameters
    ----------
    dndM_extra_fn : callable
        Signature: dndM_extra_fn(M_arr, z) -> array [(Mpc/h)^{-3} (M_sun/h)^{-1}].
    cosmo : Cosmology
    k_lo, k_hi : float
        k grid range [h/Mpc].
    n_k : int
        Number of k points.
    z_lo, z_max : float
        Redshift grid range.
    n_z : int
        Number of z points.
    M_lo, M_hi : float
        Halo mass integration limits [M_sun/h].
    n_M : int
        Number of mass points.

    Returns
    -------
    delta_pk_fn : callable
        delta_pk_fn(k_arr, z_arr) -> array of ΔP values, one per element.
    delta_pk_grid : 2-D array, shape (n_k, n_z)
        Raw precomputed grid [(Mpc/h)^3].
    k_grid, z_grid : 1-D arrays
    """
    from halo_model import nfw_fourier
    from scipy.interpolate import RegularGridInterpolator

    k_grid = np.geomspace(k_lo, k_hi, n_k)
    z_grid = np.linspace(z_lo, z_max, n_z)
    M_arr  = np.geomspace(M_lo, M_hi, n_M)

    delta_pk_grid = np.zeros((n_k, n_z))

    for j, z in enumerate(z_grid):
        rho_m = cosmo.rho_m(z)
        dndM_extra = dndM_extra_fn(M_arr, z)
        for i, k in enumerate(k_grid):
            u_arr = nfw_fourier(k, M_arr, z=z, cosmo=cosmo)
            integrand = dndM_extra * (M_arr / rho_m)**2 * u_arr**2
            delta_pk_grid[i, j] = np.trapezoid(integrand, M_arr)

    interp = RegularGridInterpolator(
        (np.log(k_grid), z_grid),
        delta_pk_grid,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    def delta_pk_fn(k_arr, z_arr):
        k_arr = np.atleast_1d(np.asarray(k_arr, dtype=float))
        z_arr = np.atleast_1d(np.asarray(z_arr, dtype=float))
        pts = np.column_stack([np.log(k_arr), z_arr])
        return np.maximum(interp(pts), 0.0)

    return delta_pk_fn, delta_pk_grid, k_grid, z_grid
