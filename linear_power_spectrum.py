"""
Compute the linear matter power spectrum at Planck 2018 cosmology using CAMB
and save it to a file for use in the halo model calculation.

Planck 2018 parameters from Table 2 of Planck 2018 cosmological parameters paper
(Planck Collaboration 2018, arXiv:1807.06209), TT,TE,EE+lowE+lensing column.
"""

import numpy as np
import camb


# ---------------------------------------------------------------------------
# Planck 2018 best-fit cosmological parameters (base LCDM)
# ---------------------------------------------------------------------------
PLANCK18 = dict(
    H0=67.36,           # km/s/Mpc
    ombh2=0.02237,      # Omega_b h^2
    omch2=0.1200,       # Omega_c h^2
    ns=0.9649,          # scalar spectral index
    As=2.100e-9,        # scalar amplitude (at k_pivot = 0.05 Mpc^-1)
    tau=0.0544,         # optical depth to reionization
    mnu=0.06,           # sum of neutrino masses (eV), minimal normal hierarchy
    nnu=3.046,          # effective number of relativistic species
)


def get_camb_results(params=None, kmax=50.0, npoints=500, z=0.0):
    """
    Run CAMB and return the matter power spectrum P(k) at redshift z.

    Parameters
    ----------
    params : dict, optional
        Cosmological parameters.  Defaults to Planck 2018.
    kmax : float
        Maximum wavenumber [h/Mpc].
    npoints : int
        Number of k points.
    z : float
        Redshift.

    Returns
    -------
    k_arr : ndarray
        Wavenumbers [h/Mpc].
    pk_arr : ndarray
        Linear matter power spectrum [(Mpc/h)^3].
    camb_results : camb.CAMBdata
        Full CAMB results object.
    """
    if params is None:
        params = PLANCK18

    cp = camb.CAMBparams()
    cp.set_cosmology(
        H0=params["H0"],
        ombh2=params["ombh2"],
        omch2=params["omch2"],
        tau=params["tau"],
        mnu=params["mnu"],
        nnu=params["nnu"],
    )
    cp.InitPower.set_params(
        ns=params["ns"],
        As=params["As"],
    )
    cp.set_matter_power(
        redshifts=[z],
        kmax=kmax,
        nonlinear=False,
    )
    cp.NonLinear = camb.model.NonLinear_none

    results = camb.get_results(cp)
    k_arr, _z_arr, pk_arr = results.get_matter_power_spectrum(
        minkh=1e-4,
        maxkh=kmax,
        npoints=npoints,
        var1="delta_tot",
        var2="delta_tot",
    )
    # pk_arr shape is (n_z, n_k); take z=0 slice
    return k_arr, pk_arr[0], results


def get_cosmology_dict(params=None):
    """Return a dict of derived cosmological quantities needed by the halo model."""
    if params is None:
        params = PLANCK18

    h = params["H0"] / 100.0
    Omega_b = params["ombh2"] / h**2
    Omega_c = params["omch2"] / h**2
    Omega_m = Omega_b + Omega_c   # matter density parameter (no nu for simplicity)

    # Critical density today in (M_sun/h) / (Mpc/h)^3
    # rho_crit = 3 H0^2 / (8 pi G)
    # In units where H0 = 100 h km/s/Mpc:
    #   rho_crit,0 = 2.775e11 * h^2  M_sun/Mpc^3
    #              = 2.775e11         M_sun h^2/Mpc^3   [comoving]
    # In h/Mpc units:
    rho_crit0 = 2.775e11  # M_sun/h / (Mpc/h)^3  (h^2 absorbed into h-units)

    return dict(
        H0=params["H0"],
        h=h,
        Omega_b=Omega_b,
        Omega_c=Omega_c,
        Omega_m=Omega_m,
        rho_m0=Omega_m * rho_crit0,  # mean matter density [(M_sun/h)/(Mpc/h)^3]
        rho_crit0=rho_crit0,
        ns=params["ns"],
        As=params["As"],
    )


if __name__ == "__main__":
    print("Computing linear power spectrum with Planck 2018 cosmology...")
    k, pk, results = get_camb_results()

    # Save k [h/Mpc] and P(k) [(Mpc/h)^3] to a text file
    out = np.column_stack([k, pk])
    header = (
        "Linear matter power spectrum at z=0, Planck 2018 cosmology (CAMB)\n"
        "Columns: k [h/Mpc]   P_lin(k) [(Mpc/h)^3]"
    )
    np.savetxt("linear_pk.txt", out, header=header)
    print(f"Saved {len(k)} k-points to linear_pk.txt")
    print(f"  k range: {k[0]:.4e} – {k[-1]:.4e} h/Mpc")
    print(f"  P(k) range: {pk.min():.4e} – {pk.max():.4e} (Mpc/h)^3")

    cosmo = get_cosmology_dict()
    print(f"\nDerived cosmology:")
    print(f"  Omega_m  = {cosmo['Omega_m']:.4f}")
    print(f"  rho_m0   = {cosmo['rho_m0']:.4e}  M_sun h^2 / Mpc^3")
