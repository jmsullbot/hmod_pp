"""
Halo Model Matter Power Spectrum
=================================
Implements the halo model power spectrum using:
  - Tinker et al. 2010 halo mass function (arXiv:1001.3162)
  - Tinker et al. 2010 halo bias (arXiv:1005.2239)
  - NFW density profile (Navarro, Frenk & White 1997)
  - Linear power spectrum from CAMB (saved to file by linear_power_spectrum.py)

Mass definition: M_200m (mass enclosed within r_200m, where mean density is 200
times the mean matter density of the universe).

Units throughout:
  - Mass          : M_sun / h
  - Length/wavenumber : Mpc/h, h/Mpc
  - Power spectrum: (Mpc/h)^3
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import warnings


# ---------------------------------------------------------------------------
# Load linear power spectrum and build spline interpolator
# ---------------------------------------------------------------------------

def load_linear_pk(filename="linear_pk.txt"):
    """
    Load linear P(k) from file and return a log-log spline interpolator.

    The interpolator is defined in log-log space so that it is well-behaved
    over many decades in k.  Calling it returns P_lin(k) in (Mpc/h)^3.

    Parameters
    ----------
    filename : str
        Path to the text file produced by linear_power_spectrum.py.

    Returns
    -------
    pk_spline : callable
        pk_spline(k) -> P_lin(k) [(Mpc/h)^3], for k in h/Mpc.
    k_arr : ndarray
        Raw k values from file.
    pk_arr : ndarray
        Raw P(k) values from file.
    """
    data = np.loadtxt(filename, comments="#")
    k_arr = data[:, 0]
    pk_arr = data[:, 1]

    # Build spline in log-log space for numerical stability
    log_k = np.log(k_arr)
    log_pk = np.log(pk_arr)
    # ext=0: extrapolate outside domain (needed for sigma^2 integration)
    _spline = InterpolatedUnivariateSpline(log_k, log_pk, k=3, ext=0)

    _log_k_min = log_k[0]
    _log_k_max = log_k[-1]

    def pk_spline(k):
        scalar_in = np.isscalar(k)
        k = np.atleast_1d(np.asarray(k, dtype=float))
        lk = np.clip(np.log(k), _log_k_min, _log_k_max)
        result = np.exp(_spline(lk))
        if scalar_in:
            return float(result[0])
        return result

    return pk_spline, k_arr, pk_arr


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------

class Cosmology:
    """
    Simple flat LCDM cosmology container with methods needed by the halo model.

    Parameters
    ----------
    Omega_m : float
        Matter density parameter today.
    h : float
        Dimensionless Hubble parameter (H0 = 100 h km/s/Mpc).
    """

    def __init__(self, Omega_m=0.3138, h=0.6736):
        self.Omega_m = Omega_m
        self.Omega_L = 1.0 - Omega_m   # flat universe
        self.h = h
        # Mean matter density today [(M_sun/h) / (Mpc/h)^3]
        # rho_crit,0 = 2.775e11 h^2 M_sun/Mpc^3 = 2.775e11 M_sun h^2/Mpc^3
        self.rho_crit0 = 2.775e11  # M_sun/h / (Mpc/h)^3
        self.rho_m0 = self.Omega_m * self.rho_crit0

    def E(self, z):
        """Dimensionless Hubble factor E(z) = H(z)/H0."""
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_L)

    def rho_m(self, z):
        """Mean matter density at redshift z [(M_sun/h) / (Mpc/h)^3]."""
        return self.rho_m0 * (1 + z)**3

    def rho_crit(self, z):
        """Critical density at z [(M_sun/h) / (Mpc/h)^3]."""
        return self.rho_crit0 * self.E(z)**2

    def comoving_distance(self, z, n=500):
        """
        Comoving distance to redshift z [Mpc/h].
        Integrated numerically using Simpson's rule.
        """
        if np.isscalar(z):
            z_arr = np.linspace(0, z, n)
            integrand = 1.0 / self.E(z_arr)
            chi = np.trapezoid(integrand, z_arr) * (2997.92458)  # c/H0 in Mpc/h
            return chi
        else:
            chi = np.array([self.comoving_distance(zi, n=n) for zi in z])
            return chi

    def dchi_dz(self, z):
        """d chi / dz = c / H(z) in (Mpc/h) / (dimensionless)."""
        return 2997.92458 / self.E(z)  # c/H0 / E(z)  [Mpc/h]

    def growth_factor(self, z, n=200):
        """
        Linear growth factor D(z) normalised to D(0) = 1.
        Uses the integral form valid for flat LCDM.
        """
        def integrand(zp):
            return (1 + zp) / self.E(zp)**3

        D0_inv = quad(integrand, 0, 1000)[0]
        Dz_inv = quad(integrand, z, 1000)[0]
        return self.E(z) * Dz_inv / D0_inv

    def Omega_m_z(self, z):
        """Omega_m(z) = rho_m(z) / rho_crit(z)."""
        return self.Omega_m * (1 + z)**3 / self.E(z)**2


# ---------------------------------------------------------------------------
# Variance of the linear power spectrum: sigma^2(R)
# ---------------------------------------------------------------------------

def sigma_sq(R, pk_spline, k_lo=1e-4, k_hi=1e2, n=2000):
    """
    Compute the variance of the linear density field smoothed with a top-hat
    filter of radius R [Mpc/h]:

        sigma^2(R) = int_0^inf (k^2/2pi^2) P_lin(k) W^2(kR) dk

    Uses log-spaced k sampling and Simpson's rule.

    Parameters
    ----------
    R : float or array
        Smoothing scale(s) [Mpc/h].
    pk_spline : callable
        Linear power spectrum spline.
    k_lo, k_hi : float
        Integration limits.
    n : int
        Number of integration points.

    Returns
    -------
    sig2 : float or ndarray
        sigma^2(R).
    """
    k = np.geomspace(k_lo, k_hi, n)
    pk = pk_spline(k)

    def _sigma_sq_single(R_val):
        x = k * R_val
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
        integrand = k**2 / (2 * np.pi**2) * pk * W**2
        return np.trapezoid(integrand, k)

    if np.isscalar(R):
        return _sigma_sq_single(R)
    else:
        return np.array([_sigma_sq_single(Ri) for Ri in R])


def mass_to_radius(M, rho_m):
    """
    Lagrangian radius R corresponding to mass M (top-hat filter):
        M = (4/3) pi R^3 rho_m

    Parameters
    ----------
    M : float or array  [M_sun/h]
    rho_m : float       [(M_sun/h)/(Mpc/h)^3]

    Returns
    -------
    R : float or array  [Mpc/h]
    """
    return (3.0 * M / (4.0 * np.pi * rho_m))**(1.0 / 3.0)


def radius_to_mass(R, rho_m):
    """Inverse of mass_to_radius."""
    return (4.0 / 3.0) * np.pi * R**3 * rho_m


# ---------------------------------------------------------------------------
# Tinker et al. 2010 Halo Mass Function  (arXiv:1001.3162)
# ---------------------------------------------------------------------------
# Table 4 of Tinker 2010 gives parameters for Delta = 200 w.r.t. *mean* density.
# We use Delta=200 (mean density), i.e. M_200m.

_TINKER10_HMF_PARAMS = {
    # Delta_m -> (alpha, beta, gamma, phi, eta)
    # from Table 4 of Tinker et al. 2010
    200:  (0.368, 0.589, 0.864, -0.729, -0.243),
    300:  (0.363, 0.585, 0.922, -0.789, -0.261),
    400:  (0.385, 0.544, 0.987, -0.910, -0.261),
    600:  (0.389, 0.543, 1.09,  -1.05,  -0.273),
    800:  (0.393, 0.564, 1.20,  -1.20,  -0.278),
    1200: (0.365, 0.623, 1.34,  -1.26,  -0.301),
    1600: (0.379, 0.637, 1.50,  -1.45,  -0.301),
    2400: (0.355, 0.673, 1.68,  -1.50,  -0.319),
    3200: (0.327, 0.702, 1.81,  -1.49,  -0.336),
}

# Redshift evolution of beta, gamma, phi, eta (Eq 9 of Tinker 2010)
def _tinker10_hmf_params_z(Delta, z):
    """Return Tinker 2010 HMF parameters at redshift z for overdensity Delta."""
    alpha0, beta0, gamma0, phi0, eta0 = _TINKER10_HMF_PARAMS[Delta]
    # Eq. 9
    beta  = beta0  * (1 + z)**0.20
    phi   = phi0   * (1 + z)**(-0.08)
    eta   = eta0   * (1 + z)**0.27
    gamma = gamma0 * (1 + z)**(-0.01)
    alpha = alpha0  # alpha does not evolve
    return alpha, beta, gamma, phi, eta


def tinker10_f_sigma(sigma, z=0.0, Delta=200):
    """
    Tinker et al. 2010 multiplicity function f(sigma) for M_200m.

    Eq. 8 of Tinker et al. 2010:
        f(sigma) = alpha * [ 1 + (beta/sigma)^(-2*phi) ] * exp(-gamma/(sigma^2))
                   * sigma^(2*eta)

    Parameters
    ----------
    sigma : float or array
        RMS linear fluctuation (sqrt of sigma^2(M)).
    z : float
        Redshift.
    Delta : int
        Overdensity w.r.t. mean density (200 by default).

    Returns
    -------
    f : float or array  (dimensionless)
    """
    alpha, beta, gamma, phi, eta = _tinker10_hmf_params_z(Delta, z)
    return (alpha * (1 + (beta / sigma)**(-2 * phi))
            * np.exp(-gamma / sigma**2)
            * sigma**(2 * eta))


def tinker10_dndM(M, pk_spline, cosmo, z=0.0, Delta=200):
    """
    Tinker 2010 halo mass function dn/dM at mass M [M_sun/h].

        dn/dM = (rho_m / M) * f(sigma) * |d ln sigma^{-1} / dM|

    Parameters
    ----------
    M : float or array  [M_sun/h]
    pk_spline : callable
    cosmo : Cosmology
    z : float
    Delta : int

    Returns
    -------
    dndM : float or array  [(Mpc/h)^{-3} (M_sun/h)^{-1}]
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    # Tinker 2010 uses the comoving mean matter density (rho_m0 = const) for
    # both the Lagrangian radius and the dn/dM prefactor.  Using rho_m(z) here
    # would introduce a spurious (1+z)^3 factor that makes dn/dM *increase*
    # with z at fixed M, which is unphysical.
    rho_m0 = cosmo.rho_m0
    Dz = cosmo.growth_factor(z)

    R = mass_to_radius(M, rho_m0)
    # sigma^2 uses the z=0 linear P(k); growth factor scales it
    sig2_R = sigma_sq(R, pk_spline) * Dz**2
    sigma_R = np.sqrt(sig2_R)

    f = tinker10_f_sigma(sigma_R, z=z, Delta=Delta)

    # Numerical derivative d ln(sigma^{-1}) / dM = -1/(2 sigma^2) * d(sigma^2)/dM
    # Use finite difference
    dM = M * 1e-4
    sig2_plus  = sigma_sq(mass_to_radius(M + dM, rho_m0), pk_spline) * Dz**2
    sig2_minus = sigma_sq(mass_to_radius(M - dM, rho_m0), pk_spline) * Dz**2
    dsig2_dM = (sig2_plus - sig2_minus) / (2 * dM)

    dlninvsigma_dM = -dsig2_dM / (2 * sig2_R)

    dndM = (rho_m0 / M) * f * np.abs(dlninvsigma_dM)
    return dndM.squeeze() if dndM.size == 1 else dndM


# ---------------------------------------------------------------------------
# Tinker et al. 2010 Halo Bias  (arXiv:1005.2239)
# ---------------------------------------------------------------------------
# Table 2 of Tinker et al. 2010 bias paper, Delta=200 (w.r.t. mean density)

_TINKER10_BIAS_PARAMS = {
    # Delta_m -> (A, a, B, b, C, c)
    200:  (1.340, 0.241, 0.183, 1.500, 0.265, 2.400),
    300:  (1.315, 0.240, 0.183, 1.466, 0.261, 2.368),
    400:  (1.306, 0.240, 0.183, 1.441, 0.259, 2.346),
    600:  (1.293, 0.240, 0.183, 1.405, 0.256, 2.312),
    800:  (1.285, 0.240, 0.183, 1.381, 0.254, 2.290),
    1200: (1.273, 0.240, 0.183, 1.342, 0.251, 2.253),
    1600: (1.266, 0.240, 0.183, 1.317, 0.249, 2.228),
    2400: (1.255, 0.240, 0.183, 1.280, 0.246, 2.195),
    3200: (1.248, 0.240, 0.183, 1.252, 0.244, 2.168),
}

def tinker10_bias(M, pk_spline, cosmo, z=0.0, Delta=200):
    """
    Tinker et al. 2010 halo bias b(M).

    Eq. 6 of Tinker et al. 2010 (bias paper):
        b(nu) = 1 - A*nu^a/(nu^a + delta_c^a) + B*nu^b + C*nu^c
    where nu = delta_c / sigma(M).

    Parameters
    ----------
    M : float or array  [M_sun/h]
    pk_spline : callable
    cosmo : Cosmology
    z : float
    Delta : int

    Returns
    -------
    b : float or array  (dimensionless)
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    delta_c = 1.686  # critical overdensity for collapse (EdS approximation)

    Dz = cosmo.growth_factor(z)
    R = mass_to_radius(M, cosmo.rho_m0)   # comoving Lagrangian radius
    sigma_R = np.sqrt(sigma_sq(R, pk_spline) * Dz**2)
    nu = delta_c / sigma_R

    A, a, B, b, C, c = _TINKER10_BIAS_PARAMS[Delta]
    bias = 1.0 - A * nu**a / (nu**a + delta_c**a) + B * nu**b + C * nu**c
    return bias.squeeze() if bias.size == 1 else bias


# ---------------------------------------------------------------------------
# NFW profile
# ---------------------------------------------------------------------------

def nfw_concentration(M, z=0.0, cosmo=None):
    """
    Concentration parameter c(M,z) for M_200m using the Duffy et al. 2008
    relation (full sample, Table 1):
        c = A * (M / M_pivot)^B * (1+z)^C
    with A=10.14, B=-0.081, C=-1.01, M_pivot=2e12 M_sun/h.

    Parameters
    ----------
    M : float or array  [M_sun/h]
    z : float

    Returns
    -------
    c : float or array
    """
    A, B, C = 10.14, -0.081, -1.01
    M_pivot = 2.0e12  # M_sun / h
    return A * (M / M_pivot)**B * (1 + z)**C


def nfw_radius(M, Delta, rho_ref):
    """
    Halo radius r_Delta such that M = (4/3) pi Delta rho_ref r_Delta^3.

    Parameters
    ----------
    M : float or array  [M_sun/h]
    Delta : float
        Overdensity (e.g. 200).
    rho_ref : float
        Reference density [(M_sun/h)/(Mpc/h)^3].

    Returns
    -------
    r : float or array  [Mpc/h]
    """
    return (3.0 * M / (4.0 * np.pi * Delta * rho_ref))**(1.0 / 3.0)


def nfw_fourier(k, M, z=0.0, cosmo=None, Delta=200):
    """
    Normalised Fourier transform of the NFW density profile u(k|M),
    defined such that u(k->0|M) = 1.

    This is the NFW profile truncated at r_200m, normalised by the halo mass:
        u(k|M) = int_0^{r200} rho_NFW(r) e^{ikr} 4pi r^2 dr / M

    Analytic form (see e.g. Cooray & Sheth 2002, Eq. 11):

        u(k|M) = [sin(k rs)(Si((1+c) k rs) - Si(k rs))
                  - cos(k rs)(Ci((1+c) k rs) - Ci(k rs))
                  - sin(c k rs)/((1+c) k rs)] / [ln(1+c) - c/(1+c)]

    Parameters
    ----------
    k : float or array  [h/Mpc]
    M : float           [M_sun/h]
    z : float
    cosmo : Cosmology
    Delta : int

    Returns
    -------
    u : float or array  (dimensionless, normalised to 1 at k=0)
    """
    from scipy.special import sici

    if cosmo is None:
        cosmo = Cosmology()

    c = nfw_concentration(M, z=z)
    rho_ref = cosmo.rho_m(z)  # mean matter density for M_200m
    r200 = nfw_radius(M, Delta, rho_ref)
    rs = r200 / c  # scale radius

    k = np.atleast_1d(np.asarray(k, dtype=float))
    x = k * rs         # dimensionless
    xc = x * (1 + c)  # k * r200

    Si_xc, Ci_xc = sici(xc)
    Si_x,  Ci_x  = sici(x)

    norm = np.log(1 + c) - c / (1 + c)

    # Correct analytic NFW Fourier transform (Cooray & Sheth 2002, Eq. 11):
    # u = [cos(x)(Ci(xc)-Ci(x)) + sin(x)(Si(xc)-Si(x)) - sin(cx)/xc] / norm
    u = (np.cos(x) * (Ci_xc - Ci_x)
         + np.sin(x) * (Si_xc - Si_x)
         - np.sin(c * x) / xc) / norm

    return u.squeeze() if u.size == 1 else u


def nfw_profile_check(M, z=0.0, cosmo=None, Delta=200, n=2000):
    """
    Verify that the NFW profile is normalised: int rho(r) 4pi r^2 dr = M.

    Returns the fractional deviation (result/M - 1).
    """
    if cosmo is None:
        cosmo = Cosmology()

    c = nfw_concentration(M, z=z)
    rho_ref = cosmo.rho_m(z)
    r200 = nfw_radius(M, Delta, rho_ref)
    rs = r200 / c

    # NFW characteristic density rho_s such that M = integral
    norm = np.log(1 + c) - c / (1 + c)
    rho_s = M / (4 * np.pi * rs**3 * norm)

    r = np.geomspace(1e-4 * rs, r200, n)
    rho = rho_s / ((r / rs) * (1 + r / rs)**2)
    integral = np.trapezoid(4 * np.pi * r**2 * rho, r)

    return (integral / M) - 1.0


# ---------------------------------------------------------------------------
# Halo Model Power Spectrum
# ---------------------------------------------------------------------------

class HaloModel:
    """
    Halo model matter power spectrum.

    Uses:
      - Tinker 2010 mass function and bias (Delta=200, mean density)
      - NFW profiles (Duffy 2008 concentration)
      - Linear power spectrum from file

    Parameters
    ----------
    pk_spline : callable
        Linear P(k) spline (output of load_linear_pk).
    cosmo : Cosmology
    z : float
        Redshift.
    M_lo, M_hi : float
        Integration limits in halo mass [M_sun/h].
    n_M : int
        Number of mass integration points (log-spaced).
    Delta : int
        Overdensity definition (200 = M_200m).
    """

    def __init__(self, pk_spline, cosmo=None, z=0.0,
                 M_lo=1e10, M_hi=1e16, n_M=100, Delta=200):
        self.pk_spline = pk_spline
        self.cosmo = cosmo if cosmo is not None else Cosmology()
        self.z = z
        self.M_lo = M_lo
        self.M_hi = M_hi
        self.n_M = n_M
        self.Delta = Delta

        self._precompute()

    def _precompute(self):
        """Pre-compute mass-dependent quantities on a grid."""
        z = self.z
        cosmo = self.cosmo
        pk_spline = self.pk_spline

        self.M_arr = np.geomspace(self.M_lo, self.M_hi, self.n_M)
        rho_m = cosmo.rho_m(z)

        # Mass function
        self.dndM_arr = tinker10_dndM(
            self.M_arr, pk_spline, cosmo, z=z, Delta=self.Delta
        )

        # Raw Tinker bias
        bias_raw = tinker10_bias(
            self.M_arr, pk_spline, cosmo, z=z, Delta=self.Delta
        )

        # Normalize bias so that int dn/dM * b_norm * M/rho_m dM = 1.
        # This ensures the 2-halo term -> P_lin at k -> 0, consistent with
        # the assumption that all matter traces the large-scale structure.
        # The Tinker HMF is not mass-conserving (integrates to ~0.73 of rho_m),
        # so without this normalization P_2h(k->0) ≠ P_lin.
        # Reference: Murray et al. 2013 (halomod); Cooray & Sheth 2002, Sect. 3.
        I_bias_0 = np.trapezoid(self.dndM_arr * bias_raw * self.M_arr / rho_m,
                                self.M_arr)
        self.bias_arr = bias_raw / I_bias_0

    def two_halo(self, k):
        """
        Two-halo term of the matter power spectrum:

            P_2h(k) = [ int dM (dn/dM) b(M) u(k|M) (M/rho_m) ]^2 P_lin(k)

        The integral is evaluated by Simpson's rule over the precomputed mass grid.

        Parameters
        ----------
        k : float or array  [h/Mpc]

        Returns
        -------
        P_2h : float or array  [(Mpc/h)^3]
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        rho_m = self.cosmo.rho_m(self.z)
        M_arr = self.M_arr
        dndM = self.dndM_arr
        bias = self.bias_arr

        P2h = np.empty(len(k))
        for i, ki in enumerate(k):
            u = nfw_fourier(ki, M_arr, z=self.z, cosmo=self.cosmo,
                            Delta=self.Delta)
            integrand = dndM * bias * u * (M_arr / rho_m)
            I = np.trapezoid(integrand, M_arr)
            P2h[i] = I**2 * self.pk_spline(float(ki))

        return P2h.squeeze() if P2h.size == 1 else P2h

    def one_halo(self, k):
        """
        One-halo term of the matter power spectrum:

            P_1h(k) = int dM (dn/dM) (M/rho_m)^2 |u(k|M)|^2

        Parameters
        ----------
        k : float or array  [h/Mpc]

        Returns
        -------
        P_1h : float or array  [(Mpc/h)^3]
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        rho_m = self.cosmo.rho_m(self.z)
        M_arr = self.M_arr
        dndM = self.dndM_arr

        P1h = np.empty(len(k))
        for i, ki in enumerate(k):
            u = nfw_fourier(ki, M_arr, z=self.z, cosmo=self.cosmo,
                            Delta=self.Delta)
            integrand = dndM * (M_arr / rho_m)**2 * u**2
            P1h[i] = np.trapezoid(integrand, M_arr)

        return P1h.squeeze() if P1h.size == 1 else P1h

    def total(self, k):
        """
        Total halo model power spectrum P(k) = P_1h(k) + P_2h(k).

        Parameters
        ----------
        k : float or array  [h/Mpc]

        Returns
        -------
        P_tot : float or array  [(Mpc/h)^3]
        """
        return self.one_halo(k) + self.two_halo(k)

    def build_pk_spline(self, k_arr):
        """
        Evaluate the total halo model P(k) on k_arr and return a spline.
        Useful for the Limber integral.

        Parameters
        ----------
        k_arr : array  [h/Mpc]

        Returns
        -------
        spline : InterpolatedUnivariateSpline
        k_arr : array
        pk_arr : array
        """
        pk_arr = self.total(k_arr)
        log_k = np.log(k_arr)
        log_pk = np.log(pk_arr)
        _sp = InterpolatedUnivariateSpline(log_k, log_pk, k=3, ext=2)

        def spline(k):
            k = np.atleast_1d(np.asarray(k, dtype=float))
            return np.exp(_sp(np.log(k)))

        return spline, k_arr, pk_arr


def gaussian_dndM_extra(M, z, A=1.3e-5, logM0=11.0, sigma_logM=1.0,
                        z0=8.2, sigma_z=0.43):
    """
    Extra HMF component: bivariate Gaussian in (log10 M, z).

        dn_extra/dM(M, z) = A * exp[-0.5*((log10M - logM0)/sigma_logM)^2]
                              * exp[-0.5*((z - z0)/sigma_z)^2]
                              / M

    A [(Mpc/h)^{-3}] is the amplitude of the number-density Gaussian
    before the per-mass conversion; dividing by M [(M_sun/h)] gives the
    standard HMF units [(Mpc/h)^{-3} (M_sun/h)^{-1}].

    Parameters
    ----------
    M : array  [M_sun/h]
    z : float
    A : float  [(Mpc/h)^{-3}]
    logM0, sigma_logM, z0, sigma_z : Gaussian shape parameters

    Returns
    -------
    dn_extra/dM : array  [(Mpc/h)^{-3} (M_sun/h)^{-1}]
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    logM = np.log10(M)
    gauss_M = np.exp(-0.5 * ((logM - logM0) / sigma_logM)**2)
    gauss_z = np.exp(-0.5 * ((z - z0) / sigma_z)**2)
    return A * gauss_M * gauss_z / M
