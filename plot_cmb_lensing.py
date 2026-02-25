"""
CMB Lensing Power Spectrum: Halo Model vs Linear Theory vs Planck 2018 Data
============================================================================
This script:
  1. Loads the linear P(k) and builds the halo model
  2. Computes C_L^{kappakappa} from both the halo model and linear theory
     via the Limber approximation
  3. Also computes the exact C_L^{kappakappa} from CAMB (Boltzmann code)
  4. Loads Planck 2018 lensing bandpowers (Planck 2018 VIII, arXiv:1807.06210)
     from CosmoMC data files and converts phi -> kappa
  5. Makes a comparison plot

Data files required:
  - planck2018_lensing_aggressive_bandpowers.dat  (19 bins, L=8-2048)
  - planck2018_lensing_aggressive_cov.dat         (19x19 covariance matrix)
  - linear_pk.txt                                  (from linear_power_spectrum.py)

Run after:
  python linear_power_spectrum.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import camb

from linear_power_spectrum import get_cosmology_dict, PLANCK18, get_camb_results
from halo_model import load_linear_pk, Cosmology, HaloModel, gaussian_dndM_extra, tinker10_dndM
from cmb_lensing import cl_kappakappa, make_delta_pk_callable

# -------------------------------------------------------------------------
# 1. Setup cosmology and build halo model
# -------------------------------------------------------------------------
print("Loading linear power spectrum and building halo model...")
pk_spline, k_arr, pk_arr = load_linear_pk("linear_pk.txt")
cosmo = Cosmology(Omega_m=0.3138, h=0.6736)

# Build the halo model at z=0
# n_M=120 for good mass integration accuracy
hm = HaloModel(pk_spline, cosmo=cosmo, z=0.0, n_M=120)
print(f"  Halo model built: M=[{hm.M_lo:.0e}, {hm.M_hi:.0e}] M_sun/h, n_M={hm.n_M}")

# Build halo model P(k) spline for Limber integral
k_hm = np.geomspace(1e-4, 49.0, 400)
print("  Computing halo model P(k)...")
pk_hm_arr = hm.total(k_hm)
pk1h_arr = hm.one_halo(k_hm)
pk2h_arr = hm.two_halo(k_hm)

# Build spline for the halo model P(k)
from scipy.interpolate import InterpolatedUnivariateSpline
_log_spline = InterpolatedUnivariateSpline(np.log(k_hm), np.log(pk_hm_arr), k=3, ext=0)
def pk_hm_spline(k):
    k = np.atleast_1d(np.asarray(k, dtype=float))
    lk = np.clip(np.log(k), np.log(k_hm[0]), np.log(k_hm[-1]))
    return np.exp(_log_spline(lk))

# -------------------------------------------------------------------------
# 2. Compute C_L^{kappakappa} via Limber approximation
# -------------------------------------------------------------------------
print("Computing Limber integrals...")
ell_arr = np.unique(np.concatenate([
    np.geomspace(8, 100, 30).astype(int),
    np.geomspace(100, 2000, 40).astype(int),
]))

# Halo model C_L
CL_hm = cl_kappakappa(ell_arr, pk_hm_spline, cosmo, z_max=100.0, n_chi=500)
print(f"  Halo model C_L range: {CL_hm.min():.3e} – {CL_hm.max():.3e}")

# Linear theory C_L
CL_lin = cl_kappakappa(ell_arr, pk_spline, cosmo, z_max=100.0, n_chi=500)
print(f"  Linear C_L range: {CL_lin.min():.3e} – {CL_lin.max():.3e}")

# -------------------------------------------------------------------------
# 2b. Modified HMF: Tinker + bivariate Gaussian in (log10 M, z)
# -------------------------------------------------------------------------
# Gaussian parameters centred on the high-z, low-mass population
GAUSS_PARAMS = dict(
    A         = 0.3,    # peak amplitude [(Mpc/h)^{-3}] per unit log10M per unit z
    logM0     = 11.0,   # central log10(M_h / (M_sun/h))
    sigma_logM= 1.0,    # width in log10(M) [dex]
    z0        = 8.2,    # central redshift
    sigma_z   = 0.43,   # width in z
)
print("Precomputing extra 1-halo power from Gaussian HMF modification...")
print(f"  Gaussian params: {GAUSS_PARAMS}")

def dndM_extra_fn(M_arr, z):
    return gaussian_dndM_extra(M_arr, z, **GAUSS_PARAMS)

# Custom z grid: coarse away from the Gaussian, dense around its peak.
# Dense patch around z0=8.2 with dz~0.03 (~14 steps per sigma_z=0.43).
# Total z range covers 0.02 to 100.
# M_lo=1e8 to cover 3 dex below logM0=11.
_z_dpk = np.unique(np.concatenate([
    np.linspace(0.02, 6.0,  30),    # coarse below peak
    np.linspace(6.0,  11.0, 170),   # dense around Gaussian (dz~0.03)
    np.linspace(11.0, 100.0, 30),   # coarse above peak
]))
delta_pk_fn, delta_pk_grid, k_dpk, z_dpk = make_delta_pk_callable(
    dndM_extra_fn, cosmo,
    k_lo=5e-3, k_hi=50.0, n_k=60,
    M_lo=1e8, M_hi=1e16, n_M=60,
    z_arr=_z_dpk,
)
print(f"  ΔP grid peak: {delta_pk_grid.max():.3e} (Mpc/h)^3 "
      f"at k={k_dpk[delta_pk_grid.max(axis=1).argmax()]:.2f} h/Mpc")

CL_hm_mod = cl_kappakappa(ell_arr, pk_hm_spline, cosmo, z_max=100.0, n_chi=500,
                           delta_pk_fn=delta_pk_fn)
print(f"  Modified halo model C_L range: {CL_hm_mod.min():.3e} – {CL_hm_mod.max():.3e}")

# -------------------------------------------------------------------------
# 3. Compute exact C_L^{kappakappa} from CAMB
# -------------------------------------------------------------------------
print("Computing CAMB lensing power spectrum...")
cp = camb.CAMBparams()
cp.set_cosmology(
    H0=PLANCK18["H0"],
    ombh2=PLANCK18["ombh2"],
    omch2=PLANCK18["omch2"],
    tau=PLANCK18["tau"],
    mnu=PLANCK18["mnu"],
    nnu=PLANCK18["nnu"],
)
cp.InitPower.set_params(ns=PLANCK18["ns"], As=PLANCK18["As"])
cp.set_for_lmax(2048, lens_potential_accuracy=2)

camb_results = camb.get_results(cp)
lens_cls = camb_results.get_lens_potential_cls(lmax=2048)
# lens_cls columns: [phi-phi, T-phi, E-phi]  in units of C_L * L^2(L+1)^2/(2pi)
# The first column is [L(L+1)]^2/(2pi) * C_L^{phiphi}
ell_camb = np.arange(lens_cls.shape[0])
# Convert to C_L^{kappakappa}: C_L^kk = [L(L+1)/2]^2 * C_L^{phiphi}
# CAMB gives: col[0] = [L(L+1)]^2/(2pi) * C_L^{pp}
# so C_L^{pp} = col[0] * 2pi / [L(L+1)]^2
# C_L^{kk} = [L(L+1)/2]^2 * C_L^{pp} = [L(L+1)]^2/4 * C_L^{pp}
#           = col[0] * 2pi / 4 = col[0] * pi/2
with np.errstate(divide='ignore', invalid='ignore'):
    CL_camb_kk = np.where(ell_camb > 1,
                          lens_cls[:, 0] * np.pi / 2.0,
                          0.0)
ell_camb_plot = ell_camb[2:]
CL_camb_kk_plot = CL_camb_kk[2:]
print(f"  CAMB C_L^kk at L=100: {CL_camb_kk[100]:.3e}")

# -------------------------------------------------------------------------
# 3b. CAMB HaloFit P(k) at z=0
# -------------------------------------------------------------------------
print("Computing CAMB HaloFit P(k)...")
cp_hf = camb.CAMBparams()
cp_hf.set_cosmology(H0=PLANCK18["H0"], ombh2=PLANCK18["ombh2"],
                    omch2=PLANCK18["omch2"], tau=PLANCK18["tau"],
                    mnu=PLANCK18["mnu"], nnu=PLANCK18["nnu"])
cp_hf.InitPower.set_params(ns=PLANCK18["ns"], As=PLANCK18["As"])
cp_hf.set_matter_power(redshifts=[0.0], kmax=50.0, nonlinear=True)
camb_hf = camb.get_results(cp_hf)
kh_hf, _, pk_hf_2d = camb_hf.get_matter_power_spectrum(
    minkh=1e-4, maxkh=50.0, npoints=400)
pk_halofit_z0 = pk_hf_2d[0]   # shape (400,), z=0, units (Mpc/h)^3
print(f"  HaloFit P(k) at k=1 h/Mpc: {np.interp(1.0, kh_hf, pk_halofit_z0):.3e} (Mpc/h)^3")

# -------------------------------------------------------------------------
# 3c. P(k) at z=8.2 – base halo model and modified halo model
# -------------------------------------------------------------------------
z_peak = GAUSS_PARAMS['z0']
Dz_peak = cosmo.growth_factor(z_peak)
D0      = cosmo.growth_factor(0.0)
growth_sq_82 = (Dz_peak / D0)**2

pk_lin_z82     = pk_spline(k_hm) * growth_sq_82
pk_hm_z82      = pk_hm_arr * growth_sq_82
delta_pk_z82   = delta_pk_fn(k_hm, np.full(len(k_hm), z_peak))
pk_hm_mod_z82  = pk_hm_z82 + delta_pk_z82
print(f"  ΔP(k,z=8.2) peak: {delta_pk_z82.max():.3e} (Mpc/h)^3 "
      f"at k={k_hm[delta_pk_z82.argmax()]:.3f} h/Mpc")

# -------------------------------------------------------------------------
# 3d. HMF as a function of M at z=8.2 and as a function of z at M=1e11
# -------------------------------------------------------------------------
print("Computing HMF curves...")
M_hmf = np.geomspace(1e8, 1e16, 120)
dndM_tinker_z82 = tinker10_dndM(M_hmf, pk_spline, cosmo, z=z_peak)
dndM_gauss_z82  = gaussian_dndM_extra(M_hmf, z_peak, **GAUSS_PARAMS)
dndM_mod_z82    = dndM_tinker_z82 + dndM_gauss_z82

# HMF vs z at M=1e11 M_sun/h
M_fixed  = 1e11
z_hmf    = np.linspace(0.02, 15.0, 80)
dndM_tinker_vz = np.array([
    float(tinker10_dndM(np.array([M_fixed]), pk_spline, cosmo, z=z))
    for z in z_hmf])
# Gaussian factor: vectorise analytically (z is the varying axis)
_lM = np.log10(M_fixed)
_gauss_M_fac = np.exp(-0.5 * ((_lM - GAUSS_PARAMS['logM0'])
                               / GAUSS_PARAMS['sigma_logM'])**2)
dndM_gauss_vz = (GAUSS_PARAMS['A'] * _gauss_M_fac
                 * np.exp(-0.5 * ((z_hmf - GAUSS_PARAMS['z0'])
                                  / GAUSS_PARAMS['sigma_z'])**2)
                 / (M_fixed * np.log(10.0)))
dndM_mod_vz = dndM_tinker_vz + dndM_gauss_vz

# -------------------------------------------------------------------------
# 4. Load Planck 2018 lensing bandpowers
# -------------------------------------------------------------------------
print("Loading Planck 2018 lensing bandpowers...")
try:
    bp_data = np.loadtxt("planck2018_lensing_aggressive_bandpowers.dat", comments="#")
    # Columns: bin, L_min, L_max, L_av, PP (C_L^kappakappa), Error, Ahat
    # The PP column already stores C_L^{kappakappa} directly.
    L_eff = bp_data[:, 3]
    C_kk_planck   = bp_data[:, 4]   # C_L^{kappakappa}  [dimensionless]
    sigma_kk_planck = bp_data[:, 5] # 1-sigma error on C_L^{kappakappa}

    # Load covariance matrix (already in C_L^{kk} units)
    cov_raw = np.loadtxt("planck2018_lensing_aggressive_cov.dat")
    n_bins = len(L_eff)
    cov_kk = cov_raw.reshape(n_bins, n_bins)
    print(f"  Loaded {n_bins} Planck bandpowers, L=[{L_eff[0]:.0f}, {L_eff[-1]:.0f}]")
    planck_ok = True
except FileNotFoundError:
    print("  WARNING: Planck bandpower files not found. Run plot script after downloading data.")
    planck_ok = False

# -------------------------------------------------------------------------
# 5. Make the comparison plot
# -------------------------------------------------------------------------
print("Making plot...")
fig = plt.figure(figsize=(10, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax_main = fig.add_subplot(gs[0])
ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

# Standard Planck lensing convention:
#   [L(L+1)]^2/(2pi) * C_L^{phiphi}  =  (2/pi) * C_L^{kappakappa}
# (uses C_L^{kk} = [L(L+1)/2]^2 * C_L^{phiphi})
def scale_cl(ell, cl):
    return (2.0 / np.pi) * cl

ell_plot = ell_arr.astype(float)

# -- Main panel --
# CAMB (exact linear theory)
ax_main.plot(ell_camb_plot,
             scale_cl(ell_camb_plot, CL_camb_kk_plot) * 1e7,
             color='k', lw=2, label='CAMB (exact linear theory)', zorder=5)

# Linear theory (Limber)
ax_main.plot(ell_plot,
             scale_cl(ell_plot, CL_lin) * 1e7,
             color='steelblue', lw=2, ls='--', label='Linear theory (Limber)')

# Halo model total
ax_main.plot(ell_plot,
             scale_cl(ell_plot, CL_hm) * 1e7,
             color='firebrick', lw=2.5, label='Halo model (1h+2h, Limber)')

# Modified halo model (Tinker + Gaussian HMF perturbation)
ax_main.plot(ell_plot,
             scale_cl(ell_plot, CL_hm_mod) * 1e7,
             color='darkorchid', lw=2.5, ls='-.', label='Halo model + Gaussian HMF mod')

# Planck data
if planck_ok:
    # Mask negative C_L values (noise dominated at high L)
    mask_pos = C_kk_planck > 0
    ax_main.errorbar(
        L_eff[mask_pos],
        scale_cl(L_eff[mask_pos], C_kk_planck[mask_pos]) * 1e7,
        yerr=scale_cl(L_eff[mask_pos], sigma_kk_planck[mask_pos]) * 1e7,
        fmt='o', color='darkorange', ms=6, lw=1.5, capsize=3,
        label='Planck 2018 MV lensing (arXiv:1807.06210)',
        zorder=10
    )
    # Show negative bandpowers as upper limits
    mask_neg = C_kk_planck < 0
    if mask_neg.sum() > 0:
        ax_main.errorbar(
            L_eff[mask_neg],
            scale_cl(L_eff[mask_neg], np.abs(C_kk_planck[mask_neg])) * 1e7,
            fmt='v', color='darkorange', ms=6, alpha=0.5, zorder=10
        )

ax_main.set_xscale('log')
ax_main.set_yscale('log')
ax_main.set_xlim(8, 2100)
ax_main.set_ylim(1e-3, 1e1)
ax_main.set_ylabel(
    r'$\frac{[L(L+1)]^2}{2\pi}\,C_L^{\phi\phi}\;\times\;10^7$'
    '\n'
    r'$= \frac{2}{\pi}\,C_L^{\kappa\kappa}\;\times\;10^7$',
    fontsize=11
)
ax_main.legend(fontsize=11, loc='lower left')
gauss_label = (fr"Gaussian mod: $A={GAUSS_PARAMS['A']}$, "
               fr"$\log M_0={GAUSS_PARAMS['logM0']}$, "
               fr"$\sigma_{{\log M}}={GAUSS_PARAMS['sigma_logM']}$, "
               fr"$z_0={GAUSS_PARAMS['z0']}$, "
               fr"$\sigma_z={GAUSS_PARAMS['sigma_z']}$")
ax_main.set_title('CMB Lensing Convergence Power Spectrum: Halo Model vs Data\n'
                  + gauss_label, fontsize=11)
ax_main.grid(True, alpha=0.3)
ax_main.tick_params(labelbottom=False)

# -- Ratio panel: halo model / CAMB linear --
# Interpolate CAMB onto our ell grid for comparison
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
camb_spline = IUS(np.log(ell_camb_plot[ell_camb_plot > 2]),
                  np.log(CL_camb_kk_plot[ell_camb_plot > 2]), k=3, ext=3)
CL_camb_interp = np.exp(camb_spline(np.log(ell_plot)))

ratio_hm = CL_hm / CL_camb_interp
ratio_lin_limber = CL_lin / CL_camb_interp
ratio_hm_mod = CL_hm_mod / CL_camb_interp

ax_ratio.axhline(1.0, color='k', lw=1.5, ls='-', label='CAMB exact')
ax_ratio.plot(ell_plot, ratio_lin_limber, color='steelblue', lw=2, ls='--',
              label='Linear Limber / CAMB')
ax_ratio.plot(ell_plot, ratio_hm, color='firebrick', lw=2.5,
              label='Halo model / CAMB linear')
ax_ratio.plot(ell_plot, ratio_hm_mod, color='darkorchid', lw=2.5, ls='-.',
              label='Halo model + Gaussian mod / CAMB')

# Planck data ratio
if planck_ok and mask_pos.sum() > 0:
    CL_camb_at_planck = np.exp(camb_spline(np.log(L_eff[mask_pos])))
    ratio_planck = C_kk_planck[mask_pos] / CL_camb_at_planck
    ratio_planck_err = sigma_kk_planck[mask_pos] / CL_camb_at_planck
    ax_ratio.errorbar(L_eff[mask_pos], ratio_planck, yerr=ratio_planck_err,
                      fmt='o', color='darkorange', ms=5, lw=1.5, capsize=3,
                      label='Planck / CAMB linear')

ax_ratio.axhspan(0.85, 1.15, alpha=0.1, color='gray')
ax_ratio.set_xscale('log')
ax_ratio.set_xlim(8, 2100)
ax_ratio.set_ylim(0.4, 2.0)
ax_ratio.set_xlabel(r'Multipole $L$', fontsize=13)
ax_ratio.set_ylabel(r'Ratio to CAMB', fontsize=12)
ax_ratio.legend(fontsize=9, loc='upper right')
ax_ratio.grid(True, alpha=0.3)
ax_ratio.axhline(1.0, color='k', lw=0.8)

plt.savefig("cmb_lensing_comparison.pdf", dpi=150, bbox_inches='tight')
plt.savefig("cmb_lensing_comparison.png", dpi=150, bbox_inches='tight')
print("Saved cmb_lensing_comparison.pdf and cmb_lensing_comparison.png")

# -------------------------------------------------------------------------
# 6. P(k) comparison: 3-panel figure
# -------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
ax_p0, ax_p82, ax_rat = axes2

pk_lin_at_khm = pk_spline(k_hm)

# --- Panel 0: P(k) at z=0 ---
ax_p0.loglog(k_hm, pk_lin_at_khm,   'k',         lw=2,   label='Linear')
ax_p0.loglog(kh_hf, pk_halofit_z0,  'forestgreen', lw=2, ls='--', label='HaloFit (CAMB)')
ax_p0.loglog(k_hm, pk2h_arr,        'steelblue',  lw=1.5, ls=':',  label='2-halo')
ax_p0.loglog(k_hm, pk1h_arr,        'tomato',     lw=1.5, ls=':',  label='1-halo')
ax_p0.loglog(k_hm, pk_hm_arr,       'firebrick',  lw=2.5,          label='Halo model total')
ax_p0.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax_p0.set_ylabel(r'$P(k,z{=}0)$ [$({\rm Mpc}/h)^3$]', fontsize=12)
ax_p0.set_xlim(1e-3, 50)
ax_p0.set_ylim(1e0, 1e5)
ax_p0.legend(fontsize=10)
ax_p0.set_title('Matter Power Spectrum at $z=0$', fontsize=12)
ax_p0.grid(True, alpha=0.3)

# --- Panel 1: P(k) at z=8.2 ---
ax_p82.loglog(k_hm, pk_lin_z82,    'k',          lw=2,   label='Linear (growth rescaled)')
ax_p82.loglog(k_hm, pk_hm_z82,     'firebrick',  lw=2.5,          label='Halo model')
ax_p82.loglog(k_hm, pk_hm_mod_z82, 'darkorchid', lw=2.5, ls='-.',
              label='Halo model + Gaussian HMF mod')
ax_p82.loglog(k_hm, delta_pk_z82,  'darkorchid', lw=1.5, ls=':',
              label=r'$\Delta P_{1h}$ (extra only)')
ax_p82.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax_p82.set_ylabel(r'$P(k,z{=}8.2)$ [$({\rm Mpc}/h)^3$]', fontsize=12)
ax_p82.set_xlim(1e-3, 50)
ax_p82.legend(fontsize=10)
ax_p82.set_title(fr'Matter Power Spectrum at $z={z_peak}$'
                 '\n(Gaussian HMF modification)', fontsize=12)
ax_p82.grid(True, alpha=0.3)

# --- Panel 2: Boost P/P_lin at z=0 ---
# interpolate halofit onto k_hm grid
pk_hf_at_khm = np.interp(k_hm, kh_hf, pk_halofit_z0)
ax_rat.semilogx(k_hm, pk_hf_at_khm    / pk_lin_at_khm, 'forestgreen', lw=2,   ls='--',
                label='HaloFit / Linear')
ax_rat.semilogx(k_hm, pk_hm_arr       / pk_lin_at_khm, 'firebrick',   lw=2.5,
                label='Halo model / Linear')
ax_rat.semilogx(k_hm, pk2h_arr        / pk_lin_at_khm, 'steelblue',   lw=1.5, ls=':',
                label='2-halo / Linear')
ax_rat.axhline(1.0, color='k', lw=1)
ax_rat.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax_rat.set_ylabel(r'$P(k) / P_{\rm lin}(k)$', fontsize=12)
ax_rat.set_xlim(1e-3, 50)
ax_rat.set_ylim(0, 8)
ax_rat.legend(fontsize=10)
ax_rat.set_title('Nonlinear Boost at $z=0$', fontsize=12)
ax_rat.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig("power_spectrum_comparison.pdf", dpi=150, bbox_inches='tight')
fig2.savefig("power_spectrum_comparison.png", dpi=150, bbox_inches='tight')
print("Saved power_spectrum_comparison.pdf and power_spectrum_comparison.png")

# -------------------------------------------------------------------------
# 7. HMF comparison: dn/dM vs M and dn/dM vs z
# -------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
ax_hm, ax_hz = axes3

# --- Panel 0: HMF vs M at z=8.2 ---
ax_hm.loglog(M_hmf, dndM_tinker_z82, 'firebrick',  lw=2.5,
             label=fr'Tinker 2010, $z={z_peak}$')
ax_hm.loglog(M_hmf, dndM_gauss_z82,  'darkorchid', lw=2, ls=':',
             label=fr'Gaussian extra, $z={z_peak}$')
ax_hm.loglog(M_hmf, dndM_mod_z82,    'darkorchid', lw=2.5, ls='-.',
             label=fr'Tinker + Gaussian, $z={z_peak}$')
ax_hm.set_xlabel(r'$M_h$ [$M_\odot/h$]', fontsize=12)
ax_hm.set_ylabel(r'$dn/dM$ [$({\rm Mpc}/h)^{-3}\,(M_\odot/h)^{-1}$]', fontsize=12)
ax_hm.set_xlim(1e8, 1e16)
ax_hm.legend(fontsize=10)
_gp = GAUSS_PARAMS
ax_hm.set_title(fr'HMF at $z={z_peak}$: '
                fr'$A={_gp["A"]}$, $\log M_0={_gp["logM0"]}$, '
                fr'$\sigma_{{\log M}}={_gp["sigma_logM"]}$',
                fontsize=11)
ax_hm.grid(True, alpha=0.3)

# --- Panel 1: HMF vs z at M=1e11 M_sun/h ---
ax_hz.semilogy(z_hmf, dndM_tinker_vz, 'firebrick',  lw=2.5,
               label=r'Tinker 2010, $M_h=10^{11}\,M_\odot/h$')
ax_hz.semilogy(z_hmf, dndM_gauss_vz,  'darkorchid', lw=2, ls=':',
               label=r'Gaussian extra, $M_h=10^{11}\,M_\odot/h$')
ax_hz.semilogy(z_hmf, dndM_mod_vz,    'darkorchid', lw=2.5, ls='-.',
               label=r'Tinker + Gaussian, $M_h=10^{11}\,M_\odot/h$')
ax_hz.set_xlabel(r'Redshift $z$', fontsize=12)
ax_hz.set_ylabel(r'$dn/dM$ [$({\rm Mpc}/h)^{-3}\,(M_\odot/h)^{-1}$]', fontsize=12)
ax_hz.set_xlim(0, 15)
ax_hz.legend(fontsize=10)
ax_hz.set_title(fr'HMF vs $z$ at $M_h=10^{{11}}\,M_\odot/h$: '
                fr'$z_0={_gp["z0"]}$, $\sigma_z={_gp["sigma_z"]}$',
                fontsize=11)
ax_hz.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("hmf_comparison.pdf", dpi=150, bbox_inches='tight')
fig3.savefig("hmf_comparison.png", dpi=150, bbox_inches='tight')
print("Saved hmf_comparison.pdf and hmf_comparison.png")
print("\nDone!")
