"""Toolkit for the KAAI 2026 robust-inference hackathon.

The data is CAMELS: galaxy catalogs from cosmological hydrodynamic simulations.

Dependencies are limited to numpy, scipy, h5py, torch and scikit-learn so the
participant environment stays small and fast to install.
"""

__version__ = "0.3.0"

BOX_SIZE = 25000.0  # ckpc/h, periodic
HUBBLE_PARAM = 0.6711
PARAM_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
PUBLIC_SUITES = ("IllustrisTNG", "SIMBA", "Astrid")
OOD_SUITE = "Swift-EAGLE"

# Present in TNG/SIMBA/Astrid but absent from Swift-EAGLE, which is why a model that
# leans on photometry degrades on the out-of-distribution condition.
SWIFT_MISSING_SUBHALO_FIELDS = (
    "SubhaloStellarPhotometrics",
    "SubhaloStellarPhotometricsMassInRad",
    "SubhaloStellarPhotometricsRad",
)
