# Data card

What you are given, what it means, and what is done to it before your code is scored.

## Provenance

[CAMELS](https://camels.readthedocs.io) — Cosmology and Astrophysics with MachinE Learning
Simulations. Specifically the **latin-hypercube (LH) set** at `L25n256` resolution: a
25 cMpc/h periodic box with 256³ dark-matter particles and, in the hydrodynamic runs, 256³ gas
elements. Every simulation is run with a different combination of six parameters.

We ship the **halo and subhalo catalogs at snapshot 090**, which is redshift 0 — the
present-day universe. These are the output of the FoF and Subfind algorithms: not the particles
themselves, but the objects found in them.

| suite | hydrodynamics code | role here |
|---|---|---|
| IllustrisTNG | Arepo | public, 900 simulations with labels |
| SIMBA | Gizmo | public, 900 simulations with labels |
| Astrid | MP-Gadget | public, 900 simulations with labels |
| *(a fourth)* | *(a different code)* | **never shipped.** 100 simulations, used only to score you |

The fourth suite is the point of the event. Nothing about it is published, including its name,
and you cannot obtain it — which is what makes it a real test of whether a model transfers to a
simulation code it has never seen.

## Units, and the two mistakes everyone makes

CAMELS uses the Gadget convention. Nothing raises an error if you get this wrong; the numbers
are simply incorrect.

| quantity | stored as | to physical |
|---|---|---|
| positions, radii | ckpc/h | × 10⁻³ / h → cMpc |
| masses | 10¹⁰ M☉/h | × 10¹⁰ / h → M☉ |
| velocities | km/s | already peculiar velocity at z = 0 |

`HubbleParam` = 0.6711. The box is **25000 ckpc/h and periodic**: a galaxy at x = 24900 and one
at x = 100 are 200 ckpc/h apart, not 24800. Every distance needs the minimum-image convention.
`scipy.spatial.cKDTree(positions, boxsize=25000.0)` applies it for you.

The second mistake is the particle-type axis. Several columns are 2-D with six entries per row:

| index | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| type | gas | dark matter | — | — | **stars** | black holes |

So `SubhaloMassType[:, 4]` is stellar mass and `SubhaloMassType[:, 1]` is dark-matter mass.

## What is in a file

Two tables, linked. `Group/` has one row per FoF halo, `Subhalo/` one row per Subfind subhalo,
and `Subhalo/SubhaloGrNr` gives the row in `Group/` that a subhalo belongs to.

A **galaxy** is not a separate table. It is a subhalo with stars, above a stellar-mass threshold
you choose. Most subhalos have none.

Typical sizes, measured over 60 simulations per suite (median, and the range):

| suite | subhalos | M⋆ > 1.3×10⁸ M☉/h | M⋆ > 1.95×10⁸ M☉/h |
|---|---|---|---|
| IllustrisTNG | 17054 (10904–20645) | 709 (145–1822) | 607 (118–1616) |
| SIMBA | 16114 (12807–19038) | 1046 (404–3001) | 840 (307–2363) |
| Astrid | 19303 (13854–22852) | 938 (73–3966) | 740 (46–3592) |

The spread within a suite is a factor of ten or more. That is not noise — it is the parameters
doing their work, and it is most of what a model reads.

## The field contract

Verified by a direct schema comparison of all four suites:

- **`Group/`: 25 columns, identical in every suite.**
- **`Subhalo/`: 50 columns in the three public suites, 47 in the held-out one.**

The held-out suite is missing exactly three, all of them photometry:

```
SubhaloStellarPhotometrics
SubhaloStellarPhotometricsMassInRad
SubhaloStellarPhotometricsRad
```

**So 25 + 47 columns are guaranteed to exist everywhere, and those three are not.** A model that
depends on them will fail or silently degrade on the condition that matters most. This is not
compensated for. It is a real, physically meaningful instance of a feature you cannot count on,
and noticing it is part of the task.

There is no curated column subset. You get the complete catalog and choose your own features.

## The split

900 public and 100 held-out per public suite, partitioned **by simulation id** with
`SPLIT_SEED = 20260825`, reproducible from `kaai_hackathon.splits.make_split`.

**The public ids are not `LH_0` … `LH_899`.** They are a pinned random 900 out of 1000, so
`LH_2` may simply not be in your copy. Use `kaai_hackathon.splits.local_split(suite)`, which
splits the simulations you actually have into `train` and `test`, or `example_sims(suite, n)`
for ids that are certain to exist.

Split by simulation, never below it. Two galaxies from one simulation share a cosmology and are
not independent samples; a split that separates them leaks the answer into your own validation
set and you will believe a score that is not real.

## The labels

Six parameters per simulation, in `params_LH_<suite>.txt`, **row `n` is `LH_n`** — a positional
mapping, so never zip a sorted list of directory names against it (`LH_10` sorts before `LH_2`).

| | meaning | range |
|---|---|---|
| `Omega_m` | matter fraction of the energy density | [0.10, 0.50], uniform |
| `sigma_8` | clumpiness of matter on 8 Mpc/h scales | [0.60, 1.00], uniform |
| `A_SN1`, `A_AGN1` | feedback strength multipliers | [0.25, 4.00], log-uniform |
| `A_SN2`, `A_AGN2` | feedback strength multipliers | [0.50, 2.00], log-uniform |

**Astrid's `A_AGN2` runs over [0.25, 4.00], not [0.50, 2.00].** The suites do not share ranges;
check rather than assume.

`Omega_m` and `sigma_8` are the scored targets. The four feedback parameters are an optional
bonus track, scored separately.

## What is done to a catalog before your code sees it

Every catalog you are evaluated on — including the `clean` condition, which goes through the
same writer as an identity operation — is rewritten. The writer emits exactly:

- `Group/` — all 25 columns
- `Subhalo/` — every column present in that suite
- `Header` — rebuilt, with only `BoxSize`, `Redshift`, `HubbleParam`, `NumFiles`,
  `Ngroups_Total`, `Nsubgroups_Total`

Everything else is gone. `Parameters` and `Config` are never written, so the following, which
are all present in the raw CAMELS files, do not reach your code:

| attribute | what it would give away |
|---|---|
| `Header/Omega0`, `Parameters/Omega0` | `Omega_m`, exactly |
| `Header/OmegaLambda` | `Omega_m`, as 1 − Omega_m |
| `Parameters/RadioFeedbackFactor` and three others | the four feedback parameters, exactly |
| `Header/Git_commit` | which simulation code wrote the file |
| `IDs/` | the member particle ids |

`sigma_8` never appears in any attribute — it enters through the initial conditions rather than
the runtime parameters, which is one reason it is the harder target.

Reading `Header`, `Parameters` or `Config` for anything beyond box size and redshift is
therefore pointless as well as against the rules.
