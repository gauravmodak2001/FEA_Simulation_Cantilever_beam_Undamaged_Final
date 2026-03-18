# sampling.py
# =============================================================
# Parameter set generator for beam FEA simulations
# Produces randomized combinations of beam parameters
# All units in IPS system: inches, lbf, seconds
#
# Three sampling modes:
#   Mode 1 — Discrete:              full Cartesian product of all candidate lists
#   Mode 2 — Continuous uniform:    uniform random draws from value ranges
#   Mode 3 — Latin Hypercube (LHS): space-filling design, best for ML training
# =============================================================

import numpy as np
import random
from itertools import product
from materials import get_material
from config import (
    LENGTH_VALUES, WIDTH_VALUES, THICKNESS_VALUES, FORCE_VALUES,
    MATERIAL_NAMES, BOUNDARY_CONDITIONS, PULSE_DURATION,
    RAYLEIGH_DAMPING_PAIRS,
    T_TOTAL, N_STEPS, DT,
    N_ELEMENTS, N_NODES,
    RANDOM_SEED, N_SIMULATIONS
)


def generate_parameter_sets(
        length_values       = LENGTH_VALUES,
        width_values        = WIDTH_VALUES,
        thickness_values    = THICKNESS_VALUES,
        force_values        = FORCE_VALUES,
        material_names      = MATERIAL_NAMES,
        boundary_conditions = BOUNDARY_CONDITIONS,
        rayleigh_pairs      = RAYLEIGH_DAMPING_PAIRS,
        n_simulations       = N_SIMULATIONS,
        seed                = RANDOM_SEED,
        mode                = 1):
    """
    Generate parameter sets for beam FEA simulations.

    Mode 1 — Discrete permutation:
        Builds all combinations of L x b x t x F0 x material x bc x damping,
        shuffles with seed, takes first n_simulations.

    Mode 2 — Continuous uniform random:
        Draws uniformly from value ranges; randomly picks material,
        BC type, and damping pair.

    Mode 3 — Latin Hypercube Sampling (LHS):
        Space-filling design — divides each parameter dimension into
        n_simulations equal strata and samples exactly one point per
        stratum. Ensures uniform coverage across the full parameter space,
        which is significantly better than random sampling for ML training.
        Requires n_simulations to be set to an integer (not None).

    Inputs:
        length_values       : list of candidate lengths (in)
        width_values        : list of candidate widths (in)
        thickness_values    : list of candidate thicknesses (in)
        force_values        : list of candidate forces (lbf)
        material_names      : list of material name strings
        boundary_conditions : list of BC type strings
        rayleigh_pairs      : list of (alpha, beta) tuples
        n_simulations       : number of parameter sets to generate
                              (must be an integer for mode 3)
        seed                : random seed for reproducibility
        mode                : 1 = discrete, 2 = continuous, 3 = LHS

    Output:
        param_sets : list of dicts, each containing all parameters
                     for one simulation run
    """
    random.seed(seed)
    np.random.seed(seed)

    if mode == 1:
        return _discrete_mode(
            length_values, width_values, thickness_values,
            force_values, material_names, boundary_conditions,
            rayleigh_pairs, n_simulations, seed)

    elif mode == 2:
        return _continuous_mode(
            length_values, width_values, thickness_values,
            force_values, material_names, boundary_conditions,
            rayleigh_pairs, n_simulations, seed)

    elif mode == 3:
        if n_simulations is None:
            raise ValueError(
                "n_simulations must be set to an integer for LHS (mode 3). "
                "Set N_SIMULATIONS in config.py, e.g. N_SIMULATIONS = 200.")
        return _lhs_mode(
            length_values, width_values, thickness_values,
            force_values, material_names, boundary_conditions,
            rayleigh_pairs, n_simulations, seed)

    else:
        raise ValueError(f"Unknown sampling mode {mode}. Use 1, 2, or 3.")


# =============================================================
# MODE 1 — DISCRETE PERMUTATION
# =============================================================

def _discrete_mode(length_values, width_values, thickness_values,
                   force_values, material_names, boundary_conditions,
                   rayleigh_pairs, n_simulations, seed):
    """
    Build all combinations of every candidate list, shuffle, take first N.
    Dimensions: L x b x t x F0 x material x bc_type x (alpha, beta)
    """
    all_combos = list(product(
        length_values,
        width_values,
        thickness_values,
        force_values,
        material_names,
        boundary_conditions,
        rayleigh_pairs          # each element is a (alpha, beta) tuple
    ))

    total_combos = len(all_combos)
    print(f"Total possible combinations : {total_combos}")

    if n_simulations is None:
        n_simulations = total_combos

    print(f"Simulations requested       : {n_simulations}")

    random.seed(seed)
    random.shuffle(all_combos)

    if total_combos < n_simulations:
        print(f"  Only {total_combos} unique combos available — capping to {total_combos}.")
        n_simulations = total_combos

    selected   = all_combos[:n_simulations]
    param_sets = []
    for sim_id, (L, b, t, F0, mat_name, bc_type, (alpha, beta_r)) in enumerate(selected):
        mat   = get_material(mat_name)
        entry = _build_param_dict(sim_id, L, b, t, F0, mat_name, mat,
                                  bc_type, alpha, beta_r)
        param_sets.append(entry)

    return param_sets


# =============================================================
# MODE 2 — CONTINUOUS UNIFORM RANDOM
# =============================================================

def _continuous_mode(length_values, width_values, thickness_values,
                     force_values, material_names, boundary_conditions,
                     rayleigh_pairs, n_simulations, seed):
    """
    Uniform random draws from ranges; random picks for categorical params.
    """
    if n_simulations is None:
        raise ValueError(
            "n_simulations must be set for continuous mode (mode 2). "
            "Set N_SIMULATIONS in config.py.")

    np.random.seed(seed)

    L_min,  L_max  = min(length_values),    max(length_values)
    b_min,  b_max  = min(width_values),     max(width_values)
    t_min,  t_max  = min(thickness_values), max(thickness_values)
    F0_min, F0_max = min(force_values),     max(force_values)

    param_sets = []
    for sim_id in range(n_simulations):
        L               = np.random.uniform(L_min, L_max)
        b               = np.random.uniform(b_min, b_max)
        t               = np.random.uniform(t_min, t_max)
        F0              = np.random.uniform(F0_min, F0_max)
        mat_name        = np.random.choice(material_names)
        bc_type         = np.random.choice(boundary_conditions)
        alpha, beta_r   = rayleigh_pairs[np.random.randint(len(rayleigh_pairs))]
        mat             = get_material(mat_name)
        entry           = _build_param_dict(sim_id, L, b, t, F0, mat_name, mat,
                                            bc_type, alpha, beta_r)
        param_sets.append(entry)

    return param_sets


# =============================================================
# MODE 3 — LATIN HYPERCUBE SAMPLING (LHS)
# =============================================================

def _lhs_mode(length_values, width_values, thickness_values,
              force_values, material_names, boundary_conditions,
              rayleigh_pairs, n_simulations, seed):
    """
    Latin Hypercube Sampling across 7 dimensions:
        0: L          (continuous, range from length_values)
        1: b          (continuous, range from width_values)
        2: t          (continuous, range from thickness_values)
        3: F0         (continuous, range from force_values)
        4: material   (categorical, index into material_names)
        5: bc_type    (categorical, index into boundary_conditions)
        6: damping    (categorical, index into rayleigh_pairs)

    Each dimension is stratified into n_simulations equal-width bins.
    Exactly one sample is drawn per bin in every dimension.
    Bins are then randomly permuted across dimensions so the joint
    distribution covers the space uniformly — unlike pure random sampling,
    which can leave gaps and clusters.
    """
    rng    = np.random.default_rng(seed)
    n_dims = 7

    # Build LHS sample matrix: shape (n_simulations, n_dims), values in [0, 1)
    # Each column is a permutation of (bin + uniform offset) / n_simulations
    samples = np.zeros((n_simulations, n_dims))
    for d in range(n_dims):
        perm          = rng.permutation(n_simulations)          # which bin each row gets
        offset        = rng.uniform(size=n_simulations)          # random offset within bin
        samples[:, d] = (perm + offset) / n_simulations

    # --- Map continuous dimensions ---
    L_min,  L_max  = min(length_values),    max(length_values)
    b_min,  b_max  = min(width_values),     max(width_values)
    t_min,  t_max  = min(thickness_values), max(thickness_values)
    F0_min, F0_max = min(force_values),     max(force_values)

    L_vals  = samples[:, 0] * (L_max  - L_min)  + L_min
    b_vals  = samples[:, 1] * (b_max  - b_min)  + b_min
    t_vals  = samples[:, 2] * (t_max  - t_min)  + t_min
    F0_vals = samples[:, 3] * (F0_max - F0_min) + F0_min

    # --- Map categorical dimensions ---
    mat_idx     = np.floor(samples[:, 4] * len(material_names)).astype(int).clip(0, len(material_names)     - 1)
    bc_idx      = np.floor(samples[:, 5] * len(boundary_conditions)).astype(int).clip(0, len(boundary_conditions) - 1)
    damping_idx = np.floor(samples[:, 6] * len(rayleigh_pairs)).astype(int).clip(0, len(rayleigh_pairs)    - 1)

    print(f"Latin Hypercube Sampling    : {n_simulations} samples across 7 dimensions")
    print(f"  L range           : [{L_min}, {L_max}] in")
    print(f"  b range           : [{b_min}, {b_max}] in")
    print(f"  t range           : [{t_min}, {t_max}] in")
    print(f"  F0 range          : [{F0_min}, {F0_max}] lbf")
    print(f"  Materials         : {material_names}")
    print(f"  Boundary conds    : {boundary_conditions}")
    print(f"  Damping pairs     : {rayleigh_pairs}")

    param_sets = []
    for sim_id in range(n_simulations):
        mat_name      = material_names[mat_idx[sim_id]]
        bc_type       = boundary_conditions[bc_idx[sim_id]]
        alpha, beta_r = rayleigh_pairs[damping_idx[sim_id]]
        mat           = get_material(mat_name)
        entry         = _build_param_dict(
            sim_id,
            float(L_vals[sim_id]),
            float(b_vals[sim_id]),
            float(t_vals[sim_id]),
            float(F0_vals[sim_id]),
            mat_name, mat, bc_type, alpha, beta_r
        )
        param_sets.append(entry)

    return param_sets


# =============================================================
# SHARED HELPER
# =============================================================

def _build_param_dict(sim_id, L, b, t, F0, mat_name, mat,
                      bc_type, alpha, beta_r):
    """
    Build a single simulation parameter dictionary.
    Contains all parameters needed for one simulation run.
    """
    return {
        # Simulation ID
        'sim_id'        : sim_id,

        # Boundary condition
        'bc_type'       : bc_type,

        # Material
        'material'      : mat_name,
        'E_psi'         : mat['E'],
        'rho_lbm_in3'   : mat['rho_lbm'],
        'rho_consistent': mat['rho'],

        # Geometry
        'length_in'     : L,
        'width_in'      : b,
        'thickness_in'  : t,

        # Impact
        'impact_F0_lbf' : F0,
        'impact_tau_s'  : PULSE_DURATION,

        # Damping
        'rayleigh_alpha': alpha,
        'rayleigh_beta' : beta_r,

        # Time
        'dt_s'          : DT,
        'T_s'           : T_TOTAL,
        'n_steps'       : N_STEPS,

        # Mesh
        'n_elements'    : N_ELEMENTS,
        'n_nodes'       : N_NODES,
    }
