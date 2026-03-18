# config.py
# =============================================================
# Central configuration file for Euler-Bernoulli Beam FEA
# All units in IPS system: inches, lbf, seconds
# =============================================================

# -------------------------------------------------------------
# UNIT SYSTEM NOTE
# Length   : inches (in)
# Force    : pound-force (lbf)
# Time     : seconds (s)
# Pressure : psi (lbf/in²)
# Mass     : lbf·s²/in  (consistent mass unit)
# -------------------------------------------------------------

# Gravitational constant (for density unit conversion)
G_C = 386.088  # in/s²

# -------------------------------------------------------------
# GEOMETRY
# -------------------------------------------------------------
THICKNESS_VALUES = [0.4
                   # , 0.1, 0.3
                   # , 0.5
                    ]  # t (in)

LENGTH_VALUES = [50
                 # , 60, 80, 100, 120, 140, 160, 180
                 ]  # L (in)

WIDTH_VALUES  = [2 
                # ,0.5, 1 ,1.5 
                # ,2.5,  3 , 4 , 5
                 ]             # b (in)

# -------------------------------------------------------------
# IMPACT FORCE
# -------------------------------------------------------------
FORCE_VALUES   = [5
                # , 20, 45, 60
                 #, 75, 90, 150, 300, 500
                  ]  # F0 (lbf)
PULSE_DURATION = 0.0021  # tau, impact duration (s) — 25ms pulse

# -------------------------------------------------------------
# TIME SETTINGS
# Matched to Photron FASTCAM at 2000 fps
# dt = 0.0005 s, f_max captured = 1000 Hz
# Resolution at 2000 fps: 1024 x 500 pixels
# -------------------------------------------------------------
T_TOTAL = 1.0   # total simulation time (s)
N_STEPS = 2000  # number of timesteps (matches SA5 at 2000 fps)
DT      = T_TOTAL / (N_STEPS - 1)  # 0.0005005 s per step

# -------------------------------------------------------------
# MESH
# -------------------------------------------------------------
N_ELEMENTS = 100  # number of beam elements
N_NODES    = 101  # number of nodes (N_ELEMENTS + 1)

# -------------------------------------------------------------
# RAYLEIGH DAMPING
# Each tuple is (alpha, beta) — one dataset is produced per pair.
# alpha : mass-proportional coefficient
# beta  : stiffness-proportional coefficient
# -------------------------------------------------------------
RAYLEIGH_DAMPING_PAIRS = [
    (0.005,  0.000005),   # low damping    (~0.1–0.3 % zeta)
    # (0.01,   0.00001),  # medium damping (~0.5–1.0 % zeta) — original default
    # (0.05,   0.00005),  # high damping   (~1.0–2.0 % zeta)
]

# Backward-compatibility aliases (used by damping.py / other imports)
RAYLEIGH_ALPHA = RAYLEIGH_DAMPING_PAIRS[0][0]
RAYLEIGH_BETA  = RAYLEIGH_DAMPING_PAIRS[0][1]

# -------------------------------------------------------------
# NEWMARK-BETA PARAMETERS
# -------------------------------------------------------------
NEWMARK_BETA  = 0.25  # average acceleration (unconditionally stable)
NEWMARK_GAMMA = 0.50

# -------------------------------------------------------------
# BOUNDARY CONDITION
# -------------------------------------------------------------
# Supported types: 'cantilever', 'simply_supported', 'fixed_fixed'
BOUNDARY_CONDITIONS = ['cantilever', 'simply_supported', 'fixed_fixed']
BOUNDARY_CONDITION  = 'cantilever'   # default for single-BC runs (backward compat)

# -------------------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------------------
N_SIMULATIONS = None  # None = all unique combinations (modes 1 & 2)
                      # Set an integer when using LHS (mode 3), e.g. 200
RANDOM_SEED   = 42

OUTPUT_DIR    = 'simulation_results'  # folder for all CSV output files
N_JOBS        = -1                    # parallel CPU cores (-1 = all available)
SAMPLING_MODE = 1                     # 1 = discrete combinations (all permutations)
                                      # 2 = continuous uniform random
                                      # 3 = Latin Hypercube Sampling (LHS) — best for ML
ENCODING      = 'C'                   # CSV encoding: 'A' serialized string per node
                                      #               'B' one column per node per timestep
                                      #               'C' one row per timestep (recommended)

# -------------------------------------------------------------
# MATERIALS AVAILABLE
# -------------------------------------------------------------
MATERIAL_NAMES = ['steel', 'aluminum']

# -------------------------------------------------------------
# GAUSSIAN NOISE (sensor output noise)
# Units match the acceleration signal: in/s²
# Typical range for good speckle + good camera: 0.01 – 0.05
# One separate CSV is exported per noise level.
# -------------------------------------------------------------
ADD_NOISE = True   # True  → export noisy CSVs (one per level in NOISE_STD_LEVELS)
                   # False → export clean CSVs only, noise step is skipped

NOISE_STD_LEVELS = [0.01,   # low noise  — good speckle, good camera
                    0.05    # higher noise — moderate measurement uncertainty
                    ]