# excitation.py
# =============================================================
# Impact force pulse definition and global force vector builder
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
from config import N_NODES, N_STEPS, DT, PULSE_DURATION, BOUNDARY_CONDITION

# -------------------------------------------------------------
# UNIT NOTE
# F0      : lbf
# tau     : seconds
# F_vector: lbf  (force at each free DOF at each timestep)
# -------------------------------------------------------------

def half_sine_pulse(F0, tau, dt, n_steps):
    """
    Generate half-sine impact pulse time history.

    F(t) = F0 * sin(pi * t / tau)   for 0 <= t <= tau
    F(t) = 0                         for t > tau

    Inputs:
        F0      : peak impact force (lbf)
        tau     : pulse duration (s)
        dt      : timestep size (s)
        n_steps : total number of timesteps

    Output:
        force_time : 1D array of force values (lbf), length = n_steps
        time_vector: 1D array of time values (s),  length = n_steps
    """
    time_vector = np.linspace(0, dt * (n_steps - 1), n_steps)
    force_time  = np.zeros(n_steps)

    # Apply half sine pulse where t <= tau
    pulse_mask             = time_vector <= tau
    force_time[pulse_mask] = F0 * np.sin(np.pi * time_vector[pulse_mask] / tau)

    return force_time, time_vector


def build_force_vector(F0, free_dofs, tau=PULSE_DURATION, dt=DT,
                       n_steps=N_STEPS, bc_type=BOUNDARY_CONDITION):
    """
    Build global force vector for all timesteps.

    Force applied at free end transverse DOF only.
    All other DOFs receive zero force.

    Inputs:
        F0        : peak impact force (lbf)
        free_dofs : array of free DOF indices (from assembly)
        tau       : pulse duration (s)
        dt        : timestep size (s)
        n_steps   : number of timesteps
        bc_type   : boundary condition type string

    Output:
        F_global  : 2D array shape (n_free_dofs, n_steps)
        force_time: 1D array of force values at tip (lbf)
        time_vector: 1D array of time values (s)
    """
    n_free = len(free_dofs)

    # Generate pulse time history
    force_time, time_vector = half_sine_pulse(F0, tau, dt, n_steps)

    # Initialize force matrix — all zeros
    F_global = np.zeros((n_free, n_steps))

    # Find which free DOF corresponds to the impact location
    impact_dof_global = get_impact_dof(bc_type)

    # Map global DOF index to free DOF index
    impact_dof_free = np.where(free_dofs == impact_dof_global)[0]

    if len(impact_dof_free) == 0:
        raise ValueError(f"Impact DOF {impact_dof_global} not found in free DOFs. "
                         f"Check boundary condition.")

    impact_dof_free = impact_dof_free[0]

    # Assign pulse to impact DOF row
    F_global[impact_dof_free, :] = force_time

    return F_global, force_time, time_vector


def get_impact_dof(bc_type):
    """
    Return the global DOF index where impact force is applied.

    For cantilever: free end is node 100, transverse DOF = 2*100 = 200
    This can be extended for other BC types easily.

    Inputs:
        bc_type : boundary condition string

    Output:
        impact_dof : global DOF index (int)
    """
    if bc_type == 'cantilever':
        # Free end = last node = node 100
        # Transverse DOF of node 100 = 2 * 100 = 200
        impact_dof = 2 * (N_NODES - 1)

    elif bc_type == 'simply_supported':
        # Apply at ~22% span (node 22) to excite both odd and even modes.
        # Midspan (node 50) is a nodal point for all even modes,
        # so hitting there would miss modes 2, 4, 6, ...
        # 0.22L avoids nodal points of the first ~8 modes.
        impact_node = int(round(0.22 * (N_NODES - 1)))   # node 22
        impact_dof  = 2 * impact_node

    elif bc_type == 'fixed_fixed':
        # Same reasoning as simply_supported — avoid midspan.
        impact_node = int(round(0.22 * (N_NODES - 1)))   # node 22
        impact_dof  = 2 * impact_node

    else:
        raise ValueError(f"Unknown BC type: '{bc_type}'")

    return impact_dof