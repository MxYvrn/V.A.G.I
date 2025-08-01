import numpy as np

def lif_sim(I,
            dt      = 1e-3,     # s
            tau_m   = 20e-3,    # s
            V_rest  = -0.065,   # V
            V_th    = -0.050,   # V
            V_reset = -0.065,   # V
            R       = 1.0,      # Ω   (normalised)
            t_ref   = 2e-3):    # s
    """
    Vectorised LIF simulator.

    Parameters
    ----------
    I : ndarray, shape (T, N)
        External input current in amperes.
    dt, tau_m, V_rest, V_th, V_reset, R, t_ref : float
        Standard LIF parameters (see documentation).

    Returns
    -------
    spikes : ndarray, shape (T, N), dtype=np.int8
        1 where a spike occurred, else 0.
    V_trace : ndarray, shape (T, N)
        Membrane potential for every time-step (optional for analysis).
    """
    T, N         = I.shape
    step_fac     = dt / tau_m             # pre-compute constant
    ref_steps    = int(t_ref / dt)        # refractory period in steps

    V        = np.full(N, V_rest, dtype=np.float32)
    ref_ctr  = np.zeros(N, dtype=np.int32)
    spikes   = np.zeros((T, N), dtype=np.int8)
    V_trace  = np.empty((T, N),  dtype=np.float32)

    for t in range(T):
        active        = ref_ctr == 0                 # not in refractory
        V[active]    += step_fac * (-(V[active] - V_rest) + R * I[t, active])

        spk_mask      = V >= V_th                    # threshold test
        if spk_mask.any():
            spikes[t, spk_mask] = 1
            V[spk_mask]        = V_reset
            ref_ctr[spk_mask]  = ref_steps

        V_trace[t]    = V                            # record state
        ref_ctr[ref_ctr > 0] -= 1                    # countdown

    return spikes, V_trace
