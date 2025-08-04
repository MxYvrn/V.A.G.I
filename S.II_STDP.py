import numpy as np

class STDP:
    """
    Pair-based online STDP for a fully connected weight matrix W (posts × pres).
    Call `step(pre_spk, post_spk)` each simulation tick.
    """
    def __init__(self, W,
                 dt       = 1e-3,
                 tau_plus = 20e-3, tau_minus = 20e-3,
                 A_plus   = 0.01,  A_minus   = 0.012,
                 w_min    = 0.0,   w_max     = 1.0):

        self.W      = W.astype(np.float32)
        self.P      = np.zeros(W.shape[1], dtype=np.float32)  # pre trace  (size N_pre)
        self.Q      = np.zeros(W.shape[0], dtype=np.float32)  # post trace (size N_post)

        self.decay_P = np.exp(-dt / tau_plus).astype(np.float32)
        self.decay_Q = np.exp(-dt / tau_minus).astype(np.float32)

        self.A_plus  = A_plus
        self.A_minus = A_minus
        self.w_min   = w_min
        self.w_max   = w_max

    def step(self, pre_spk, post_spk):
        """
        pre_spk  : shape (N_pre,)  – 0/1 presynaptic spikes at this Δt
        post_spk : shape (N_post,) – 0/1 postsynaptic spikes at this Δt
        """

        # --- decay and add new spikes to traces ---
        self.P *= self.decay_P
        self.Q *= self.decay_Q
        self.P += pre_spk
        self.Q += post_spk

        # --- weight potentiation (pre fires now, look at recent post) ---
        if pre_spk.any():
            self.W += self.A_plus * np.outer(self.Q, pre_spk)

        # --- weight depression (post fires now, look at recent pre) ---
        if post_spk.any():
            self.W -= self.A_minus * np.outer(post_spk, self.P)

        # --- hard bounds ---
        np.clip(self.W, self.w_min, self.w_max, out=self.W)
