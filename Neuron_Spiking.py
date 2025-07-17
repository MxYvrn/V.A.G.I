import torch
from torch import nn

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    """
    def __init__(self, num_neurons, dt=1e-3, tau_mem=20e-3, v_rest=-65.0, v_reset=-65.0, v_th=-50.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        # membrane potentials per batch, initialized in forward
        self.V = None

    def forward(self, I_ext: torch.Tensor) -> torch.Tensor:
        """
        Advance the membrane potential given input current I_ext.
        Returns binary spikes tensor of shape [B, num_neurons].
        """
        B, N = I_ext.shape
        if self.V is None or self.V.size(0) != B:
            # initialize membrane potential to rest
            self.V = torch.full((B, N), self.v_rest, device=I_ext.device)

        # dV/dt = -(V - V_rest)/tau_mem + I_ext
        dV = (-(self.V - self.v_rest) + I_ext) * (self.dt / self.tau_mem)
        self.V = self.V + dV
        # generate spikes
        spikes = (self.V >= self.v_th).float()
        # reset potentials where spikes occurred
        self.V = torch.where(spikes.bool(), torch.full_like(self.V, self.v_reset), self.V)
        return spikes

class STDP:
    """
    Spike-Timing-Dependent Plasticity (STDP) rule.
    Maintains pre- and post-synaptic traces and updates weights.
    """
    def __init__(self, num_pre, num_post,
                 tau_pre=20e-3, tau_post=20e-3,
                 A_plus=0.01, A_minus=-0.01):
        # time constants for traces
        self.tau_pre = tau_pre
        self.tau_post = tau_pst
        # amplitude constants
        self.A_plus = A_plus
        self.A_minus = A_minus
        # eligibility traces
        self.pre_trace = None  # shape [B, num_pre]
        self.post_trace = None # shape [B, num_post]

    def init_traces(self, batch_size: int, device: torch.device, num_pre: int, num_post: int):
        self.pre_trace = torch.zeros(batch_size, num_pre, device=device)
        self.post_trace = torch.zeros(batch_size, num_post, device=device)

    def update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        Update the pre- and post-synaptic traces based on current spikes.
        pre_spikes: [B, num_pre], post_spikes: [B, num_post]
        """
        # exponential decay
        self.pre_trace = self.pre_trace * torch.exp(-self.tau_pre) + pre_spikes
        self.post_trace = self.post_trace * torch.exp(-self.tau_post) + post_spikes

    def apply_stdp(self, weights: torch.Tensor,
                   pre_spikes: torch.Tensor,
                   post_spikes: torch.Tensor,
                   lr: float = 1e-3) -> torch.Tensor:
        """
        Update synaptic weights according to pair-based STDP rule:
        dw = A_plus * pre_trace * post_spikes + A_minus * post_trace * pre_spikes
        weights shape: [num_pre, num_post]
        pre_spikes: [B, num_pre], post_spikes: [B, num_post]
        returns updated weights
        """
        # ensure traces exist
        if self.pre_trace is None or self.post_trace is None:
            raise RuntimeError("Traces not initialized. Call init_traces().")
        # average over batch
        batch_dw = torch.zeros_like(weights)
        # for each sample in batch, accumulate dw
        # (this can be vectorized for efficiency)
        for b in range(pre_spikes.size(0)):
            pre_t = self.pre_trace[b].unsqueeze(1)   # [num_pre,1]
            post_s = post_spikes[b].unsqueeze(0)      # [1,num_post]
            dw_plus = self.A_plus * pre_t * post_s   # pre before post

            post_t = self.post_trace[b].unsqueeze(0) # [1,num_post]
            pre_s  = pre_spikes[b].unsqueeze(1)      # [num_pre,1]
            dw_minus = self.A_minus * pre_s * post_t # post before pre

            batch_dw += (dw_plus + dw_minus)

        # normalize by batch and apply learning rate
        dw = lr * (batch_dw / pre_spikes.size(0))
        weights = weights + dw
        return weights
