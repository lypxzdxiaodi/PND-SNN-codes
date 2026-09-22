"""Core PND layers used by the experiments.

The historical public class names are retained so that old experiment scripts
and checkpoints remain readable.  New code should treat ``psnns4`` as the main
PND layer and the other classes as ablations.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


class DropoutNd(nn.Module):
    """Drop complete feature channels while sharing a mask over time."""

    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = x.shape[:2] + (1,) * (x.ndim - 2)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    """Convert an image represented as a sequence into patch tokens."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)
        self.num_patches = (
            self.img_size[0] // self.patch_size[0]
        ) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, *self.img_size)
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"expected image size {self.img_size}, got {x.shape[-2:]}")
        return self.proj(x).flatten(2).transpose(1, 2)


class MyKernel(nn.Module):
    """Learned complex diagonal temporal response bank.

    Each feature channel owns ``N/2`` conjugate modal components.  Their real
    sum produces the real-valued convolution kernel consumed by the PND layer.
    ``N`` must be even because the implementation stores one member of each
    conjugate pair.
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        if N % 2:
            raise ValueError("d_state/N must be even")
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
        log_dt = log_dt + math.log(dt_min)
        coefficients = torch.randn(d_model, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(coefficients))
        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", torch.log(0.5 * torch.ones(d_model, N // 2)), lr)
        frequencies = math.pi * torch.arange(N // 2).repeat(d_model, 1)
        self.register("A_imag", frequencies, lr)

    def forward(self, L: int):
        dt = torch.exp(self.log_dt)
        poles = -torch.exp(self.log_A_real) + 1j * self.A_imag
        coefficients = torch.view_as_complex(self.C)
        discrete_poles = poles * dt.unsqueeze(-1)
        time = torch.arange(L, device=poles.device)
        exponents = discrete_poles.unsqueeze(-1) * time
        coefficients = coefficients * torch.expm1(discrete_poles) / poles
        kernel = 2 * torch.einsum(
            "hn,hnl->hl", coefficients, torch.exp(exponents)
        ).real
        return kernel, coefficients.real

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
            return
        parameter = nn.Parameter(tensor)
        self.register_parameter(name, parameter)
        parameter._optim = {"weight_decay": 0.0}
        if lr is not None:
            parameter._optim["lr"] = lr


def _fft_convolution(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    length = u.size(-1)
    u_frequency = torch.fft.rfft(u, n=2 * length)
    k_frequency = torch.fft.rfft(kernel, n=2 * length)
    return torch.fft.irfft(u_frequency * k_frequency, n=2 * length)[..., :length]


class Snn_s4(nn.Module):
    """Historical hard-Gumbel PND layer (main early-method ablation)."""

    def __init__(
        self,
        d_model,
        tau=1.0,
        d_state=128,
        dropout=0.0,
        transposed=True,
        **kernel_args,
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = d_model
        self.transposed = transposed
        self.tau = float(tau)
        self.D = nn.Parameter(torch.randn(d_model))
        self.aver = nn.Parameter(torch.rand(d_model))
        self.kernel = MyKernel(d_model, N=d_state, **kernel_args)
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def membrane(self, u: torch.Tensor) -> torch.Tensor:
        kernel, _ = self.kernel(L=u.size(-1))
        return _fft_convolution(u, kernel) + u * self.D.unsqueeze(-1)

    def firing_probability(self, membrane: torch.Tensor) -> torch.Tensor:
        evidence = F.relu(self.aver.unsqueeze(-1) * membrane)
        return -torch.expm1(-evidence)

    def _hard_gumbel(self, probability: torch.Tensor, hard: bool) -> torch.Tensor:
        safe = probability.clamp(1e-7, 1.0 - 1e-7)
        logits = torch.stack([torch.log1p(-safe), torch.log(safe)], dim=-1)
        return F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)[..., 1]

    def _project(self, spikes: torch.Tensor) -> torch.Tensor:
        return self.output_linear(self.dropout(spikes))

    def forward(self, u, spike=True, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        probability = self.firing_probability(self.membrane(u))
        output = self._project(self._hard_gumbel(probability, hard=spike))
        if not self.transposed:
            output = output.transpose(-1, -2)
        return output, None

    def pif(self, x, beta=0.5, threshold=1.0, spike=False):
        """Historical parallel integrate-and-fire comparison retained for ablation."""
        length = x.size(-1)
        dtype, device = x.dtype, x.device
        steps = torch.arange(length, device=device, dtype=dtype)
        first = torch.fft.rfft(torch.exp(steps * math.log(beta)), n=2 * length)
        second = torch.fft.rfft(
            torch.exp(steps * math.log(1.0 - beta)), n=2 * length
        )
        coefficients = torch.fft.irfft(first * second, n=2 * length)[..., :length]
        output = _fft_convolution(x, coefficients)
        return (output > threshold).to(dtype) if spike else output

    def my_gumbel_softmax(self, logit, hard=True, eps=1e-7, dim=-1):
        """Historical binary relaxation retained for reproducibility."""
        if self.tau != 1.0:
            raise ValueError("the historical binary relaxation is defined for tau=1")
        uniform = torch.rand_like(logit).clamp(eps, 1.0 - eps)
        odds = uniform / (1.0 - uniform)
        survival = torch.exp(-logit).clamp(max=1.0 - eps)
        odds = odds * survival / (1.0 - survival)
        soft = 1.0 / (1.0 + odds)
        pair = torch.stack([soft, 1.0 - soft], dim=-1)
        if hard:
            index = pair.max(dim, keepdim=True)[1]
            discrete = torch.zeros_like(pair).scatter_(dim, index, 1.0)
            pair = discrete - pair.detach() + pair
        return pair[..., 0]


class bsnns4(Snn_s4):
    """Bernoulli-forward, probability-ST ablation."""

    def forward(self, u, spike=True, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        probability = self.firing_probability(self.membrane(u))
        if spike:
            sampled = torch.bernoulli(probability)
            spikes = sampled - probability.detach() + probability
        else:
            spikes = probability
        output = self._project(spikes)
        if not self.transposed:
            output = output.transpose(-1, -2)
        return output, None


class psnns4(Snn_s4):
    """Main PND: hard Gumbel-Max forward with probability-ST gradients."""

    def forward(self, u, spike=True, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        probability = self.firing_probability(self.membrane(u))
        if spike:
            sampled = self._hard_gumbel(probability, hard=True).detach()
            spikes = sampled - probability.detach() + probability
        else:
            spikes = probability
        output = self._project(spikes)
        if not self.transposed:
            output = output.transpose(-1, -2)
        return output, None


class bidir_psnns4(psnns4):
    """Exploratory non-causal PND with independent forward/backward modes."""

    def __init__(self, *args, fusion_scale=2**-0.5, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_args = {
            key: kwargs[key] for key in ("dt_min", "dt_max", "lr") if key in kwargs
        }
        self.backward_kernel = MyKernel(self.h, N=self.n, **kernel_args)
        self.fusion_scale = float(fusion_scale)

    def membrane(self, u):
        forward_kernel, _ = self.kernel(L=u.size(-1))
        backward_kernel, _ = self.backward_kernel(L=u.size(-1))
        forward = _fft_convolution(u, forward_kernel)
        backward = _fft_convolution(u.flip(-1), backward_kernel).flip(-1)
        return self.fusion_scale * (forward + backward) + u * self.D.unsqueeze(-1)


class Snn_s4_linear(nn.Module):
    """Linear two-logit firing ablation without exponential probability mapping."""

    def __init__(
        self,
        d_model,
        tau=1e-3,
        d_state=128,
        dropout=0.0,
        transposed=True,
        **kernel_args,
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = d_model
        self.transposed = transposed
        self.tau = float(tau)
        self.D = nn.Parameter(torch.randn(d_model))
        self.kernel = MyKernel(d_model, N=d_state, **kernel_args)
        self.spike_prob_layer = nn.Linear(d_model, 2 * d_model)
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1), nn.GLU(dim=-2)
        )

    def _logits(self, u):
        kernel, _ = self.kernel(L=u.size(-1))
        membrane = _fft_convolution(u, kernel) + u * self.D.unsqueeze(-1)
        logits = self.spike_prob_layer(membrane.transpose(-1, -2))
        logits = logits.transpose(-1, -2).reshape(u.size(0), self.h, 2, u.size(-1))
        return logits.permute(0, 1, 3, 2)

    def forward(self, u, spike=True, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        logits = self._logits(u)
        spikes = F.gumbel_softmax(logits, tau=self.tau, hard=spike, dim=-1)[..., 0]
        output = self.output_linear(self.dropout(spikes))
        if not self.transposed:
            output = output.transpose(-1, -2)
        return output, None


class bsnns4_linear(Snn_s4_linear):
    """Linear-logit ablation with Bernoulli forward and probability-ST backward."""

    def forward(self, u, spike=True, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        probability = torch.softmax(self._logits(u), dim=-1)[..., 0]
        if spike:
            sampled = torch.bernoulli(probability)
            spikes = sampled - probability.detach() + probability
        else:
            spikes = probability
        output = self.output_linear(self.dropout(spikes))
        if not self.transposed:
            output = output.transpose(-1, -2)
        return output, None
