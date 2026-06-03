# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import math
import torch
from ..utils.compile import torch_compile_lazy


@torch_compile_lazy
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offsetq: torch.Tensor,
    offsetk: torch.Tensor = None,
    max_period: float = 10_000,
    time_before_heads: bool = False
):
    """
    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]`.
        offsetq (int): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin.
        time_before_heads (bool):  if True, expected [B, T, H, D], else [B, H, T ,D]
        
    """

    if time_before_heads:
        B, T_q, H, D = q.shape
        T_k = k.shape[1]
    else:
        B, H, T_q, D = q.shape
        T_k = k.shape[2]
    
    # assert k.shape == q.shape
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))

    if offsetk is None:
        offsetk = offsetq
    
    ts_q = offsetq.float().view(-1, 1) + torch.arange(T_q, device=q.device, dtype=torch.float32)
    ts_k = offsetk.float().view(-1, 1) + torch.arange(T_k, device=k.device, dtype=torch.float32)
    if time_before_heads:
        ts_q = ts_q.view(B, -1, 1, 1)
        ts_k = ts_k.view(B, -1, 1, 1)
    else:
        ts_q = ts_q.view(B, 1, -1, 1)
        ts_k = ts_k.view(B, 1, -1, 1)

    dims_q = q.shape[:-1]
    dims_k = k.shape[:-1]
    q = q.view(*dims_q, D // 2, 2)
    k = k.view(*dims_k, D // 2, 2)

    # convention is `r` suffix is real part, `i` is imaginary.
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr_q = torch.cos(freqs * ts_q)
    roti_q = torch.sin(freqs * ts_q)
    qor = qr * rotr_q - qi * roti_q
    qoi = qr * roti_q + qi * rotr_q

    rotr_k = torch.cos(freqs * ts_k)
    roti_k = torch.sin(freqs * ts_k)
    kor = kr * rotr_k - ki * roti_k
    koi = kr * roti_k + ki * rotr_k

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(*dims_q, D), ko.view(*dims_k, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        max_period (float): Maximum period of the rotation frequencies.

    Update: this is modified to support different shapes of q and k.
    """

    def __init__(self, max_period: float = 10000.0):
        super().__init__()
        self.max_period = max_period

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offsetq: torch.Tensor,
        offsetk: torch.Tensor = None,
        time_before_heads: bool = False
    ):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(q, k, offsetq, offsetk, self.max_period, time_before_heads)
