"""
Taken from 
Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data Fan et al. ICCV 2025
https://github.com/VankouF/MotionMillion-Codes/blob/main/models/modules.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


from einops import rearrange

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = True

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Patcher1D(torch.nn.Module):
    """A module to convert 1D signal tensors into patches using torch operations."""
    
    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B, C, L)
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # 1D padding
        x = F.pad(x, pad=(n - 2, n - 1), mode=mode).to(dtype)
        
        # 1Dwavelet transform
        xl = F.conv1d(x, hl, groups=g, stride=2)  # low-pass filter
        xh = F.conv1d(x, hh, groups=g, stride=2)  # high-pass filter

        out = torch.cat([xl, xh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (l p) -> b (c p) l",
            p=self.patch_size,
        ).contiguous()
        return x


class UnPatcher1D(torch.nn.Module):
    """A module to convert 1D patches back into signal tensors."""
    
    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # shape: (B, C*p, T)
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 2  # Divided into low frequency and high frequency parts
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xl, xh = torch.chunk(x.to(dtype), 2, dim=1)

        # Inverse transform
        yl = F.conv_transpose1d(xl, hl, groups=g, stride=2, padding=(n - 2))
        yh = F.conv_transpose1d(xh, hh, groups=g, stride=2, padding=(n - 2))
        y = yl + yh

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p) l -> b c (l p)",
            p=self.patch_size,
        )
        return x