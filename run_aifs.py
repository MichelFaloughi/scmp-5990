"""Run anemoi-inference (AIFS-Single 2.0) on a machine without CUDA / flash-attn.

Usage:  python run_aifs.py run.yaml [key=value overrides...]

Three compatibility shims, all needed on macOS / CPU:

1. Open Data plugin vs anemoi-transform 0.1.16.post2 naming mismatch
   ECMWF pins anemoi-transform==0.1.16.post2 (its `apply_mask` filter accepts `param`,
   which AIFS 2.0's inference.yaml relies on). The Open Data input plugin
   (anemoi-plugins-ecmwf-inference 0.2.1) expects attribute names from 0.1.17+.
   -> give the plugin class both spellings.

2. Checkpoint pickles `flash_attn.flash_attn_interface.flash_attn_func`
   torch.load fails with ModuleNotFoundError if flash_attn isn't importable.
   -> register a stub module so unpickling succeeds.

3. Attention on CPU
   anemoi-models' SDPA fallback materialises the full (40320 x 40320) attention matrix
   per head (~100 GB). AIFS uses sliding-window attention (|i-j| <= 1120), so
   -> swap in a banded implementation that processes query blocks against only the
      keys inside the window. Same maths, bounded memory.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

# ---- shim 1: plugin attribute names -------------------------------------------------
from anemoi.plugins.ecmwf.inference.opendata import geopotential_height as _gh

_gh.InferenceOrography.optional_inputs = {
    "orog": "gh", "z": "z",                  # anemoi-transform 0.1.16.post2 names
    "orography": "gh", "geopotential": "z",  # plugin names
}

# ---- shim 2: stub flash_attn so the pickle resolves ----------------------------------
if "flash_attn" not in sys.modules:
    try:
        import flash_attn  # noqa: F401  (real one, if present)
    except ImportError:
        _fa = types.ModuleType("flash_attn")
        _fa.__version__ = "0.0.0-stub"
        _fai = types.ModuleType("flash_attn.flash_attn_interface")

        def flash_attn_func(*a, **k):
            raise RuntimeError("flash_attn stub called; attention should have been swapped to BandedSDPA")

        _fai.flash_attn_func = flash_attn_func
        _fa.flash_attn_interface = _fai
        _fa.flash_attn_func = flash_attn_func
        sys.modules["flash_attn"] = _fa
        sys.modules["flash_attn.flash_attn_interface"] = _fai


# ---- shim 3: banded sliding-window attention -----------------------------------------
class BandedSDPAAttention(nn.Module):
    """Drop-in for anemoi-models' attention wrappers, computing only the |i-j| <= window band.

    Input/outputs are (batch, heads, seq, dim), matching what MultiHeadSelfAttention passes.
    """

    def __init__(self, block_size: int | None = None):
        super().__init__()
        self.block_size = block_size or int(os.environ.get("AIFS_ATTN_BLOCK", 2048))

    def forward(self, query, key, value, batch_size, causal=False, window_size=None,
                dropout_p=0.0, softcap=None, alibi_slopes=None):
        assert not causal, "causal attention not supported by BandedSDPAAttention"
        assert alibi_slopes is None, "alibi slopes not supported by BandedSDPAAttention"
        assert not softcap, "softcap not supported by BandedSDPAAttention"

        L = query.shape[-2]
        if window_size is None or window_size >= L:
            return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p)

        w = int(window_size)
        out = torch.empty_like(query)
        idx = torch.arange(L, device=query.device)
        for s in range(0, L, self.block_size):
            e = min(s + self.block_size, L)
            ks, ke = max(0, s - w), min(L, e + w)
            q = query[..., s:e, :]
            k = key[..., ks:ke, :]
            v = value[..., ks:ke, :]
            mask = (idx[s:e, None] - idx[None, ks:ke]).abs() <= w   # (e-s, ke-ks)
            out[..., s:e, :] = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        return out


def swap_attention(model: nn.Module) -> int:
    from anemoi.models.layers.attention import MultiHeadSelfAttention

    n = 0
    for mod in model.modules():
        if isinstance(mod, MultiHeadSelfAttention):
            mod.attention_implementation = "banded_sdpa"
            mod.attention = BandedSDPAAttention()
            n += 1
    return n


_orig_torch_load = torch.load


def _patched_load(*args, **kwargs):
    obj = _orig_torch_load(*args, **kwargs)
    if isinstance(obj, nn.Module):
        n = swap_attention(obj)
        if n:
            print(f"[run_aifs] swapped {n} attention layers to BandedSDPAAttention", file=sys.stderr)
    return obj


torch.load = _patched_load

# ---- run --------------------------------------------------------------------------
if __name__ == "__main__":
    from anemoi.inference.__main__ import main  # noqa: E402

    sys.argv = ["anemoi-inference", "run", *sys.argv[1:]]
    main()
