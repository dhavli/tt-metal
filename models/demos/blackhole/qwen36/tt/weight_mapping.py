# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Remap HuggingFace Qwen3.5-9B state dict to internal format.

Handles:
- Stripping 'model.language_model.' prefix
- Filtering out vision encoder and MTP weights
- Renaming combined in_proj_qkv → qkv_proj (DeltaNet layers; the op uses the fused weight)
- Splitting combined conv1d.weight into separate Q, K, V conv weights (DeltaNet layers)
- Renaming lm_head.weight → output.weight
- Renaming embed_tokens → tok_embeddings
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import collections.abc
import json
import os
from pathlib import Path
from typing import Dict

import torch

# Layer indices that use full (softmax) attention
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}

# DeltaNet QKV split dimensions (used to split the combined conv1d.weight into
# per-stream Q/K/V conv weights — the QKV projection itself stays combined).
# Q: num_key_heads(16) × key_head_dim(128) = 2048
# K: num_key_heads(16) × key_head_dim(128) = 2048
# (V = num_value_heads(32) × value_head_dim(128) = 4096 is the remaining slice)
LINEAR_Q_DIM = 2048
LINEAR_K_DIM = 2048


def remap_qwen36_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap HF Qwen3.5-9B state dict to internal format.

    Args:
        state_dict: Raw HuggingFace state dict loaded from safetensors.

    Returns:
        Remapped state dict with internal naming convention.
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Filter out vision encoder weights (check original key — no prefix stripping yet)
        if "visual" in key or key.startswith("model.visual"):
            continue
        # Filter out MTP (multi-token prediction) weights (original key)
        if key.startswith("mtp"):
            continue

        # Strip the language-model prefix. Two checkpoint sources produce different
        # prefixes for the same internal weights:
        #   - raw sharded safetensors:           model.language_model.X
        #   - AutoModelForCausalLM.from_pretrained (text-only Qwen3_5ForCausalLM): model.X
        # Strip whichever matches, longest first, so BOTH sources yield identical
        # internal keys. Top-level weights like lm_head.weight have no prefix and are
        # matched against the original `key` below.
        new_key = key
        for prefix in ("model.language_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        # Rename top-level weights
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
            continue
        if key == "lm_head.weight":
            remapped["output.weight"] = tensor
            continue
        # Final norm (model.language_model.norm.weight)
        if new_key == "norm.weight":
            remapped["norm.weight"] = tensor
            continue

        # Handle per-layer weights
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"layers.{layer_idx}"
            sub_key = ".".join(parts[2:])

            # DeltaNet layers: keep ONLY the combined QKV weight. The split q/k/v_proj
            # were dead — the op runs the fused QKV projection from the combined weight
            # (it only read the splits in a fallback reached when qkv_proj_weight is None,
            # which never happens for the 9B).
            if sub_key == "linear_attn.in_proj_qkv.weight":
                remapped[f"{layer_prefix}.linear_attn.qkv_proj.weight"] = tensor  # [8192, 4096]
                continue

            if sub_key == "linear_attn.conv1d.weight":
                conv = tensor  # [8192, 1, 4]
                q_conv = conv[:LINEAR_Q_DIM, :, :]
                k_conv = conv[LINEAR_Q_DIM : LINEAR_Q_DIM + LINEAR_K_DIM, :, :]
                v_conv = conv[LINEAR_Q_DIM + LINEAR_K_DIM :, :, :]
                remapped[f"{layer_prefix}.linear_attn.q_conv.weight"] = q_conv
                remapped[f"{layer_prefix}.linear_attn.k_conv.weight"] = k_conv
                remapped[f"{layer_prefix}.linear_attn.v_conv.weight"] = v_conv
                continue

            # All other keys pass through unchanged
            remapped[new_key] = tensor
            continue

        # Any remaining keys pass through
        remapped[new_key] = tensor

    return remapped


def is_fp8_checkpoint(model_path) -> bool:
    """True when the checkpoint dir holds block-wise FP8 safetensors.

    Detected by a ``*.weight_scale_inv`` entry in the safetensors index (the
    per-block dequant scales that accompany float8_e4m3fn weights). Such a
    checkpoint cannot be loaded via AutoModelForCausalLM here; use
    ``load_qwen36_state_dict_fp8`` instead.
    """
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (KeyError, ValueError, OSError):
        return False
    # bfpq: a bfp4-packed-v1 checkpoint (tools/bfp-quant/bfp4_pack.py) has no FP8 tensor
    # left, but it is the same lazy key space; LazyFP8StateDict unpacks its codes on access.
    return any(k.endswith(".weight_scale_inv") or k.endswith(_BFP4_CODES) for k in weight_map)


# bfp4-packed-v1: <mod>.weight_bfp4_codes / <mod>.weight_bfp4_exp, unpacked by bfp4_unpack.unpack to
# the exact bf16 grid values, so ttnn.as_tensor sees bit-identical input to the bf16 arm.
_BFP4_CODES = "_bfp4_codes"
_BFP4_EXP = "_bfp4_exp"


def _remap_fp8_key(key):
    """Internal key for a raw FP8-checkpoint key, or None to drop it (vision, MTP, scales). The
    same mapping the eager loader applied, factored out so the lazy loader cannot drift from it."""
    if key.endswith(".weight_scale_inv") or "visual" in key or key.startswith("mtp"):
        return None
    short = key
    for prefix in ("model.language_model.", "model."):
        if short.startswith(prefix):
            short = short[len(prefix) :]
            break
    if "embed_tokens" in short:
        return "tok_embeddings.weight"
    if key == "lm_head.weight" or short == "lm_head.weight":
        return "output.weight"
    return short


class LazyFP8StateDict(collections.abc.Mapping):
    """Qwen3.5 FP8 checkpoint as a read-on-access mapping: same keys, same tensors as the eager
    ``dict`` this replaced, but nothing is read or dequantized until a key is indexed, and nothing
    is retained afterwards.

    The eager loader dequantized every block-FP8 projection to bf16 up front -- ~54 GB of host RAM
    for the 27B -- before the first tensor went anywhere near a device. The model tree consumes the
    state dict one layer at a time and hands each tensor to ``ttnn.as_tensor`` (which tilizes and,
    on a cold cache, persists it), so the working set only ever needs to be one layer. Reads go
    through ``safetensors.safe_open`` (mmap; page cache, reclaimable), dequant happens per tensor.

    ``substate(prefix)`` filters on keys -- ``utils.substate.substate`` iterates ``items()`` and
    would otherwise dequantize the whole checkpoint once per layer. ``manifest()`` yields the
    shape/dtype view the weight-cache marker needs (meta tensors) plus real tensors for the
    host-consumed keys, again without touching the projections."""

    def __init__(self, model_path):
        from safetensors import safe_open

        model_path = Path(model_path)
        with open(model_path / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        self._model_path = model_path
        self._handles = {}
        self._safe_open = safe_open
        # internal key -> (file, raw key, (scale file, scale key) or None, shape, dtype after load)
        self._spec = {}
        by_file = {}
        for raw, fn in weight_map.items():
            by_file.setdefault(fn, []).append(raw)
        for fn, keys in sorted(by_file.items()):
            sf = self._handle(fn)
            present = set(sf.keys())
            meta = sf.metadata() or {}
            for raw in sorted(keys):
                if raw.endswith(_BFP4_EXP):
                    continue
                if raw.endswith(_BFP4_CODES):
                    name = raw[: -len(_BFP4_CODES)]
                    if raw not in present:
                        continue
                    desc = json.loads(meta["bfp4:" + name])
                    members = desc.get("members", [name])
                    out_rows, cols = desc["shape"]
                    for m in members:
                        internal = _remap_fp8_key(m)
                        if internal is None:
                            continue
                        shape = (out_rows // len(members), cols)
                        self._spec[internal] = (fn, raw, ("bfp4", name, desc, m), shape, torch.bfloat16)
                    continue
                internal = _remap_fp8_key(raw)
                if internal is None or raw not in present:
                    continue
                sl = sf.get_slice(raw)
                shape = tuple(sl.get_shape())
                st_dtype = sl.get_dtype()
                scale = raw + "_scale_inv"
                if st_dtype == "F8_E4M3":
                    dtype = torch.bfloat16  # dequant_fp8_block output (or .to(bf16) when no scale)
                    scale_key = (weight_map[scale], scale) if scale in weight_map else None
                else:
                    dtype = _SAFETENSORS_DTYPES.get(st_dtype, torch.bfloat16)
                    scale_key = None
                self._spec[internal] = (fn, raw, scale_key, shape, dtype)

    def _handle(self, fn):
        h = self._handles.get(fn)
        if h is None:
            h = self._safe_open(str(self._model_path / fn), framework="pt")
            self._handles[fn] = h
        return h

    def as_single_device(self):
        """Rewrite the key space for the num_devices == 1 model tree, in place.

        The lazy loaders keep GDN linear_attn.* RAW because the TP loaders shard
        the fused tensors themselves. The single-device modules want what
        remap_qwen36_state_dict produces instead: in_proj_qkv renamed to
        qkv_proj, and conv1d split into per-stream q/k/v_conv. Without this a
        (1,1) mesh dies in load_gdn_weights with "DeltaNet layer requires the
        combined qkv_proj weight" -- the TP path never reaches that loader, so
        goals 1-3 could not have found it.

        Laziness is preserved: the conv split is recorded as three slice specs
        and the slicing happens on read. conv1d is an unquantised .weight in
        both checkpoint formats, so the read needs no dequant branch. Returns
        self so callers can chain.
        """
        renamed = {}
        self._derived = {}
        for k, v in self._spec.items():
            if k.endswith(".linear_attn.in_proj_qkv.weight"):
                renamed[k[: -len("in_proj_qkv.weight")] + "qkv_proj.weight"] = v
                continue
            if k.endswith(".linear_attn.conv1d.weight"):
                # Q and K are each linear_num_key_heads x linear_key_head_dim,
                # which is 2048 for the 9B and the 27B alike; V is whatever
                # remains, so this split does not hardcode a model size.
                fn, raw, _scale, shape, dtype = v
                for name, lo, hi in (
                    ("q_conv", 0, LINEAR_Q_DIM),
                    ("k_conv", LINEAR_Q_DIM, LINEAR_Q_DIM + LINEAR_K_DIM),
                    ("v_conv", LINEAR_Q_DIM + LINEAR_K_DIM, None),
                ):
                    dk = k[: -len("conv1d.weight")] + name + ".weight"
                    self._derived[dk] = (fn, raw, lo, hi)
                    end = shape[0] if hi is None else hi
                    renamed[dk] = (fn, raw, None, (end - lo,) + tuple(shape[1:]), dtype)
                continue
            renamed[k] = v
        self._spec = renamed
        return self

    def _read_derived(self, key):
        """Slice of a raw checkpoint tensor, or None when the key is not derived."""
        src = getattr(self, "_derived", None)
        if not src or key not in src:
            return None
        fn, raw, lo, hi = src[key]
        t = self._handle(fn).get_tensor(raw)
        return t[lo:] if hi is None else t[lo:hi]

    def __getitem__(self, key):
        d = self._read_derived(key)
        if d is not None:
            return d
        fn, raw, scale, _shape, _dtype = self._spec[key]
        t = self._handle(fn).get_tensor(raw)
        if scale is not None and scale[0] == "bfp4":
            from models.demos.blackhole.qwen36.tt.bfp4_unpack import unpack_entry

            _tag, name, desc, member = scale
            if not getattr(self, "_bfp4_banner", False):
                self._bfp4_banner = True
                print(f"QWEN36_BFP4_PACKED engaged: unpacking bfp4-packed-v1 codes to bf16 ({key} {desc['shape']})", flush=True)
            return unpack_entry(t, self._handle(fn).get_tensor(name + _BFP4_EXP), desc, name)[member]
        if t.dtype == torch.float8_e4m3fn:
            if scale is not None:
                from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

                scale_fn, scale_key = scale
                return dequant_fp8_block(t, self._handle(scale_fn).get_tensor(scale_key))
            return t.to(torch.bfloat16)
        return t

    def __iter__(self):
        return iter(self._spec)

    def __len__(self):
        return len(self._spec)

    def __contains__(self, key):
        return key in self._spec

    def keys(self):
        return list(self._spec)

    def items(self):
        for k in self._spec:
            yield k, self[k]

    def get(self, key, default=None):
        return self[key] if key in self._spec else default

    def substate(self, prefix):
        p = prefix + "."
        return {k[len(p) :]: self[k] for k in self._spec if k.startswith(p)}

    def shape_dtype(self, key):
        _fn, _raw, _scale, shape, dtype = self._spec[key]
        return shape, dtype

    def manifest(self, is_host_weight=None):
        out = {}
        for k, (_fn, _raw, _scale, shape, dtype) in self._spec.items():
            if is_host_weight is not None and is_host_weight(k):
                out[k] = self[k]
            else:
                out[k] = torch.empty(shape, dtype=dtype, device="meta")
        return out


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def load_qwen36_state_dict_fp8(model_path, single_device=False) -> "collections.abc.Mapping[str, torch.Tensor]":
    """Load Qwen3.5 FP8 weights: block-wise dequant + minimal key remap, LAZILY.

    Produces the SAME internal key scheme as ``remap_qwen36_state_dict`` for the
    shared/simple weights (``layers.N.mlp.*``, ``layers.N.self_attn.*``,
    ``input_layernorm`` / ``post_attention_layernorm``, ``tok_embeddings``,
    ``norm``, ``output``) so ``layer.py``'s substate extraction is unchanged.

    The one deliberate difference vs the single-device remap: GDN ``linear_attn.*``
    projections are kept RAW (fused ``in_proj_qkv``, fused ``conv1d``, plus
    ``in_proj_z`` / ``in_proj_a`` / ``in_proj_b`` / ``out_proj`` / ``A_log`` /
    ``dt_bias`` / ``norm.weight``) -- NOT split or renamed -- so the tensor-parallel
    GDN weight-prep helpers (prepare_gdn_qkv / prepare_conv_taps) can reorder and
    shard them per device. The TP module loaders branch on this raw layout.

    Returns a ``LazyFP8StateDict``; set ``QWEN36_EAGER_STATE_DICT=1`` to materialize the
    old eager ``dict`` instead (A/B testing only -- it needs ~2 bytes of host RAM per parameter).
    """
    lazy = LazyFP8StateDict(model_path)
    if single_device:
        lazy.as_single_device()
    if _tt_public_getenv("QWEN36_EAGER_STATE_DICT") == "1":
        return dict(lazy.items())
    return lazy


# --- NVFP4 (compressed-tensors) -------------------------------------------
#
# The QUASAR QAT build of this model ships `nvfp4-pack-quantized`, which is a
# different format from the block-FP8 above in all three of its parts: the
# values are 4-bit E2M1 packed two per byte, the scales are per-group (16)
# float8_e4m3 rather than per-block, and there is an extra scalar
# weight_global_scale. `is_fp8_checkpoint` keys on `weight_scale_inv` and
# correctly rejects it, so this is a second loader rather than a widening of
# the first.
#
# Format determined by measurement, not by reading a spec -- see
# scripts/nvfp4-unpack-check.py in the tt-stack repo, which unpacks a Linear on
# CPU and PCCs it against the same tensor from the FP8 checkpoint. Nine
# projections score 0.994-0.996 with norm ratios 0.984-0.998. The nibble order
# in particular is determined: high-nibble-first scores PCC -0.0004.

# FP4 E2M1 magnitudes; bit 3 is the sign. Non-uniform spacing is the point.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def unpack_nvfp4(packed, scale_fp8, global_scale):
    """[out, in/2] uint8 -> [out, in] bfloat16.

    group_size is inferred from the scale shape rather than assumed, so a
    checkpoint quantised at another group size fails on the assert instead of
    silently mis-grouping.
    """
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    out, half = packed.shape
    cols = half * 2

    idx = torch.empty((out, cols), dtype=torch.long)
    idx[:, 0::2] = (packed & 0x0F).to(torch.long)  # low nibble is the FIRST element
    idx[:, 1::2] = (packed >> 4).to(torch.long)

    vals = lut[idx & 0x07] * torch.where((idx & 0x08) != 0, -1.0, 1.0)

    groups = scale_fp8.shape[1]
    assert cols % groups == 0, f"{cols} columns do not divide into {groups} scale groups"
    # global_scale DIVIDES: it folds the fp8 scale back into the weight's own
    # units. Getting this backwards leaves PCC at ~1.0 (it is a scalar) with
    # every weight wrong by a constant factor, which is why the check script
    # gates on the norm ratio and not on PCC alone.
    scale = scale_fp8.to(torch.float32) / global_scale.to(torch.float32)
    scale = scale.repeat_interleave(cols // groups, dim=1)
    return (vals * scale).to(torch.bfloat16)


def is_nvfp4_checkpoint(model_path) -> bool:
    """True when the checkpoint dir holds compressed-tensors NVFP4 safetensors.

    Detected by a ``*.weight_packed`` entry in the safetensors index. Mutually
    exclusive with ``is_fp8_checkpoint`` in practice; callers should test this
    one first only if a checkpoint ever carried both, which none does.
    """
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (KeyError, ValueError, OSError):
        return False
    return any(k.endswith(".weight_packed") for k in weight_map)


def _remap_nvfp4_key(key):
    """Internal key for a raw NVFP4-checkpoint key, or None to drop it.

    Drops the same vision/MTP keys the FP8 remap does, plus the three
    quantisation sidecars. ``input_global_scale`` is dropped DELIBERATELY: it
    is an activation scale, and we quantise activations by our own dtype policy
    rather than the checkpoint's. It is named here so it cannot fall through a
    silent else and be mistaken for a weight.
    """
    for sidecar in (".weight_scale", ".weight_global_scale", ".input_global_scale"):
        if key.endswith(sidecar):
            return None
    if "visual" in key or key.startswith("mtp"):
        return None
    if key.endswith(".weight_packed"):
        key = key[: -len(".weight_packed")] + ".weight"
    return _remap_fp8_key(key)


class LazyNVFP4StateDict(LazyFP8StateDict):
    """The NVFP4 checkpoint as a read-on-access mapping.

    Inherits every Mapping method from LazyFP8StateDict by keeping the same
    ``_spec`` 5-tuple shape; only how a spec entry is built and how it is read
    differ. The laziness contract is the load-bearing part and is preserved
    exactly: the eager loader this pattern replaced cost ~54 GB of host RAM on
    a 16 GB box, and unpacking 4-bit does not make that affordable.
    """

    def __init__(self, model_path):
        from safetensors import safe_open

        model_path = Path(model_path)
        with open(model_path / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        self._model_path = model_path
        self._handles = {}
        self._safe_open = safe_open
        self._spec = {}

        by_file = {}
        for raw, fn in weight_map.items():
            by_file.setdefault(fn, []).append(raw)
        for fn, keys in sorted(by_file.items()):
            sf = self._handle(fn)
            present = set(sf.keys())
            for raw in sorted(keys):
                internal = _remap_nvfp4_key(raw)
                if internal is None or raw not in present:
                    continue
                sl = sf.get_slice(raw)
                shape = tuple(sl.get_shape())
                if raw.endswith(".weight_packed"):
                    stem = raw[: -len(".weight_packed")]
                    scale_key = stem + ".weight_scale"
                    gscale_key = stem + ".weight_global_scale"
                    # A packed weight without its scales is not recoverable;
                    # fail here rather than serving whatever survives.
                    assert scale_key in weight_map, f"{raw} has no {scale_key}"
                    assert gscale_key in weight_map, f"{raw} has no {gscale_key}"
                    scale = (
                        (weight_map[scale_key], scale_key),
                        (weight_map[gscale_key], gscale_key),
                    )
                    # packed holds two values per byte along the input dim
                    shape = (shape[0], shape[1] * 2)
                    dtype = torch.bfloat16
                else:
                    scale = None
                    dtype = _SAFETENSORS_DTYPES.get(sl.get_dtype(), torch.bfloat16)
                self._spec[internal] = (fn, raw, scale, shape, dtype)

    def __getitem__(self, key):
        d = self._read_derived(key)
        if d is not None:
            return d
        fn, raw, scale, _shape, _dtype = self._spec[key]
        t = self._handle(fn).get_tensor(raw)
        if scale is None:
            return t
        (scale_fn, scale_key), (gscale_fn, gscale_key) = scale
        return unpack_nvfp4(
            t,
            self._handle(scale_fn).get_tensor(scale_key),
            self._handle(gscale_fn).get_tensor(gscale_key),
        )


def load_qwen36_state_dict_nvfp4(model_path, single_device=False) -> "collections.abc.Mapping[str, torch.Tensor]":
    """Load QUASAR NVFP4 weights lazily, in the same internal key scheme as the
    FP8 loader, so nothing downstream of load_state_dict can tell them apart."""
    lazy = LazyNVFP4StateDict(model_path)
    if single_device:
        lazy.as_single_device()
    if _tt_public_getenv("QWEN36_EAGER_STATE_DICT") == "1":
        return dict(lazy.items())
    return lazy
