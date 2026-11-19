#!/usr/bin/env python3
"""bfp4-packed-v1: unpack tt-metal bfp4_b codes back to the exact bf16 grid values.

Format (written by ``bfp4_pack.py``). A packed ENTRY ``<name>`` of logical torch shape [out, in] is
two uint8 tensors in a safetensors shard:

  <name>_bfp4_codes   uint8 [in, out/2]   two 4-bit codes per byte, element 2j in the LOW nibble,
                                          2j+1 in the HIGH nibble
  <name>_bfp4_exp     uint8 [in, out/16]  shared exponent per block (biased fp32 exponent)

Logical layout is the ttnn one, M = W.T ([in, out]), with blocks of 16 CONSECUTIVE elements along
the last dim: exactly the group membership of the tt-metal host packer
(``blockfloat_common.cpp::pack_as_bfp_tiles``; tilization only permutes whole groups). A code is
``sign << 3 | m`` with m in 0..7, and the value is (-1)^sign * m * 2^(exp - 127 - 2). Every such
value is exactly representable in bf16, so unpacking is lossless and deterministic.

Each entry's JSON descriptor is in the shard metadata under ``bfp4:<name>``:

  {"shape": [out, in], "dtype": "bfloat16", "layout": "T16"}
      <name> is the checkpoint key itself (``...mlp.down_proj.weight``).
  {"shape": [2*R, in], ..., "layout": "T16", "members": [A, B], "split": "tp_ab", "tp": T}
      a JOINT entry for two keys A, B of R rows each, stored in the per-device fused order
      [A_0 | B_0 | A_1 | B_1 ...] (R/T rows per piece). GDN in_proj_a / in_proj_b are stored this
      way because production fuses them per device and, with 24 rows per device, a 16-block
      straddles a and b -- their snapped values are on the grid only under that joint grouping
      (so the joint form is specific to T, here 2).

The shard metadata also carries ``format = bfp4-packed-v1``.

Library use (the TT loader hook uses these):
    unpack(codes_u8, exp_u8, (out, in)) -> bf16 [out, in]
    unpack_entry(codes_u8, exp_u8, descriptor) -> {key: bf16 tensor}   # 1 key, or 2 for a joint entry

CLI (for non-TT users): dequantize a packed checkpoint dir to a plain bf16 checkpoint:
    python bfp4_unpack.py PACKED_DIR OUT_DIR
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import torch

FORMAT = "bfp4-packed-v1"
GROUP = 16
W = 3  # bfp4_b: 3 magnitude bits (hidden bit included) + 1 sign bit
CODES_SUFFIX = "_bfp4_codes"
EXP_SUFFIX = "_bfp4_exp"
META_PREFIX = "bfp4:"


def unpack(codes: torch.Tensor, exp: torch.Tensor, shape) -> torch.Tensor:
    """codes uint8 [in, out/2], exp uint8 [in, out/16] -> bf16 [out, in] (torch layout), exact."""
    out, inn = int(shape[0]), int(shape[1])
    if tuple(codes.shape) != (inn, out // 2) or tuple(exp.shape) != (inn, out // GROUP):
        raise ValueError(f"bfp4 shapes {tuple(codes.shape)}/{tuple(exp.shape)} do not match weight {shape}")
    c = codes.to(torch.int32)
    nib = torch.stack((c & 0xF, c >> 4), dim=-1).reshape(inn, out)  # element 2j low, 2j+1 high
    m = (nib & 0x7).to(torch.float32)
    m = torch.where((nib & 0x8) != 0, -m, m)
    # scale = 2^(e - 129); e in 0..255, so the exponent is in [-129, 126]: exact in fp32 (a
    # subnormal scale only occurs for all-zero blocks, where m is 0 anyway)
    e = exp.to(torch.int32).repeat_interleave(GROUP, dim=1) - 127 - (W - 1)
    v = torch.ldexp(m, e)
    v = torch.where((nib & 0x7) == 0, torch.zeros_like(v), v)  # zero code is +0 regardless of exponent
    return v.to(torch.bfloat16).t().contiguous()


def fuse_tp_ab(a: torch.Tensor, b: torch.Tensor, tp: int) -> torch.Tensor:
    n = a.shape[0] // tp
    return torch.cat([torch.cat([a[d * n:(d + 1) * n], b[d * n:(d + 1) * n]]) for d in range(tp)])


def split_tp_ab(f: torch.Tensor, tp: int):
    n = f.shape[0] // (2 * tp)
    a = torch.cat([f[d * 2 * n: d * 2 * n + n] for d in range(tp)])
    b = torch.cat([f[d * 2 * n + n: (d + 1) * 2 * n] for d in range(tp)])
    return a.contiguous(), b.contiguous()


def unpack_entry(codes, exp, desc: dict, name: str) -> dict:
    """One packed entry -> {checkpoint key: bf16 tensor}."""
    w = unpack(codes, exp, desc["shape"])
    if "members" not in desc:
        return {name: w}
    if desc.get("split") != "tp_ab":
        raise ValueError(f"{name}: unknown split {desc.get('split')!r}")
    a, b = split_tp_ab(w, int(desc["tp"]))
    return dict(zip(desc["members"], (a, b)))


def entries(meta: dict) -> dict:
    """name -> descriptor for every packed entry described in a shard's metadata."""
    return {k[len(META_PREFIX):]: json.loads(v) for k, v in (meta or {}).items() if k.startswith(META_PREFIX)}


def dequantize_dir(src: str, dst: str) -> dict:
    """Write a plain bf16 safetensors checkpoint from a bfp4-packed-v1 directory."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(dst, exist_ok=True)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    new_map, total = {}, 0
    for fn in files:
        f = safe_open(os.path.join(src, fn), framework="pt")
        meta = f.metadata() or {}
        if meta.get("format") not in (None, "pt", FORMAT):
            raise ValueError(f"{fn}: unknown format {meta.get('format')!r}")
        ents = entries(meta)
        out = {}
        for k in f.keys():
            if k.endswith(EXP_SUFFIX):
                continue
            if k.endswith(CODES_SUFFIX):
                name = k[: -len(CODES_SUFFIX)]
                out.update(unpack_entry(f.get_tensor(k), f.get_tensor(name + EXP_SUFFIX), ents[name], name))
            else:
                out[k] = f.get_tensor(k)
        for k, t in out.items():
            new_map[k] = fn
            total += t.numel() * t.element_size()
        save_file(out, os.path.join(dst, fn), metadata={"format": "pt"})
        print(f"{fn}: {len(out)} tensors ({len(ents)} packed entries unpacked)", flush=True)
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if name.endswith(".safetensors") or name == "model.safetensors.index.json" or not os.path.isfile(p):
            continue
        if name == "config.json":
            cfg = json.load(open(p))
            cfg.pop("quantization_config", None)
            cfg.get("text_config", {}).pop("quantization_config", None)
            json.dump(cfg, open(os.path.join(dst, name), "w"), indent=2)
        else:
            shutil.copy2(p, os.path.join(dst, name))
    return {"files": len(files), "tensors": len(new_map), "bytes": total}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(json.dumps(dequantize_dir(sys.argv[1], sys.argv[2])))
