# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_GDN_DECODE_CHAIN=<mode>: the GDN decode chain between the qkvzab in-projection and the out-projection,
collapsed step by step toward one program per layer (Lottolabs FusedM1Output, THIRD-PARTY@e7439ca, ported to TP=2
with the recurrent state in DRAM so that every batch bucket and the MTP verify keep working).

mode=direct (this module, step 1): the decode GDN ops read their inputs STRAIGHT FROM the fused projection output
qkvzab [1, S, W] (L1 interleaved TILE, W = qkvz + Nv + gap + Nv = 8248 per chip at TP=2) instead of from five slice
copies of it. Today, per GDN layer and per decode step (plain decode AND the MTP state-once verify),
TPGatedDeltaNet._project_qkvzab runs ttnn.slice five times -- qkv [0:C), z [C:C+Vd), the tile-aligned ab block,
a [0:Nv) and b [Nv+gap:2Nv+gap) of it -- five programs of ~0.9-1.7 us (rz3 wb profile, B=8 4k:
[source validation record], sequence_chip0: 1.71 1.36 1.09 0.94 1.05 us
between the qkvzab matmul and gdn_conv_gates), 240 programs per step. The image ops already take column windows
(Thatch Cloud "milestone 4", in vllm-tt:k2-mr-pk3r's _ttnn.so):
  gdn_decode_conv_gates(x=qkvzab, ..., a=qkvzab, b=qkvzab, channels=C, a_col=C+Vd, b_col=C+Vd+Nv+gap)
      x pages are read as bt*Wt + cc (cc < C/32: the qkv tiles), the conv states and taps are unchanged;
  gdn_decode_norm_gate(o, z=qkvzab, ..., z_col_offset=C)
      z tile (bt, C/32 + h*Vt + ct), the same page the z slice copied.
With a/b as column windows the stock reader gathers the gate tiles element by element on one RISC (1536 narrow
L1 stores per call); the arm's overlay reader bench/probes/gdn_decode_chain/conv_gates.gdc.cpp replaces that, for
tile-aligned windows only (QWEN36_GDN_AB_PAD makes a and b start on tiles 256/257), with one full-page read per
gate tile, as the slice path reads its sliced a/b. Same bytes for every valid element (rows < S, columns < Nv):
bit-exact by construction, gated by the warmup self-test below (logical outputs bitwise on every chip) and by
bench/probes/sim_gdn_decode_chain.py in ttsim. The padding columns >= Nv of beta/g (never read: the recurrence
takes one scalar per (row, head)) may differ.

What stays: the qkvzab matmul and its sharded -> L1-interleaved copy, conv_gates, the recurrence, norm_gate.
qkvzab lives until norm_gate (the z slice lived as long before; peak L1 is ~unchanged: qkv + z + a + b slices
vs one qkvzab).

Requires: QWEN_GDN_CONV_GATES=1, QWEN_GDN_PACKED_QKV=1, QWEN36_GDN_NORM_GATE=1, the fused qkvzab projection
(QWEN36 decode DRAM-sharded weights), and tile-aligned a/b columns (QWEN36_GDN_AB_PAD=1). Refuses otherwise.
Off (unset/0): byte-identical to today's path (nothing here runs).

MUSTFAIL (QWEN36_GDN_DECODE_CHAIN_MUSTFAIL=1): b is read from a's columns (b_col = a_col), so beta =
sigmoid(a): the warmup self-test REFUSES the boot.

Plan for the rest of the chain (not built here): docs/analysis/gdn-decode-chain-20261003.md.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os

try:
    import ttnn
except ImportError:  # host unit tests: the pure helpers need no ttnn
    ttnn = None

ENV = "QWEN36_GDN_DECODE_CHAIN"
ENV_MF = "QWEN36_GDN_DECODE_CHAIN_MUSTFAIL"
MODES = ("direct",)
BANNER = "QWEN36_GDN_DECODE_CHAIN engaged: mode=direct"
MF_BANNER = "QWEN36_GDN_DECODE_CHAIN_MUSTFAIL engaged"
SELFTEST_OK = "QWEN36_GDN_DECODE_CHAIN self-test exact"
SELFTEST_REFUSED = "QWEN36_GDN_DECODE_CHAIN self-test REFUSED"
TILE = 32
_GDC_LOGGED = set()  # name matches arm_switch MODULE_GUARD_RE: a served arm switch re-earns the banners


def mode(env=None):
    """The validated mode ('direct') or None when off. Raises on an unknown value."""
    env = os.environ if env is None else env
    v = env.get(ENV, "").strip()
    mf = env.get(ENV_MF, "").strip()
    if mf not in ("", "0", "1"):
        raise ValueError(f"{ENV_MF}={mf!r}: expected unset, 0 or 1")
    if v in ("", "0", "off"):
        if mf == "1":
            raise ValueError(f"{ENV_MF}=1 needs {ENV}=direct")
        return None
    if v not in MODES:
        raise ValueError(f"{ENV}={v!r}: expected unset, 0, off or one of {MODES}")
    return v


def mustfail(env=None):
    env = os.environ if env is None else env
    return env.get(ENV_MF, "").strip() == "1"


class Layout:
    """Column layout of the fused per-chip projection [qkv (C) | z (Vd) | a (Nv) | gap | b (Nv)]."""

    def __init__(self, C, Vd, Nv, gap):
        self.C, self.Vd, self.Nv, self.gap = int(C), int(Vd), int(Nv), int(gap)
        self.z_col = self.C
        self.a_col = self.C + self.Vd
        self.b_col = self.a_col + self.Nv + self.gap
        self.W = self.b_col + self.Nv

    def check(self):
        """Refusal reasons ([] = usable). Every window must start on a tile so each op reads whole pages."""
        bad = []
        for name, col in (("C", self.C), ("z_col", self.z_col), ("a_col", self.a_col), ("b_col", self.b_col)):
            if col % TILE:
                bad.append(f"{name}={col} not a multiple of {TILE}")
        if self.Nv > TILE:
            bad.append(f"Nv={self.Nv} > {TILE}: the gate windows span more than one tile")
        return bad

    def __repr__(self):
        return f"Layout(C={self.C}, z_col={self.z_col}, a_col={self.a_col}, b_col={self.b_col}, W={self.W})"


def layout_for(dn):
    """The Layout of TPGatedDeltaNet `dn`, refusing (RuntimeError) what direct mode does not cover."""
    gap = int(getattr(dn.args, "gdn_ab_gap", 0) or 0)
    lay = Layout(dn.qkv_dim_tp, dn.qkvz_dim_tp - dn.qkv_dim_tp, dn.Nv, gap)
    bad = lay.check()
    if not getattr(dn, "_fuse_ab", False):
        bad.append("no fused qkvzab projection (decode DRAM-sharded weights)")
    if bad:
        raise RuntimeError(f"{ENV}=direct refused: {lay}: " + "; ".join(bad) + " (QWEN36_GDN_AB_PAD=1 aligns b)")
    return lay


def check_env(env=None):
    """The ops direct mode replaces must be the ones the arm serves; refuse otherwise."""
    env = os.environ if env is None else env
    need = {"QWEN_GDN_CONV_GATES": "1", "QWEN_GDN_PACKED_QKV": "1", "QWEN36_GDN_NORM_GATE": "1"}
    bad = [f"{k}={v}" for k, v in need.items() if env.get(k, "0") != v]
    if bad:
        raise RuntimeError(f"{ENV}=direct needs {', '.join(bad)} (it rewires gdn_decode_conv_gates/norm_gate inputs)")


def selftest_widths(bmax, served=()):
    """Every x width the decode chain can see: the plain decode buckets (powers of two below bmax, plus bmax), the
    verify widths M = T * Bw <= 32 (T = K + 1 in 2..4) of each of those buckets, and the M = T * Bw of every served
    verify cell (T, Bw) (mtp_batch_host.served_verify_cells: the buckets' K_b + 1, up to 16)."""
    plain, w = set(), 1
    while w < bmax:
        plain.add(w)
        w *= 2
    plain.add(int(bmax))
    widths = set(plain)
    for bw in plain:
        for t in (2, 3, 4):
            if bw * t <= TILE:
                widths.add(bw * t)
    widths.update(int(t) * int(bw) for t, bw in served)
    return sorted(x for x in widths if x <= TILE)


def _served_cells(bmax):
    """The MTP buckets' (T, Bw) verify cells; [] when QWEN36_MTP_BATCH is off."""
    if _tt_public_getenv("QWEN36_MTP_BATCH", "0") != "1":
        return []
    from models.demos.blackhole.qwen36.tt import mtp_batch_host as mbh

    return mbh.served_verify_cells(int(bmax))


def banner(path, S, lay, logger=None):
    key = (path, int(S))
    if key in _GDC_LOGGED:
        return
    _GDC_LOGGED.add(key)
    msg = (f"{BANNER} ({path} S={S}): conv_gates/norm_gate read qkvzab [1,{S},{lay.W}] directly "
           f"(qkv cols [0,{lay.C}), z@{lay.z_col}, a@{lay.a_col}, b@{lay.b_col}); 5 slice programs/layer removed")
    if mustfail():
        msg += f"; {MF_BANNER}: b read from a's columns (b_col={lay.a_col}), the self-test MUST refuse"
    (logger.info if logger is not None else print)(msg)


# ------------------------------------------------------------------------------------------------ the two ops
def conv_gates(qkvzab, states, taps, dt_bias, neg_exp_A, lay, batch, memory_config, mf=None):
    """gdn_decode_conv_gates on the fused projection output: x = qkvzab (channels C), a/b = column windows."""
    mf = mustfail() if mf is None else mf
    return ttnn.transformer.gdn_decode_conv_gates(
        qkvzab,
        states,
        taps,
        qkvzab,
        qkvzab,
        dt_bias,
        neg_exp_A,
        batch=batch,
        memory_config=memory_config,
        channels=lay.C,
        a_col=lay.a_col,
        b_col=lay.a_col if mf else lay.b_col,
    )


def norm_gate(o, qkvzab, norm_w, Nv, lay, batch, memory_config):
    """gdn_decode_norm_gate with z = qkvzab at column C."""
    return ttnn.transformer.gdn_decode_norm_gate(
        o, qkvzab, norm_w, Nv, batch=batch, z_col_offset=lay.z_col, epsilon=1e-6, memory_config=memory_config
    )


# ------------------------------------------------------------------------------------------------ self-test
def _conv_multi():
    try:
        from models.demos.blackhole.qwen36.tt.gdn import conv_multi
    except ImportError:  # loaded by path (ttsim probes): the sibling module
        import importlib.util as _ilu
        from pathlib import Path

        spec = _ilu.spec_from_file_location("conv_multi", Path(__file__).resolve().parent / "conv_multi.py")
        conv_multi = _ilu.module_from_spec(spec)
        spec.loader.exec_module(conv_multi)
    return conv_multi


def _bits(t):
    import torch

    t = t.contiguous()
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _diff(a, b):
    """Differing elements between two host tensors (bitwise); numel when the shapes differ."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    return int((_bits(a) != _bits(b)).sum())


def selftest(dn, widths, seed=4242, windows=None):
    """Fail-closed comparison of direct mode against the served slice path, on the device `dn` runs on, at every
    width in `widths`, with layer `dn`'s DEVICE taps / dt_bias / neg_exp_A / norm_w (never host weights) and random
    activations and conv states (copies: no live buffer is touched).

    Reference = dn._split_qkvzab (the served slices) + gdn_decode_conv_gates + gdn_decode_norm_gate as forward_decode
    calls them; candidate = conv_gates()/norm_gate() above (MUSTFAIL honoured). Compared bitwise on every chip:
    conv [1,S,C], beta/g [1,S,Nv], gated [1,S,Vd] (logical elements) and all four conv states on every row (rows
    outside the batch included). Then the explicit must-fail (b_col = a_col) must change beta.
    windows: optional [(T, bw)] -- also checks conv_multi.build_windows(lives, qkvzab) == build_windows(lives, qkv).
    Returns {check: n_differing_elements, ..., "mustfail_beta": n}."""
    import torch

    lay = layout_for(dn)
    tw = dn.tw
    mesh = dn.mesh
    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(mesh, ttnn.MeshDevice)
    rep = ttnn.ReplicateTensorToMesh(mesh) if is_mesh else None
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    taps = [tw["conv_taps"][j] for j in range(dn.K)]
    dt = taps[0].dtype
    tdt = torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32
    Nv, Dv, C = dn.Nv, dn.Dv, lay.C
    g = torch.Generator().manual_seed(seed)

    def up(t, mc, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dt, layout=layout, device=mesh, mesh_mapper=rep, memory_config=mc)

    def back(t):
        return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] if is_mesh else [ttnn.to_torch(t)]

    def diff(xs, ys):
        return sum(_diff(x, y) for x, y in zip(back(xs), back(ys)))

    res = {}
    for S in widths:
        rows = max(int(dn.B), int(S))
        qz_t = torch.randn(1, S, lay.W, generator=g).to(tdt)
        st_t = [torch.randn(1, rows, C, generator=g).to(tdt) for _ in range(dn.K)]
        o_t = (torch.randn(S, 1, Nv, Dv, generator=g) * 0.5).to(tdt)
        keep = []
        try:
            qkvzab = up(qz_t, L1)
            o = up(o_t, DR, ttnn.ROW_MAJOR_LAYOUT)
            keep += [qkvzab, o]
            # reference: the served slices + the ops as forward_decode calls them
            st_r = [up(t, DR) for t in st_t]
            keep += st_r
            qkv, z, a, b = dn._split_qkvzab(qkvzab, S, L1, L1)
            keep += [qkv, z, a, b]
            ref = ttnn.transformer.gdn_decode_conv_gates(qkv, st_r, taps, a, b, tw["dt_bias"], tw["neg_exp_A"],
                                                         batch=S, memory_config=L1)
            keep += list(ref)
            gated_r = ttnn.transformer.gdn_decode_norm_gate(o, z, tw["norm_w"], Nv, batch=S, epsilon=1e-6,
                                                            memory_config=L1)
            keep.append(gated_r)
            # candidate: direct mode (served MUSTFAIL honoured)
            st_c = [up(t, DR) for t in st_t]
            keep += st_c
            cand = conv_gates(qkvzab, st_c, taps, tw["dt_bias"], tw["neg_exp_A"], lay, S, L1)
            keep += list(cand)
            gated_c = norm_gate(o, qkvzab, tw["norm_w"], Nv, lay, S, L1)
            keep.append(gated_c)
            for n, x, y in zip(("conv", "beta", "g"), ref, cand):
                res[f"{n}_S{S}"] = diff(x, y)
            res[f"gated_S{S}"] = diff(gated_r, gated_c)
            res[f"states_S{S}"] = sum(diff(x, y) for x, y in zip(st_r, st_c))
            # explicit must-fail: b read from a's columns must change beta
            st_m = [up(t, DR) for t in st_t]
            keep += st_m
            mfo = conv_gates(qkvzab, st_m, taps, tw["dt_bias"], tw["neg_exp_A"], lay, S, L1, mf=True)
            keep += list(mfo)
            res.setdefault("mustfail_beta", 0)
            res["mustfail_beta"] += diff(ref[1], mfo[1])
        finally:
            for t in keep:
                ttnn.deallocate(t)
    for T, bw in windows or ():
        conv_multi = _conv_multi()
        M = T * bw
        qz_t = torch.randn(1, M, lay.W, generator=g).to(tdt)
        lv_t = [torch.randn(1, int(dn.B), C, generator=g).to(tdt) for _ in range(dn.K)]
        keep = []
        try:
            qkvzab = up(qz_t, L1)
            lives = [up(t, DR) for t in lv_t]
            keep += [qkvzab] + lives
            qkv = dn._split_qkvzab(qkvzab, M, L1, L1)
            keep += list(qkv)
            w_ref = conv_multi.build_windows(lives, qkv[0], T, bw)
            w_dir = conv_multi.build_windows(lives, qkvzab, T, bw)
            keep += list(w_ref) + list(w_dir)
            # P_0 is only allocated (never read by the op); P_1..P_3 are the gathered windows
            res[f"windows_T{T}_B{bw}"] = sum(diff(x, y) for x, y in zip(w_ref[1:], w_dir[1:]))
        finally:
            for t in keep:
                ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return bool(res) and res.get("mustfail_beta", 0) > 0 and all(
        v == 0 for k, v in res.items() if k != "mustfail_beta")


_GDC_SELFTEST_LOGGED = set()  # {"ok"} once passed in this process; reset by a served arm switch (re-runs)


def warmup_selftest(model, logger):
    """Run once per process from warmup_model_decode, BEFORE any decode program is compiled or captured: the
    self-test on GDN layer 0 at every selftest_widths(max batch) and, with QWEN36_MTP_CONV_MULTI, the window build of
    every (T, bw) verify pair. Fails closed: raises -> the boot is refused (MUSTFAIL: expected)."""
    if mode() is None or "ok" in _GDC_SELFTEST_LOGGED:
        return
    dns = model._gdn_layers()
    if not dns:
        return
    dn = dns[0]
    dn._decode_chain()  # refuses an uncovered configuration
    served = _served_cells(int(dn.B))
    widths = selftest_widths(int(dn.B), served)
    pairs = []
    if _tt_public_getenv("QWEN36_MTP_CONV_MULTI", "0") == "1":
        pairs = [(t, bw) for bw in widths if bw <= int(dn.B) for t in (2, 3, 4) if t * bw <= TILE]
        pairs += [(int(t), int(bw)) for t, bw in served if (int(t), int(bw)) not in pairs]
    res = selftest(dn, widths, windows=pairs)
    if not selftest_ok(res):
        bad = {k: v for k, v in res.items() if (v != 0) != (k == "mustfail_beta")}
        raise RuntimeError(
            f"{SELFTEST_REFUSED}: {bad}" + (f" ({ENV_MF}=1: expected)" if mustfail() else "")
            + f"; widths {widths}, window pairs {pairs}"
        )
    _GDC_SELFTEST_LOGGED.add("ok")
    logger.info(
        f"{SELFTEST_OK}: conv/beta/g/gated/conv-states bitwise equal to the slice path on every chip at widths "
        f"{widths}; window builds {pairs}; must-fail beta differs in {res['mustfail_beta']} elements"
    )
