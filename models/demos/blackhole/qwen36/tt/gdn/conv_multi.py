# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_CONV_MULTI=1: the MTP verify's GDN conv as ONE gdn_decode_conv_gates at batch M = T*Bw
(docs/MTP-BATCH.md section 14).

The sequential verify (TPGatedDeltaNet.forward_verify) runs, per candidate j, 3 slices (qkv/a/b rows j*Bw..),
one gdn_decode_conv_gates at width Bw (which shifts the live conv taps in place) and, for j < T-1, 4 ttnn.clone
snapshots of the taps. c140/c140b measured ~3.8 ms per extra candidate in the slices + conv_gates and ~0.8 ms in the
clones over 48 GDN layers: the cost is per launch, not per row.

The conv is an FIR, not a recursion. For row b let s_b = [st0_b, st1_b, st2_b, st3_b, x_{0,b}, ..., x_{T-1,b}]
(the live taps before the verify, then the T verify inputs). Candidate j's sequential call sees the pre-shift
window [s_b[j], s_b[j+1], s_b[j+2], s_b[j+3]] and x_j = s_b[j+4]. So with window tensors P_k ([1, M, C], row
u = j*Bw + b = s_b[j+k]) one conv_gates call at batch M computes every candidate's conv, beta and g at once, with
exactly the per-element inputs the sequential calls saw (the op is elementwise per row: same FPU/SFPU sequence),
and its in-place shift leaves P_k row u = s_b[j+k+1] = the live tap k after j+1 tokens, i.e. the sequential
path's snapshot j (j < T-1) or final live tap (j = T-1). The live taps are not touched by the verify; the
QWEN36_MTP_STATE_SLOT commit (state_slot.commit_conv_multi) writes the selected window rows into them.

Programs (all ttnn.generic_op with JIT dataflow kernels in kernels/conv_multi/, pure data movement, bit-exact by
construction: whole 16-element face rows are copied, no arithmetic, no format change):
  build_windows  P_1..P_3 from the live taps and qkv (P_0 is never read by the op, which drops the oldest tap;
                 its whole tiles are overwritten by the op's shift, so it is only allocated)
  split_rows     the T candidate row blocks of conv (width C) -> T tensors [1, Bw, C]; and of beta and g
                 (width Nv) -> 2T tensors [1, Bw, Nv], one program each, in place of 3T ttnn.slice
The recurrence still runs T times (its state chains), at width Bw on those blocks.

QWEN36_MTP_CONV_MULTI_SPLIT=slice keeps ttnn.slice for the per-candidate blocks (3T ops) instead of the two
gather programs. QWEN36_MTP_CONV_MULTI_MUSTFAIL=1 builds the LAST candidate's window one step too far
(s_b[j+k+1]): the served equivalence must catch it.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn

try:
    from models.demos.blackhole.qwen36.tt.gdn import state_slot as _ss
except ImportError:  # loaded by path (ttsim probes): the sibling module
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("state_slot", Path(__file__).resolve().parent / "state_slot.py")
    _ss = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ss)

ENV = "QWEN36_MTP_CONV_MULTI"
ENV_MF = "QWEN36_MTP_CONV_MULTI_MUSTFAIL"
ENV_SPLIT = "QWEN36_MTP_CONV_MULTI_SPLIT"
NTAP = 4
ZERO = None  # row-map entry: zero row

_KDIR = Path(__file__).resolve().parent / "kernels" / "conv_multi"
_SRC = {}
_CB_OUT, _CB_SCR = 0, 1


def enabled():
    return _tt_public_getenv(ENV, "0") == "1"


def mustfail():
    return _tt_public_getenv(ENV_MF, "0") == "1"


def split_mode():
    m = _tt_public_getenv(ENV_SPLIT, "gather")
    if m not in ("gather", "slice"):
        raise RuntimeError(f"{ENV_SPLIT}={m!r}: expected gather or slice")
    return m


def op_counts(T, split="gather"):
    """GDN conv + recurrence device ops per layer in one verify: (with CONV_MULTI, sequential). Sequential: per
    candidate 3 slices + conv_gates + recurrence, plus 4 clones for j < T-1. CONV_MULTI: build + conv_gates +
    (2 split programs | 3T slices) + T recurrences. The commit is 2 programs either way (not counted)."""
    seq = T * 5 + NTAP * (T - 1)
    if T == 1:
        return 2 + T, seq
    return (2 + (2 if split == "gather" else 3 * T) + T), seq


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _shape(t):
    return [int(t.shape[i]) for i in range(len(t.shape))]


def _interleaved(t, what):
    if t.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: {what} must be TILE, got {t.layout}")
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: {what} must be interleaved, got {t.memory_config()}")


def alloc(device, rows, width, dtype, mc):
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1, rows, width]), dtype, ttnn.TILE_LAYOUT, device, mc)


MAX_DST = 16  # destinations per row-gather program (rows_reader.cpp / rows_writer.cpp static_assert NDST <= 16)


def gather_rows(srcs, dsts, rowmap):
    """dsts[d] row r <- srcs[s] row sr for rowmap[d][r] = (s, sr), zero for ZERO / rows past the map. Every tensor
    TILE interleaved [1, R, W] with one dtype and one W; every row read or written < 32 (tile-row 0). Writes dsts in
    place; returns dsts[-1]."""
    nsrc, ndst = len(srcs), len(dsts)
    if ndst > MAX_DST and len(rowmap) == ndst:
        # the kernels take at most MAX_DST destinations (rows_reader.cpp static_assert): split_rows at T > 8 (2 sources
        # x T blocks) runs as several programs over consecutive destination groups. Pure copies: same bytes as one.
        out = None
        for d0 in range(0, ndst, MAX_DST):
            out = gather_rows(srcs, dsts[d0:d0 + MAX_DST], rowmap[d0:d0 + MAX_DST])
        return out
    if not (1 <= nsrc <= 32 and 1 <= ndst <= MAX_DST and len(rowmap) == ndst):
        raise RuntimeError(f"{ENV}: row gather {nsrc} sources / {ndst} destinations / {len(rowmap)} maps")
    dt = srcs[0].dtype
    W = _shape(dsts[0])[-1]
    for i, t in enumerate(list(srcs) + list(dsts)):
        _interleaved(t, f"row-gather tensor {i}")
        s = _shape(t)
        # A SOURCE may be wider than W (QWEN36_GDN_DECODE_CHAIN=direct passes the fused qkvzab [1, M, W' >= C] as
        # qkv): every row read is < 32, so page p < ceil(W/32) of tile-row 0 is tile column p in either tensor.
        wide_ok = i < nsrc and len(s) == 3 and s[-1] >= W
        if t.dtype != dt or len(s) != 3 or s[0] != 1 or not (s[-1] == W or wide_ok):
            raise RuntimeError(f"{ENV}: row gather needs one dtype and [1, R, {W}] ({t.dtype} {s})")
    for d, m in enumerate(rowmap):
        if len(m) > min(32, _shape(dsts[d])[1]):
            raise RuntimeError(f"{ENV}: map {d} has {len(m)} rows for a {_shape(dsts[d])[1]}-row destination")
        for e in m:
            if e is not ZERO and not (0 <= e[0] < nsrc and 0 <= e[1] < min(32, _shape(srcs[e[0]])[1])):
                raise RuntimeError(f"{ENV}: map entry {e} out of range")
    page = _ss._tile_bytes(dt)
    ct = (W + 31) // 32
    s_ct = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in srcs]
    classes = []
    for c in s_ct:
        if c not in classes:
            classes.append(c)
    if len(classes) > 2:
        raise RuntimeError(f"{ENV}: row gather sources span {len(classes)} placements (max 2)")
    cls_mask = sum(1 << i for i, c in enumerate(s_ct) if c == classes[-1] and len(classes) == 2)
    d_ct = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in dsts]
    if any(c != d_ct[0] for c in d_ct):
        raise RuntimeError(f"{ENV}: row gather destinations must share one placement")
    need, words = [], []
    for m in rowmap:
        ent = [0xFFFF] * 32
        nd = 0
        for r, e in enumerate(m):
            if e is not ZERO:
                ent[r] = (e[0] << 5) | e[1]
                nd |= 1 << e[0]
        need.append(nd)
        words += [ent[2 * i] | (ent[2 * i + 1] << 16) for i in range(16)]
    crs, work = _ss._cores(srcs[0].device(), ndst * ct)
    r_ct = [nsrc, ndst, ct, _CB_OUT, _CB_SCR, page] + classes[0] + classes[-1]
    w_ct = [ndst, ct, _CB_OUT, page] + d_ct[0]
    s_addr = [t.buffer_address() for t in srcs]
    d_addr = [t.buffer_address() for t in dsts]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, n in work:
        r_rt[core.x][core.y] = [start, n, cls_mask] + s_addr + need + words
        w_rt[core.x][core.y] = [start, n] + d_addr

    def k(name, ct_, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=crs, compile_time_args=ct_, runtime_args=rt, config=cfg)

    desc = ttnn.ProgramDescriptor(
        kernels=[
            k("rows_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            k("rows_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[],
        cbs=[_ss._cb(_CB_OUT, dt, page, 2, crs), _ss._cb(_CB_SCR, dt, page, 1, crs)],
    )
    return ttnn.generic_op(list(srcs) + list(dsts), desc)


def window_map(T, bw, k, mf=False):
    """Row map of window P_k over sources [st0, st1, st2, st3, qkv]: row u = j*bw + b <- s_b[j+k]
    (s_b[i] = st_i row b for i < 4, else qkv row (i-4)*bw + b). mf: the last candidate reads s_b[j+k+1]."""
    out = []
    for u in range(T * bw):
        j, b = divmod(u, bw)
        i = j + k + (1 if (mf and j == T - 1) else 0)
        out.append((i, b) if i < NTAP else (NTAP, (i - NTAP) * bw + b))
    return out


def build_windows(lives, qkv, T, bw, mf=False, mc=None):
    """The four [1, T*bw, C] window tensors for one batch-M conv_gates call (P_0 allocated only)."""
    if len(lives) != NTAP:
        raise RuntimeError(f"{ENV}: needs {NTAP} conv taps, got {len(lives)}")
    M, C = T * bw, _shape(lives[0])[-1]
    # qkv may be the wider fused projection output (QWEN36_GDN_DECODE_CHAIN=direct): its first C columns are qkv.
    max_m = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))
    if M > max_m or _shape(qkv)[1] != M or _shape(qkv)[-1] < C:
        raise RuntimeError(f"{ENV}: qkv {_shape(qkv)} vs M={M} C={C} (max rows {max_m})")
    if qkv.dtype != lives[0].dtype:
        raise RuntimeError(f"{ENV}: qkv dtype {qkv.dtype} != conv tap dtype {lives[0].dtype}")
    mc = mc or ttnn.DRAM_MEMORY_CONFIG
    dev = lives[0].device()
    wins = [alloc(dev, M, C, lives[0].dtype, mc) for _ in range(NTAP)]
    gather_rows(list(lives) + [qkv], wins[1:], [window_map(T, bw, k, mf) for k in range(1, NTAP)])
    return wins


def split_rows(srcs, T, bw, mc=None):
    """Per source [1, T*bw, W] -> T tensors [1, bw, W] (rows j*bw..), all sources one width, ONE program.
    Returns a list (per source) of T tensors."""
    dev = srcs[0].device()
    mc = mc or ttnn.L1_MEMORY_CONFIG
    W = _shape(srcs[0])[-1]
    outs = [[alloc(dev, bw, W, s.dtype, mc) for _ in range(T)] for s in srcs]
    dsts, maps = [], []
    for si, row in enumerate(outs):
        for j, t in enumerate(row):
            dsts.append(t)
            maps.append([(si, j * bw + b) for b in range(bw)])
    gather_rows(list(srcs), dsts, maps)
    return outs


# ------------------------------------------------------------------------------------------------ self-test
def _bits(t):
    import torch

    t = t.contiguous()
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _same(a, b):
    import torch

    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def selftest(lives, taps, dt_bias, neg_exp_A, T, bw, seed=0, conv_gates=None, check_commit=True):
    """Fail-closed comparison of the CONV_MULTI conv path against the sequential verify, on the device this runs
    on, at the shapes and dtypes of `lives` (the layer's live conv taps, read only: every tensor here is a copy).
    Random qkv / a / b (L1, as served) and random tap states; the sequential reference is forward_verify's loop
    op for op (3 slices + conv_gates at width bw + 4 clones per candidate), the candidate is build_windows + one
    conv_gates at batch M + split_rows. Bitwise: conv, beta and g of every candidate; after commit_conv (seq) and
    commit_conv_multi (multi) with one idx page holding depths 0..T-1, the full live taps (every row, the rows
    outside the bucket included). Must-fail: the shifted last window must change the last candidate's conv.
    conv_gates: the op (default ttnn.transformer.gdn_decode_conv_gates; ttsim probes pass a port).
    Returns {check: "exact" | "differs" | ...}."""
    import torch

    conv_gates = conv_gates or ttnn.transformer.gdn_decode_conv_gates
    dev = lives[0].device()
    mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    rep = ttnn.ReplicateTensorToMesh(dev) if mesh else None
    Bmax, C = _shape(lives[0])[1], _shape(lives[0])[2]
    Nv = _shape(dt_bias)[-1]
    M = T * bw
    dt = lives[0].dtype
    tdt = torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    g = torch.Generator().manual_seed(seed)

    def up(t, mc, layout=ttnn.TILE_LAYOUT, dtype=None):
        return ttnn.from_torch(t, dtype=dtype or dt, layout=layout, device=dev, mesh_mapper=rep, memory_config=mc)

    def back(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0] if mesh else t)

    live_t = [torch.randn(1, Bmax, C, generator=g).to(tdt) for _ in range(NTAP)]
    qkv_t = torch.randn(1, M, C, generator=g).to(tdt)
    a_t = torch.randn(1, M, Nv, generator=g).to(tdt)
    b_t = torch.randn(1, M, Nv, generator=g).to(tdt)
    tap_l = list(taps)
    res = {}
    keep = []
    try:
        # sequential reference (forward_verify's loop)
        seq_live = [up(t, DR) for t in live_t]
        qkv, a, b = up(qkv_t, L1), up(a_t, L1), up(b_t, L1)
        keep += seq_live + [qkv, a, b]
        seq_blocks, snaps = [], []
        for j in range(T):
            lo, hi = j * bw, (j + 1) * bw
            qj = ttnn.slice(qkv, (0, lo, 0), (1, hi, C), memory_config=L1)
            aj = ttnn.slice(a, (0, lo, 0), (1, hi, Nv), memory_config=L1)
            bj = ttnn.slice(b, (0, lo, 0), (1, hi, Nv), memory_config=L1)
            out = conv_gates(qj, seq_live, tap_l, aj, bj, dt_bias, neg_exp_A, batch=bw, memory_config=L1)
            for t in (qj, aj, bj):
                ttnn.deallocate(t)
            seq_blocks.append([back(t) for t in out])
            for t in out:
                ttnn.deallocate(t)
            if j < T - 1:
                snaps.append([ttnn.clone(cs) for cs in seq_live])
                keep += snaps[-1]

        def multi(mf):
            cm_live = [up(t, DR) for t in live_t]
            q2, a2, b2 = up(qkv_t, L1), up(a_t, L1), up(b_t, L1)
            wins = build_windows(cm_live, q2, T, bw, mf=mf)
            conv_all, beta_all, g_all = conv_gates(q2, wins, tap_l, a2, b2, dt_bias, neg_exp_A, batch=M, memory_config=L1)
            (convs,) = split_rows([conv_all], T, bw, mc=L1)
            betas, gs = split_rows([beta_all, g_all], T, bw, mc=L1)
            blocks = [[back(convs[j]), back(betas[j]), back(gs[j])] for j in range(T)]
            for t in [q2, a2, b2, conv_all, beta_all, g_all] + convs + betas + gs:
                ttnn.deallocate(t)
            return cm_live, wins, blocks

        cm_live, wins, cm_blocks = multi(False)
        keep += cm_live + wins
        for j in range(T):
            for n, x, y in zip(("conv", "beta", "g"), seq_blocks[j], cm_blocks[j]):
                res[f"{n}{j}"] = "exact" if _same(x, y) else "differs"
        if check_commit:
            pick = torch.randint(0, T, (bw,), generator=g).tolist()
            pick[0], pick[-1] = 0, T - 1
            page = [_ss.SKIP] * _ss.IDX_LEN
            page[:bw] = pick
            idx = up(torch.tensor(page, dtype=torch.int32).reshape(1, -1), DR, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
            keep.append(idx)
            _ss.commit_conv(seq_live, snaps, idx, bw)
            _ss.commit_conv_multi(cm_live, wins, idx, bw, T)
            res["commit_taps"] = "exact" if all(_same(back(x), back(y)) for x, y in zip(seq_live, cm_live)) else "differs"
        mf_live, mf_wins, mf_blocks = multi(True)
        keep += mf_live + mf_wins
        res["mustfail_last"] = "exact" if _same(seq_blocks[-1][0], mf_blocks[-1][0]) else "differs"
    finally:
        for t in keep:
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return bool(res) and res.get("mustfail_last") == "differs" and all(
        v == "exact" for k, v in res.items() if k != "mustfail_last")
