# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_COMMIT_PACK=1|rec|conv: the MTP GDN commit of ALL GDN layers in one program per state kind
(docs/MTP-BATCH.md section 19).

Why (measured): under QWEN36_MTP_GDN_STATE_ONCE + QWEN36_MTP_GDN_SO_T0 the commit phase of one speculative step costs
3.74 / 3.29 / 3.93 / 4.98 ms at Bw 1 / 2 / 4 / 8 (synced split, tm-so1-t0,
[source validation record]
20261002T080519Z.jsonl): nearly flat in Bw, so at B=1 it is ~8% of the ~44 ms cycle while its bytes are tiny
(one 0.79 MB bf16 row state per layer). The pure-move cm commit was already 1.72 ms at Bw=1 (tm-cm,
...-warm-kb-tm-cm-20261001T140716Z.jsonl). Per step it is 2 x 48 programs (state_once.commit + state_slot.
commit_conv_multi per GDN layer), and each layer's recompute runs on only Bw x Nv head instances (24 at Bw=1, on 24 of
the grid's cores), the 48 layers one after another.

What: the layers' commits are independent (disjoint rec_state / conv taps / cin / window buffers; one shared slot page),
so one program takes the L x Bw x Nv recompute instances (kernels/state_once/so_pack_*.cpp, the unchanged
so_compute.cpp MODE 1) spread over the whole grid, and one program takes the L x U conv-tap units
(kernels/state_slot/conv_multi_pack_*.cpp). Per instance / unit the kernels do exactly what the per-layer kernels do
(same reads, same op sequence, same writes), so the result is bit-identical by construction; only the instance -> core
map changes. 96 programs -> 2 per step. Unset: nothing here runs, the per-layer path is untouched (its kernel sources
are not edited).

Moved cost (ESTIMATE, not measured): the per-layer program overhead goes; the recompute that ran on 24 cores per layer
runs on the whole grid. Whether the dispatch or the per-core serial recompute dominates the flat ~3.3 ms is what the
card shows (tm twin: the `commit` phase).

Values: 1 = both programs packed; rec = only the recurrent-state commit; conv = only the conv-tap commit (attribution).
Requires QWEN36_MTP_STATE_SLOT=1, QWEN36_MTP_CONV_MULTI=1 and QWEN36_MTP_GDN_STATE_ONCE=1; refuses otherwise.

MUSTFAIL (QWEN36_MTP_COMMIT_PACK_MUSTFAIL=1): the packed programs hand the LAST layer the second-to-last layer's cin
sticks (rec) and window tensors (conv): a wrong committed state. The warmup self-test (packed vs per-layer, bitwise)
then differs and REFUSES the boot.

The flag is read per call (runtime class): an arm switch re-runs compile() (self-test) and re-captures the C traces.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import importlib
import importlib.util
import os
import sys
from pathlib import Path

try:
    import ttnn
except ImportError:  # host unit tests: the pure helpers below need no ttnn
    ttnn = None

ENV = "QWEN36_MTP_COMMIT_PACK"
ENV_MF = "QWEN36_MTP_COMMIT_PACK_MUSTFAIL"
BANNER = "QWEN36_MTP_COMMIT_PACK engaged"
SELFTEST_BANNER = "QWEN36_MTP_COMMIT_PACK self-test exact"
MF_BANNER = "QWEN36_MTP_COMMIT_PACK_MUSTFAIL engaged"
REQUIRES = ("QWEN36_MTP_STATE_SLOT", "QWEN36_MTP_CONV_MULTI", "QWEN36_MTP_GDN_STATE_ONCE")
KINDS = ("rec", "conv")
MAX_RT_ARGS = 256  # per-core runtime words the wrapper allows (tt-metal's limit is higher; refuse well before it)
NTAP = 4

_KDIR = Path(__file__).resolve().parent / "kernels"
_SRC = {}
# per-process "log once" guards (the arm switch resets *_LOGGED sets)
_COMMIT_PACK_LOGGED = set()


# ------------------------------------------------------------------------------------------------ pure helpers
def mode(env=None):
    """The packed kinds as a frozenset ({} when unset). Raises on a bad value or missing prerequisites."""
    env = os.environ if env is None else env
    v = env.get(ENV, "0").strip()
    mf = env.get(ENV_MF, "0").strip()
    if mf not in ("0", "1", ""):
        raise ValueError(f"{ENV_MF}={mf!r}: 0 | 1")
    if v in ("0", ""):
        if mf == "1":
            raise RuntimeError(f"{ENV_MF}=1 refused: it needs {ENV} set")
        return frozenset()
    if v == "1":
        kinds = frozenset(KINDS)
    elif v in KINDS:
        kinds = frozenset((v,))
    else:
        raise ValueError(f"{ENV}={v!r}: 0 | 1 | rec | conv")
    missing = [k for k in REQUIRES if env.get(k, "0") != "1"]
    if missing:
        raise RuntimeError(f"{ENV}={v} refused: it needs {'=1, '.join(missing)}=1 (it packs the STATE_ONCE slot commit)")
    return kinds


def mustfail(env=None):
    env = os.environ if env is None else env
    return env.get(ENV_MF, "0").strip() == "1"


def split_work(total, ncores):
    """[(core_index, start, count)] over `total` units, ceil-balanced like the per-layer factories
    (state_once._grid / state_slot._cores): per = ceil(total / ncores), active = ceil(total / per)."""
    if total <= 0:
        return [], 0
    per = (total + ncores - 1) // ncores
    active = (total + per - 1) // per
    return [(c, c * per, min(per, total - c * per)) for c in range(active)], per


def layer_window(start, count, per_layer):
    """The layers a contiguous unit range [start, start + count) touches: (l_first, nl)."""
    l_first = start // per_layer
    l_last = (start + count - 1) // per_layer
    return l_first, l_last - l_first + 1


def rec_args(start, count, BH, addrs_state, addrs_cin, idx_addr):
    """(reader, writer) runtime words for one core of the packed recurrent-state commit."""
    l0, nl = layer_window(start, count, BH)
    r = [start, count, idx_addr, BH, l0, nl] + [addrs_state[l0 + i] for i in range(nl)] + [addrs_cin[l0 + i] for i in range(nl)]
    w = [start, count, BH, l0, nl] + [addrs_state[l0 + i] for i in range(nl)]
    return r, w


def conv_args(start, count, U, addrs_live, addrs_win, idx_addr):
    """(reader, writer) runtime words for one core of the packed conv-tap commit. addrs_live / addrs_win: per layer a
    list of NTAP addresses."""
    l0, nl = layer_window(start, count, U)
    r = [idx_addr, start, count, U, l0, nl]
    for i in range(nl):
        r += list(addrs_live[l0 + i]) + list(addrs_win[l0 + i])
    w = [start, count, U, l0, nl]
    for i in range(nl):
        w += list(addrs_live[l0 + i])
    return r, w


def mustfail_addrs(addrs):
    """The must-fail address list: the last layer gets the second-to-last layer's inputs (cin sticks or windows)."""
    if len(addrs) < 2:
        raise RuntimeError(f"{ENV_MF}: needs at least 2 layers")
    out = list(addrs)
    out[-1] = out[-2]
    return out


def _sib(name):
    """A sibling module (state_once / state_slot): the package import when served, the bare module a ttsim probe
    loaded (or a load by path) otherwise."""
    try:
        return importlib.import_module(f"models.demos.blackhole.qwen36.tt.gdn.{name}")
    except ImportError:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod


# ------------------------------------------------------------------------------------------------ device programs
def _src(rel):
    if rel not in _SRC:
        _SRC[rel] = (_KDIR / rel).read_text()
    return _SRC[rel]


def _grid_colmajor(device, n_work, cap=None):
    """state_once._grid's map (column-major, ceil-balanced) over at most `cap` cores (None: the whole grid; the probes
    cap it to force core ranges that straddle layers)."""
    g = device.compute_with_storage_grid_size()
    gx, gy = g.x, g.y
    ncores = gx * gy if cap is None else max(1, min(cap, gx * gy))
    work0, per = split_work(n_work, ncores)
    work = [(ttnn.CoreCoord(c // gy, c % gy), start, n) for c, start, n in work0]
    active = len(work)
    full, rem = divmod(active, gy)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full - 1, gy - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(full, 0), ttnn.CoreCoord(full, rem - 1)))
    return ttnn.CoreRangeSet(ranges), work, per


def _grid_rowmajor(device, n_work, cap=None):
    """state_slot._cores's map (row-wise, ceil-balanced) over at most `cap` cores."""
    g = device.compute_with_storage_grid_size()
    ncores = g.x * g.y if cap is None else max(1, min(cap, g.x * g.y))
    work0, _per = split_work(n_work, ncores)
    work = [(ttnn.CoreCoord(c % g.x, c // g.x), start, n) for c, start, n in work0]
    active = len(work)
    full, rem = divmod(active, g.x)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges), work


def _ct(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _same_accessor(ts, what):
    ref = _ct(ts[0])
    for i, t in enumerate(ts[1:], 1):
        if _ct(t) != ref or t.dtype != ts[0].dtype or list(t.shape) != list(ts[0].shape) or t.layout != ts[0].layout:
            raise RuntimeError(f"{ENV}: {what} of layer {i} differs from layer 0 (accessor / dtype / shape / layout)")


def _check_rt(words, what):
    if len(words) > MAX_RT_ARGS:
        raise RuntimeError(f"{ENV}: {what} needs {len(words)} runtime words on one core (> {MAX_RT_ARGS})")


def _kernel(src, crs, ct, rt, cfg, prefix=""):
    return ttnn.KernelDescriptor(
        kernel_source=prefix + src,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=ct,
        runtime_args=rt,
        config=cfg,
    )


def commit_rec_packed(states, cins, idx, Nk, Nv, Dk, Dv, scale, T, Bw, mf=None, fid=None, t0=None, pack_mf=None,
                      max_cores=None, row0=None, row0_mf=None, fuse=None, fuse_mf=None, iosrc=None, iosrc_mf=None):
    """In place, for every layer l: what state_once.commit(states[l], cins[l], idx, ...) does, in ONE program.
    Same validation per layer as the per-layer wrapper; the accessor args of every state (and every cin) must match.
    row0 / row0_mf: QWEN36_MTP_GDN_SO_SFPU_ROW0 and its must-fail (default: the env), as the per-layer commit."""
    so = _sib("state_once")
    row0 = so.row0_enabled() if row0 is None else row0
    row0_mf = (so.row0_mustfail() if row0_mf is None else row0_mf) and row0
    fuse = so._norm_fuse(so.fuse_parts() if fuse is None else fuse)  # QWEN36_MTP_GDN_SO_FUSE, as the per-layer commit
    fuse_mf = bool((so.fuse_mustfail() if fuse_mf is None else fuse_mf) and fuse)
    iosrc = bool(so.iosrc_enabled() if iosrc is None else iosrc)  # QWEN36_MTP_GDN_SO_IOSRC, as the per-layer commit
    iosrc_mf = bool((so.iosrc_mustfail() if iosrc_mf is None else iosrc_mf) and iosrc)

    L = len(states)
    if L != len(cins) or L < 1:
        raise RuntimeError(f"{ENV}: {L} states vs {len(cins)} cin tensors")
    mf = so.mustfail() if mf is None else mf
    t0 = so.t0_enabled() if t0 is None else t0
    pack_mf = mustfail() if pack_mf is None else pack_mf
    dt = states[0].dtype
    W = so.cin_width(Dk, Dv)
    for l, (st, cin) in enumerate(zip(states, cins)):
        mc = st.memory_config()
        if mc.buffer_type != ttnn.BufferType.DRAM:
            raise RuntimeError(f"{ENV}: layer {l} state must be DRAM interleaved, got {mc}")
        so._check_tile(st, f"layer {l} state", dt)
        max_m = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))
        if so._shape(st)[0] < Bw or so._shape(st)[1:] != [Nv, Dk, Dv] or T * Bw > max_m or not 1 <= T <= so.T_MAX:
            raise RuntimeError(f"{ENV}: layer {l} state {so._shape(st)} vs T={T} Bw={Bw} Nv={Nv} Dk={Dk} Dv={Dv} (max rows {max_m})")
        if cin.layout != ttnn.ROW_MAJOR_LAYOUT or cin.dtype != dt or so._shape(cin) != [T * Bw, 1, Nv, W]:
            raise RuntimeError(f"{ENV}: layer {l} cin {cin.layout} {cin.dtype} {so._shape(cin)} != {[T * Bw, 1, Nv, W]}")
        if cin.memory_config().buffer_type != ttnn.BufferType.DRAM:
            raise RuntimeError(f"{ENV}: layer {l} cin must be DRAM")
    if t0 and T < 2:
        raise RuntimeError(f"{so.ENV_T0}: T={T}: needs T >= 2")
    _same_accessor(states, "rec_state")
    _same_accessor(cins, "cin")
    if idx.layout != ttnn.ROW_MAJOR_LAYOUT or idx.dtype != ttnn.uint32 or so._shape(idx)[-1] != so.IDX_LEN:
        raise RuntimeError(f"{ENV}: idx must be ROW_MAJOR uint32 [1, {so.IDX_LEN}]")
    try:
        ipage = int(idx.buffer_aligned_page_size())
    except Exception:
        ipage = so.IDX_LEN * 4
    io = so._tile_bytes(dt)
    if ipage > io:
        raise RuntimeError(f"{ENV}: idx page {ipage} B does not fit the scratch tile ({io} B)")
    desc_fid, fid_define, fid_why = so.fidelity() if fid is None else fid
    cin_page = W * (io // 1024)
    Kt, Vt = Dk // 32, Dv // 32
    BH = Bw * Nv
    total = L * BH
    dev = states[0].device()
    crs, work, per = _grid_colmajor(dev, total, max_cores)  # state_once._grid's map, over L*BH instances
    plan, cb_total = so.cb_plan(dt, Dk, Dv, fuse)
    fmt = {"io": dt, "f32": ttnn.float32, "u32": ttnn.uint32}
    cbs = [
        ttnn.CBDescriptor(
            total_size=nbytes,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[kind], page_size=page)],
        )
        for i, nbytes, page, kind in plan
    ]
    import struct

    eps_bits = struct.unpack("<I", struct.pack("<f", 1e-6))[0]
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale if scale is not None else Dk**-0.5)))[0]
    s_addr = [st.buffer_address() for st in states]
    c_addr = [c.buffer_address() for c in cins]
    if pack_mf:
        c_addr = mustfail_addrs(c_addr)
    r_ct = [Kt, Vt, Nv, T, Bw, 1 if mf else 0, ipage, cin_page, 1 if t0 else 0] + _ct(states[0]) + _ct(idx) + _ct(cins[0])
    w_ct = [Kt, Vt] + _ct(states[0])
    c_ct = [Kt, Vt, eps_bits, scale_bits, T, 1, 1 if t0 else 0]  # so_compute.cpp MODE 1, unchanged
    r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, n in work:
        r, w = rec_args(start, n, BH, s_addr, c_addr, idx.buffer_address())
        _check_rt(r, "rec reader")
        r_rt[core.x][core.y] = r
        w_rt[core.x][core.y] = w
        c_rt[core.x][core.y] = [n]
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = desc_fid
    ccfg.fp32_dest_acc_en = True
    ccfg.math_approx_mode = False
    desc = ttnn.ProgramDescriptor(
        kernels=[
            _kernel(_src("state_once/so_pack_reader.cpp"), crs, r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            _kernel(_src("state_once/so_pack_writer.cpp"), crs, w_ct, w_rt, ttnn.WriterConfigDescriptor()),
            _kernel(so._src("so_compute.cpp"), crs, c_ct, c_rt, ccfg,
                    prefix=so.compute_prefix(fid_define, row0, row0_mf, fuse, fuse_mf, iosrc, iosrc_mf)),
        ],
        semaphores=[],
        cbs=cbs,
    )
    ttnn.generic_op([idx] + list(cins) + list(states), desc)
    return {"layers": L, "instances": total, "cores": len(work), "per_core": per, "cb_kib": cb_total / 1024,
            "fid": fid_why, "t0": bool(t0), "pack_mf": bool(pack_mf), "row0": bool(row0), "fuse": ",".join(fuse)}


def commit_conv_packed(lives_list, wins_list, idx, Bw, T, pack_mf=None, max_cores=None):
    """In place, for every layer l: what state_slot.commit_conv_multi(lives_list[l], wins_list[l], idx, Bw, T) does, in
    ONE program."""
    ss = _sib("state_slot")

    L = len(lives_list)
    if L != len(wins_list) or L < 1:
        raise RuntimeError(f"{ENV}: {L} tap sets vs {len(wins_list)} window sets")
    pack_mf = mustfail() if pack_mf is None else pack_mf
    lives0 = lives_list[0]
    shp = ss._shape(lives0[0])
    M = T * Bw
    for l, (lives, wins) in enumerate(zip(lives_list, wins_list)):
        if len(lives) != NTAP or len(wins) != NTAP:
            raise RuntimeError(f"{ENV}: layer {l}: needs 4 live taps and 4 windows ({len(lives)}, {len(wins)})")
        for i, t in enumerate(lives):
            ss._check(t, f"layer {l} conv tap {i}")
            if ss._dt(t) != ss._dt(lives0[0]) or ss._shape(t) != shp:
                raise RuntimeError(f"{ENV}: layer {l} conv tap {i} {ss._dt(t)} {ss._shape(t)} != {ss._dt(lives0[0])} {shp}")
        for i, t in enumerate(wins):
            ss._check(t, f"layer {l} conv window {i}")
            if ss._dt(t) != ss._dt(lives0[0]) or ss._shape(t) != [1, M, shp[2]]:
                raise RuntimeError(f"{ENV}: layer {l} conv window {i} {ss._shape(t)} != {[1, M, shp[2]]}")
    max_m = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))
    assert len(shp) == 3 and shp[0] == 1 and Bw <= shp[1] and shp[2] % 32 == 0 and 1 <= T and M <= max_m
    _same_accessor([t for lv in lives_list for t in lv], "conv tap")
    _same_accessor([t for w in wins_list for t in w], "conv window")
    page = ss._tile_bytes(ss._dt(lives0[0]))
    ct = shp[2] // 32
    bt = (shp[1] + 31) // 32
    U = bt * ct
    ipage = ss._idx_page(idx)
    crs, work = _grid_rowmajor(lives0[0].device(), L * U, max_cores)
    lv_ct = _ct(lives0[0])
    p_ct = _ct(wins_list[0][0])
    ix_ct = _ct(idx)
    r_ct = [T, Bw, ct, ss._CB_DATA, ss._CB_SCR, ss._CB_IDX, page, ipage] + lv_ct + p_ct + ix_ct
    w_ct = [ct, ss._CB_DATA, ss._CB_IDX, page] + lv_ct
    live_a = [[t.buffer_address() for t in lv] for lv in lives_list]
    win_a = [[t.buffer_address() for t in w] for w in wins_list]
    if pack_mf:
        win_a = mustfail_addrs(win_a)
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, n in work:
        r, w = conv_args(start, n, U, live_a, win_a, idx.buffer_address())
        _check_rt(r, "conv reader")
        r_rt[core.x][core.y] = r
        w_rt[core.x][core.y] = w
    desc = ttnn.ProgramDescriptor(
        kernels=[
            _kernel(_src("state_slot/conv_multi_pack_reader.cpp"), crs, r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            _kernel(_src("state_slot/conv_multi_pack_writer.cpp"), crs, w_ct, w_rt, ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[],
        cbs=[
            ss._cb(ss._CB_DATA, ss._dt(lives0[0]), page, 2 * NTAP, crs),
            ss._cb(ss._CB_IDX, ttnn.uint32, ipage, 1, crs),
            ss._cb(ss._CB_SCR, ss._dt(lives0[0]), page, 1, crs),
        ],
    )
    ios = [idx] + [t for w in wins_list for t in w] + [t for lv in lives_list for t in lv]
    ttnn.generic_op(ios, desc)
    return {"layers": L, "units": L * U, "cores": len(work), "pack_mf": bool(pack_mf)}


# ------------------------------------------------------------------------------------------------ served entry
def commit_all(dns, Bw, idx, kinds, release=False, logger=None):
    """The MTP commit of every GDN layer `dns` (TPGatedDeltaNet) for bucket Bw from their _mtp_verify[Bw] entries
    (STATE_ONCE form only), with the kinds in `kinds` packed and the rest per layer as today. release: free the verify
    leftovers (cin, windows) as mtp_commit_slot does in the eager compile run."""
    so, ss = _sib("state_once"), _sib("state_slot")

    ents = []
    for dn in dns:
        hists, wins = dn._mtp_verify[Bw]
        if not isinstance(hists, tuple):
            raise RuntimeError(f"{ENV}: layer verify state is not the STATE_ONCE form (QWEN36_MTP_GDN_STATE_ONCE off?)")
        _tag, cin, T, t0 = hists
        if not dn._conv_multi_check(T):
            raise RuntimeError(f"{ENV}: needs QWEN36_MTP_CONV_MULTI engaged")
        ents.append((cin, T, t0, wins))
    Ts = {e[1] for e in ents}
    t0s = {e[2] for e in ents}
    if len(Ts) != 1 or len(t0s) != 1:
        raise RuntimeError(f"{ENV}: layers disagree on T {sorted(Ts)} / SO_T0 {sorted(t0s)}")
    T, t0 = Ts.pop(), t0s.pop()
    d0 = dns[0]
    info = {}
    if "rec" in kinds:
        info["rec"] = commit_rec_packed([dn.rec_state for dn in dns], [e[0] for e in ents], idx, d0.Nk, d0.Nv, d0.Dk,
                                        d0.Dv, d0.scale, T, Bw, t0=t0)
    else:
        for dn, e in zip(dns, ents):
            so.commit(dn.rec_state, e[0], idx, dn.Nk, dn.Nv, dn.Dk, dn.Dv, dn.scale, T, Bw, t0=t0)
    if "conv" in kinds:
        info["conv"] = commit_conv_packed([dn.conv_states for dn in dns], [e[3] for e in ents], idx, Bw, T)
    else:
        for dn, e in zip(dns, ents):
            ss.commit_conv_multi(dn.conv_states, e[3], idx, Bw, T)
    cls = type(d0)
    if logger is not None and not getattr(cls, "_mtp_slot_banner", False):
        # the per-layer path logs this from mtp_commit_slot (gdn/tp.py), which the packed path does not call
        cls._mtp_slot_banner = True
        logger.info(
            f"QWEN36_MTP_STATE_SLOT engaged: GDN commit by slot index, rec {d0.rec_state.dtype} "
            f"[{Bw}x{d0.Nv}x{d0.Dk}x{d0.Dv}] recomputed from the base state over the accepted tokens of T={T} "
            f"(QWEN36_MTP_GDN_STATE_ONCE) + {d0.K} conv taps {d0.conv_states[0].dtype} ({ENV}: all layers packed)"
        )
    key = (Bw, T, tuple(sorted(kinds)))
    if logger is not None and key not in _COMMIT_PACK_LOGGED:
        _COMMIT_PACK_LOGGED.add(key)
        n_prog = (1 if "rec" in kinds else len(dns)) + (1 if "conv" in kinds else len(dns))
        logger.info(
            f"{BANNER}: GDN commit of {len(dns)} layers at Bw={Bw} T={T} packed={sorted(kinds)}: {n_prog} programs/step "
            f"(was {2 * len(dns)}); rec {info.get('rec', {}).get('instances', '-')} instances on "
            f"{info.get('rec', {}).get('cores', '-')} cores, conv {info.get('conv', {}).get('units', '-')} units on "
            f"{info.get('conv', {}).get('cores', '-')} cores"
            + (f"; {MF_BANNER}: the last layer gets layer {len(dns) - 2}'s commit inputs" if mustfail() else "")
        )
    if release:
        for dn, e in zip(dns, ents):
            for t in (e[0],) + tuple(e[3]):
                ttnn.deallocate(t)
            del dn._mtp_verify[Bw]
    return info


# ------------------------------------------------------------------------------------------------ self-test
def selftest(dn0, T, bw, kinds, seed=0, n_layers=3, fid=None, t0=None, pack_mf_served=None, so_mf=None, max_cores=None,
             extra_pages=()):
    """Fail-closed comparison of the packed commit against the per-layer commit (state_once.commit +
    state_slot.commit_conv_multi), bitwise on every chip, on the device dn0 lives on, at dn0's state / tap shapes and
    dtypes, for n_layers synthetic layers (fresh tensors: random states, taps, cin sticks in the served value ranges,
    random windows) and every state_once.selftest_pages() idx page. Then the explicit must-fail (last layer fed the
    previous layer's inputs) must differ. pack_mf_served: the packed calls' must-fail for the served comparisons
    (default: the env), so under QWEN36_MTP_COMMIT_PACK_MUSTFAIL=1 they differ and selftest_ok() refuses.
    Returns {check: "exact" | "differs"}."""
    import torch

    so, ss = _sib("state_once"), _sib("state_slot")

    pack_mf_served = mustfail() if pack_mf_served is None else pack_mf_served
    t0 = so.t0_enabled() if t0 is None else t0
    so_mf = so.mustfail() if so_mf is None else so_mf
    if n_layers < 2:
        raise RuntimeError(f"{ENV}: the self-test needs >= 2 layers (the must-fail swaps layer inputs)")
    st0 = dn0.rec_state
    dev = st0.device()
    mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    rep = ttnn.ReplicateTensorToMesh(dev) if mesh else None
    dt, cdt = st0.dtype, dn0.conv_states[0].dtype
    tdt = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
    Nk, Nv, Dk, Dv = dn0.Nk, dn0.Nv, dn0.Dk, dn0.Dv
    Bmax = int(st0.shape[0])
    cshape = [int(x) for x in dn0.conv_states[0].shape]
    C = cshape[2]
    M = T * bw
    W = so.cin_width(Dk, Dv)
    DR = ttnn.DRAM_MEMORY_CONFIG
    g = torch.Generator().manual_seed(seed)

    def up(t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, mesh_mapper=rep, memory_config=DR)

    def chips(t):
        return [ttnn.to_torch(x) for x in (ttnn.get_device_tensors(t) if mesh else [t])]

    def cin_host():
        c = torch.zeros(M, 1, Nv, W)
        c[..., :Dk] = torch.randn(M, 1, Nv, Dk, generator=g)
        c[..., Dk:Dk + Dv] = torch.randn(M, 1, Nv, Dv, generator=g)
        c[..., Dk + Dv] = torch.sigmoid(torch.randn(M, 1, Nv, generator=g))
        c[..., Dk + Dv + 1] = -torch.nn.functional.softplus(torch.randn(M, 1, Nv, generator=g)) * 0.5
        return c.to(tdt[dt])

    L = n_layers
    s_h = [(torch.randn(Bmax, Nv, Dk, Dv, generator=g) * 0.05).to(tdt[dt]) for _ in range(L)]
    lv_h = [[torch.randn(*cshape, generator=g).to(tdt[cdt]) for _ in range(NTAP)] for _ in range(L)]
    keep = []
    res = {}
    try:
        cins = [up(cin_host(), dt, ttnn.ROW_MAJOR_LAYOUT) for _ in range(L)]
        wins = [[up(torch.randn(1, M, C, generator=g).to(tdt[cdt]), cdt) for _ in range(NTAP)] for _ in range(L)]
        keep += cins + [t for w in wins for t in w]
        pages = list(so.selftest_pages(T, bw, Bmax, seed)) + [list(p) for p in extra_pages]

        def run(page, pack_mf):
            idx = up(torch.tensor(page, dtype=torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ref_s = [up(s, dt) for s in s_h]
            got_s = [up(s, dt) for s in s_h]
            ref_l = [[up(t, cdt) for t in lv] for lv in lv_h]
            got_l = [[up(t, cdt) for t in lv] for lv in lv_h]
            keep.extend([idx] + ref_s + got_s + [t for lv in ref_l + got_l for t in lv])
            for l in range(L):
                so.commit(ref_s[l], cins[l], idx, Nk, Nv, Dk, Dv, dn0.scale, T, bw, mf=so_mf, fid=fid, t0=t0)
                ss.commit_conv_multi(ref_l[l], wins[l], idx, bw, T)
            if "rec" in kinds:
                commit_rec_packed(got_s, cins, idx, Nk, Nv, Dk, Dv, dn0.scale, T, bw, mf=so_mf, fid=fid, t0=t0,
                                  pack_mf=pack_mf, max_cores=max_cores)
            else:
                for l in range(L):
                    so.commit(got_s[l], cins[l], idx, Nk, Nv, Dk, Dv, dn0.scale, T, bw, mf=so_mf, fid=fid, t0=t0)
            if "conv" in kinds:
                commit_conv_packed(got_l, wins, idx, bw, T, pack_mf=pack_mf, max_cores=max_cores)
            else:
                for l in range(L):
                    ss.commit_conv_multi(got_l[l], wins[l], idx, bw, T)
            rec_ok = all(so._same(a, b) for l in range(L) for a, b in zip(chips(ref_s[l]), chips(got_s[l])))
            conv_ok = all(so._same(a, b) for l in range(L) for k in range(NTAP)
                          for a, b in zip(chips(ref_l[l][k]), chips(got_l[l][k])))
            return rec_ok, conv_ok

        for k, page in enumerate(pages):
            rec_ok, conv_ok = run(page, pack_mf_served)
            res[f"rec_page{k}"] = "exact" if rec_ok else "differs"
            res[f"conv_page{k}"] = "exact" if conv_ok else "differs"
        rec_ok, conv_ok = run(pages[0], True)
        if "rec" in kinds:
            res["mustfail_rec"] = "exact" if rec_ok else "differs"
        if "conv" in kinds:
            res["mustfail_conv"] = "exact" if conv_ok else "differs"
    finally:
        for t in keep:
            ttnn.deallocate(t)
    return res


def selftest_ok(res, kinds):
    """Every served check exact; the must-fail rung of every packed kind present and differing."""
    if not res:
        return False
    for kind in kinds:
        if res.get(f"mustfail_{kind}") != "differs":
            return False
    return all((v == "differs") if k.startswith("mustfail_") else (v == "exact") for k, v in res.items())
