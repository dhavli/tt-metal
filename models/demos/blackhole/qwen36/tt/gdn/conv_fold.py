# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_CONV_FOLD=1: QWEN36_MTP_CONV_MULTI's window build folded into the batch-M conv_gates reader
(docs/OP-DATASHEETS.md D3).

Today (CONV_MULTI, gdn/conv_multi.py) the MTP verify's GDN conv is two programs per layer before the recurrence:
  build_windows   rows_reader/rows_writer on 96 cores: P_1..P_3 [1, M, C] (row u = j*Bw + b <- s_b[j+k]) composed
                  from the live conv taps and qkv, written to DRAM (P_0 only allocated)
  conv_gates      gdn_decode_conv_gates(x=qkv, states=P_0..P_3, batch=M) on 81 cores: reads P_1..P_3 + the qkv page,
                  writes conv/beta/g and the shifted windows P_k <- window slot k (in place)
c155a measured the pair at +4.05 ms per verify when doubled (84 us/layer over 48 GDN layers), bound by per-program
fixed cost and the reader chain, not by bytes (~2 MB/layer).

Here ONE generic_op does both: the conv_gates program (the image's compute and writer kernels, copied verbatim to
kernels/conv_fold/, and its factory ported line for line: same 81 cores, CBs, HiFi4 + fp32 dest) with a reader
(kernels/conv_fold/reader_conv_fold.cpp, the arm's .gdc reader plus the fold) that composes each window tile in L1
from the live taps' page and the qkv page it already reads. The window CB therefore holds exactly the bytes the
.gdc reader DMAs from build_windows' P_1..P_3 today, so conv/beta/g and the four window tensors the writer stores are
bit-identical (by construction; ttsim gate bench/probes/sim_mtp_conv_fold.py; warmup self-test below).

Moved cost (stated, not measured): build_windows' RISC row copies move into the conv_gates reader (per core 2
instances x 3 composed tiles, vs 5 composed tiles + their zero fill per build_windows core); the windows' DRAM
write + read-back (3 x 160 pages) and one program launch per GDN layer go away. The four window tensors are still
WRITTEN (by the conv_gates writer, as today): state_slot.commit_conv_multi / commit_pack read them at the commit, and
the state-once verify reads conv/beta/g (not the windows), so nothing downstream changes. The live taps are only
read (the commit still writes them).

Requires QWEN36_MTP_CONV_MULTI=1 (and so QWEN36_MTP_STATE_SLOT=1) and QWEN_GDN_CONV_GATES=1; refuses otherwise.
Off (unset/0): nothing here runs, byte-identical to CONV_MULTI.
MUSTFAIL (QWEN36_MTP_CONV_FOLD_MUSTFAIL=1): the last candidate's windows read one token too far (the window map of
QWEN36_MTP_CONV_MULTI_MUSTFAIL, folded): the warmup self-test REFUSES the boot. Every boot's self-test also runs that
wrong-input branch explicitly and refuses if it does NOT differ (so each boot proves the test can fail).
QWEN36_MTP_CONV_FID=hifi4|hifi2|lofi: the fold program's math fidelity (default hifi4 = the image op's). Not
bit-exact below hifi4 (the window x tap products only; the SFPU silu/sigmoid/softplus are fidelity-free): gated by
the fp64-reference error in ttsim, never by the bitwise self-test, which then compares the fold at hifi4 only.
"""

import os
from pathlib import Path

try:
    import ttnn
except ImportError:  # host unit tests: the pure helpers need no ttnn
    ttnn = None

ENV = "QWEN36_MTP_CONV_FOLD"
ENV_MF = "QWEN36_MTP_CONV_FOLD_MUSTFAIL"
ENV_FID = "QWEN36_MTP_CONV_FID"
FIDS = ("hifi4", "hifi2", "lofi")
BANNER = "QWEN36_MTP_CONV_FOLD engaged"
MF_BANNER = "QWEN36_MTP_CONV_FOLD_MUSTFAIL engaged"
FID_BANNER = "QWEN36_MTP_CONV_FID engaged"
SELFTEST_OK = "QWEN36_MTP_CONV_FOLD self-test exact"
SELFTEST_REFUSED = "QWEN36_MTP_CONV_FOLD self-test REFUSED"
NTAP = 4
TILE = 32
NO_MF = 0xFFFF
_CONV_FOLD_LOGGED = set()  # name matches arm_switch MODULE_GUARD_RE: a served arm switch re-earns the banners

_KDIR = Path(__file__).resolve().parent / "kernels" / "conv_fold"
_SRC = {}
# the image factory's CB plan (gdn_conv_gates_program_factory.cpp): (index, pages, buffers, fp32) + the fold scratch
_CBS = [(0, NTAP, 2, False), (1, NTAP, 2, False), (2, NTAP, 1, True), (3, 1, 2, True), (4, 1, 2, False),
        (5, NTAP, 2, False), (6, 1, 2, False), (7, 1, 2, False), (8, 1, 2, False), (9, 1, 2, False),
        (10, 1, 2, False), (11, 1, 2, False), (12, 1, 2, False), (13, 2, 1, False)]
_CB_LV = 14  # 3 pages: lives[1..3] page cc (reader-private scratch)
ONE_BITS, TWENTY_BITS = 0x3F800000, 0x41A00000  # softplus beta=1, threshold=20 (the image factory's)


def _flag(env, name):
    v = env.get(name, "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{name}={v!r}: expected unset, 0 or 1")
    return v == "1"


def enabled(env=None):
    env = os.environ if env is None else env
    on = _flag(env, ENV)
    if not on and _flag(env, ENV_MF):
        raise ValueError(f"{ENV_MF}=1 needs {ENV}=1")
    if not on and env.get(ENV_FID, "").strip():
        raise ValueError(f"{ENV_FID} needs {ENV}=1 (it sets the fold program's fidelity)")
    return on


def mustfail(env=None):
    env = os.environ if env is None else env
    return _flag(env, ENV_MF)


def fidelity(env=None):
    """The fold program's math fidelity name (default hifi4, the image op's)."""
    env = os.environ if env is None else env
    v = env.get(ENV_FID, "").strip().lower() or "hifi4"
    if v not in FIDS:
        raise ValueError(f"{ENV_FID}={v!r}: expected one of {FIDS}")
    return v


def check_env(env=None):
    """True when the fold is engaged; refuses (RuntimeError) the combinations it does not cover; False when off."""
    env = os.environ if env is None else env
    if not enabled(env):
        return False
    need = {"QWEN36_MTP_CONV_MULTI": "1", "QWEN36_MTP_STATE_SLOT": "1", "QWEN_GDN_CONV_GATES": "1"}
    bad = [f"{k}={v}" for k, v in need.items() if env.get(k, "0") != v]
    if bad:
        raise RuntimeError(f"{ENV}=1 refused: needs {', '.join(bad)} (it replaces CONV_MULTI's build_windows + "
                           "gdn_decode_conv_gates pair; the slot commit reads its windows)")
    if env.get("QWEN36_MTP_CONV_MULTI_MUSTFAIL", "0") == "1":
        raise RuntimeError(f"{ENV}=1 refused with QWEN36_MTP_CONV_MULTI_MUSTFAIL=1: use {ENV_MF}=1 (one must-fail "
                           "per path)")
    return True


def op_counts():
    """Programs per GDN layer for the verify conv: (fold, CONV_MULTI)."""
    return 1, 2


def banner(T, bw, logger=None):
    key = (int(T), int(bw))
    if key in _CONV_FOLD_LOGGED:
        return
    _CONV_FOLD_LOGGED.add(key)
    M = T * bw
    n, m = op_counts()
    msg = (f"QWEN36_MTP_CONV_FOLD engaged: build_windows folded into the conv_gates reader at batch M={M} (T={T} x Bw={bw}), "
           f"{n} conv program/layer (was {m}); reads live taps 1..3 + qkv directly, writes the 4 windows for the "
           f"commit; fidelity={fidelity()}")
    if fidelity() != "hifi4":
        msg += f"; QWEN36_MTP_CONV_FID engaged: {fidelity()} (window x tap / gate products; not bit-exact)"
    if mustfail():
        msg += "; QWEN36_MTP_CONV_FOLD_MUSTFAIL engaged: the last candidate's windows read one token too far, the self-test MUST refuse"
    (logger.info if logger is not None else print)(msg)


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


def _fid(name):
    return {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[name]


def window_map(T, bw, k, mf=False):
    """Pure helper (host unit tests): row u = j*bw + b of window P_k <- ('tap', i, b) | ('qkv', row), the source the
    reader copies (the same map as conv_multi.window_map)."""
    out = []
    for u in range(T * bw):
        j, b = divmod(u, bw)
        i = j + k + (1 if (mf and j == T - 1) else 0)
        out.append(("tap", i, b) if i < NTAP else ("qkv", (i - NTAP) * bw + b))
    return out


def conv_gates(lives, x, taps, a, b, dt_bias, neg_exp_A, T, bw, memory_config=None, channels=None, a_col=0, b_col=0,
               mf=False, fid=None, win_mc=None):
    """The fold: gdn_decode_conv_gates(x, build_windows(lives, x, T, bw), taps, a, b, dt_bias, neg_exp_A, batch=T*bw,
    channels, a_col, b_col) as ONE program. lives: the NTAP live conv taps [1, Bmax, C] (read only). x: the verify
    input [1, M, W >= C] (qkv, or the fused qkvzab under QWEN36_GDN_DECODE_CHAIN=direct with channels=C). Returns
    (conv [1, M, C], beta [1, M, Nv], g [1, M, Nv], wins) with wins the 4 window tensors [1, M, C] (DRAM, as
    build_windows allocates them) after the shift: row j*bw + b = row b's taps after j+1 tokens."""
    if len(lives) != NTAP or len(taps) != NTAP:
        raise RuntimeError(f"{ENV}: needs {NTAP} live taps and {NTAP} conv weights ({len(lives)}, {len(taps)})")
    M = T * bw
    xs, ls, as_, bs_ = _shape(x), _shape(lives[0]), _shape(a), _shape(b)
    C = channels if channels is not None else xs[-1]
    Nv = _shape(dt_bias)[-1]
    dt = lives[0].dtype
    if not (1 <= T and 1 <= bw and M <= TILE):
        raise RuntimeError(f"{ENV}: M = T*bw = {M} must be in 1..{TILE}")
    if len(xs) != 3 or xs[0] != 1 or xs[1] != M or xs[2] < C or C % TILE:
        raise RuntimeError(f"{ENV}: x {xs} vs M={M} C={C}")
    for i, t in enumerate(lives):
        _interleaved(t, f"live tap {i}")
        if _shape(t) != [1, ls[1], C] or t.dtype != dt:
            raise RuntimeError(f"{ENV}: live tap {i} {_shape(t)} {t.dtype} vs [1, Bmax, {C}] {dt}")
    if ls[1] < bw:
        raise RuntimeError(f"{ENV}: live taps hold {ls[1]} rows < bucket {bw}")
    if x.dtype != dt:
        raise RuntimeError(f"{ENV}: x dtype {x.dtype} != conv tap dtype {dt}")
    for t, n in ((x, "x"), (a, "a"), (b, "b"), (dt_bias, "dt_bias"), (neg_exp_A, "neg_exp_A")):
        _interleaved(t, n)
    for i, t in enumerate(taps):
        _interleaved(t, f"conv weight {i}")
        if _shape(t) != [1, 1, C]:
            raise RuntimeError(f"{ENV}: conv weight {i} {_shape(t)} != [1, 1, {C}]")
    if not (as_[1] >= M and as_[2] >= a_col + Nv and bs_[1] >= M and bs_[2] >= b_col + Nv):
        raise RuntimeError(f"{ENV}: a {as_} / b {bs_} vs M={M}, a_col={a_col}, b_col={b_col}, Nv={Nv}")
    if dt == ttnn.bfloat16:
        tb = 2048
    elif dt == ttnn.float32:
        tb = 4096
    else:
        raise RuntimeError(f"{ENV}: unsupported io dtype {dt}")
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    win_mc = win_mc or ttnn.DRAM_MEMORY_CONFIG
    dev = lives[0].device()

    def alloc(rows, width, mcfg):
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, rows, width]), dt, ttnn.TILE_LAYOUT, dev, mcfg)

    out = alloc(M, C, mc)
    beta = alloc(M, Nv, mc)
    g = alloc(M, Nv, mc)
    wins = [alloc(M, C, win_mc) for _ in range(NTAP)]

    # gdn_conv_gates_program_factory.cpp, line for line (states = the windows: Bmax = M, one tile-row)
    Ct, Nvt = C // TILE, (Nv + TILE - 1) // TILE
    Wt, AWt, BWt = (xs[2] + TILE - 1) // TILE, (as_[2] + TILE - 1) // TILE, (bs_[2] + TILE - 1) // TILE
    gp = 1 if (a_col != 0 or b_col != 0 or as_[2] != Nv or bs_[2] != Nv) else 0
    n_conv, n_gate = Ct, Nvt  # Bt = xBt = gBt = 1
    grid = dev.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    per = (n_conv + ncores - 1) // ncores
    n_conv_cores = (n_conv + per - 1) // per
    n_cores = n_conv_cores + 1 if n_conv_cores < ncores else n_conv_cores
    cores = [ttnn.CoreCoord(c // grid.y, c % grid.y) for c in range(n_cores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    def cb(i, n, nbuf, f32):
        ts = 4096 if f32 else tb
        return ttnn.CBDescriptor(total_size=n * nbuf * ts, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32 if f32 else dt, page_size=ts)])

    cbs = [cb(*c) for c in _CBS] + [cb(_CB_LV, 3, 1, False)]

    def acc(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    mfj = (T - 1) if mf else NO_MF
    r_ct = [NTAP, Ct, Nvt, M, 1, Wt, gp, AWt, a_col, BWt, b_col, Nv, T, bw, mfj]
    for t in [x] + list(lives[1:]) + list(taps) + [a, b, dt_bias, neg_exp_A]:
        r_ct += acc(t)
    w_ct = [NTAP, Ct, Nvt]
    for t in [out] + wins + [beta, g]:
        w_ct += acc(t)
    c_ct = [NTAP, Nvt, ONE_BITS, TWENTY_BITS]
    r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    r_addr = [x.buffer_address()] + [t.buffer_address() for t in lives[1:]] + [t.buffer_address() for t in taps] + [
        a.buffer_address(), b.buffer_address(), dt_bias.buffer_address(), neg_exp_A.buffer_address()]
    w_addr = [out.buffer_address()] + [t.buffer_address() for t in wins] + [beta.buffer_address(), g.buffer_address()]
    for c, core in enumerate(cores):
        conv_core = c < n_conv_cores
        start = c * per if conv_core else 0
        n_inst = min(per, n_conv - start) if conv_core else 0
        g_n = n_gate if c == n_cores - 1 else 0
        r_rt[core.x][core.y] = [start, n_inst, g_n] + r_addr
        w_rt[core.x][core.y] = [start, n_inst, g_n] + w_addr
        c_rt[core.x][core.y] = [n_inst, g_n]
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = _fid(fid or fidelity())
    ccfg.fp32_dest_acc_en = True
    ccfg.math_approx_mode = False

    def kd(name, ct, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)

    desc = ttnn.ProgramDescriptor(
        kernels=[kd("reader_conv_fold.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
                 kd("writer_gdn_conv_gates.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
                 kd("gdn_conv_gates.cpp", c_ct, c_rt, ccfg)],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([x] + list(lives[1:]) + list(taps) + [a, b, dt_bias, neg_exp_A, out] + wins + [beta, g], desc)
    return out, beta, g, wins


# ------------------------------------------------------------------------------------------------ self-test
def _bits(t):
    import torch

    t = t.contiguous()
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def _diff(a, b):
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    return int((_bits(a) != _bits(b)).sum())


def _image_conv_gates(x, wins, taps, a, b, dt_bias, neg_exp_A, batch, memory_config, channels=None, a_col=0, b_col=0):
    kw = {} if channels is None else {"channels": channels, "a_col": a_col, "b_col": b_col}
    return ttnn.transformer.gdn_decode_conv_gates(x, wins, taps, a, b, dt_bias, neg_exp_A, batch=batch,
                                                  memory_config=memory_config, **kw)


def selftest(lives, taps, dt_bias, neg_exp_A, T, bw, lay=None, seed=0, ref_conv_gates=None, build_windows=None):
    """Fail-closed comparison of the fold against the path it replaces, on the device `lives` sit on, at the bucket's
    T/bw and the live taps' shape/dtype (lives are only used for their shape: every tensor here is a fresh copy).
    Weights are the layer's DEVICE tensors (taps, dt_bias, neg_exp_A), never host weights (a warm-cache boot hands the
    model zero host placeholders; c157). Activations and tap states are random.

    Reference = conv_multi.build_windows(lives, x, T, bw) + the image's gdn_decode_conv_gates at batch M on the windows
    (ref_conv_gates; ttsim passes the factory port); with `lay` (QWEN36_GDN_DECODE_CHAIN=direct) x = a = b = a random
    qkvzab [1, M, lay.W] with channels=lay.C, a_col, b_col, exactly as decode_chain.conv_gates calls the op.
    Candidate = conv_gates() above at hifi4 (the served QWEN36_MTP_CONV_FOLD_MUSTFAIL honoured). Compared bitwise on
    every chip: conv [1,M,C], beta/g [1,M,Nv] (logical elements), all 4 window tensors, and the live taps (must stay
    untouched). Wrong-input branch, every boot: the explicit must-fail (last candidate one token too far) MUST change
    conv. Returns {check: n_differing_elements, ..., "mustfail_conv": n}."""
    import torch

    if build_windows is None:
        try:
            from models.demos.blackhole.qwen36.tt.gdn import conv_multi as _cm
        except ImportError:  # loaded by path (ttsim probes): the sibling module
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location("conv_multi", Path(__file__).resolve().parent / "conv_multi.py")
            _cm = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_cm)
        build_windows = _cm.build_windows
    ref_conv_gates = ref_conv_gates or _image_conv_gates
    dev = lives[0].device()
    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    rep = ttnn.ReplicateTensorToMesh(dev) if is_mesh else None
    Bmax, C = _shape(lives[0])[1], _shape(lives[0])[2]
    Nv = _shape(dt_bias)[-1]
    M = T * bw
    dt = lives[0].dtype
    tdt = torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    gen = torch.Generator().manual_seed(seed)

    def up(t, mc):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep, memory_config=mc)

    def back(t):
        return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] if is_mesh else [ttnn.to_torch(t)]

    def diff(xs, ys):
        return sum(_diff(p, q) for p, q in zip(back(xs), back(ys)))

    live_t = [torch.randn(1, Bmax, C, generator=gen).to(tdt) for _ in range(NTAP)]
    if lay is not None:
        if lay.C != C:
            raise RuntimeError(f"{ENV}: layout C={lay.C} != live tap width {C}")
        x_t = torch.randn(1, M, lay.W, generator=gen).to(tdt)
        kw = {"channels": lay.C, "a_col": lay.a_col, "b_col": lay.b_col}
    else:
        x_t = torch.randn(1, M, C, generator=gen).to(tdt)
        a_t = torch.randn(1, M, Nv, generator=gen).to(tdt)
        b_t = torch.randn(1, M, Nv, generator=gen).to(tdt)
        kw = {}
    res = {}
    keep = []
    try:
        x = up(x_t, L1)
        if lay is not None:
            a = b = x
        else:
            a, b = up(a_t, L1), up(b_t, L1)
        lv_r = [up(t, DR) for t in live_t]
        lv_c = [up(t, DR) for t in live_t]
        keep += [x] + ([] if lay is not None else [a, b]) + lv_r + lv_c
        wins_r = build_windows(lv_r, x, T, bw)
        keep += wins_r
        ref = ref_conv_gates(x, wins_r, taps, a, b, dt_bias, neg_exp_A, batch=M, memory_config=L1, **kw)
        keep += list(ref)
        cand = conv_gates(lv_c, x, taps, a, b, dt_bias, neg_exp_A, T, bw, L1, mf=mustfail(), fid="hifi4", **kw)
        keep += list(cand[:3]) + list(cand[3])
        for n, p, q in zip(("conv", "beta", "g"), ref, cand[:3]):
            res[n] = diff(p, q)
        for k in range(NTAP):
            res[f"win{k}"] = diff(wins_r[k], cand[3][k])
        lv0 = [up(t, DR) for t in live_t]
        keep += lv0
        res["lives_untouched"] = sum(diff(p, q) for p, q in zip(lv0, lv_c))
        # wrong-input branch (every boot): the last candidate's windows one token too far must change conv
        mfo = conv_gates(lv_c, x, taps, a, b, dt_bias, neg_exp_A, T, bw, L1, mf=True, fid="hifi4", **kw)
        keep += list(mfo[:3]) + list(mfo[3])
        res["mustfail_conv"] = diff(ref[0], mfo[0])
    finally:
        for t in keep:
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return bool(res) and res.get("mustfail_conv", 0) > 0 and all(
        v == 0 for k, v in res.items() if k != "mustfail_conv")
