# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFILL_DNORM_FUSE=1: the prefill distributed RMSNorm (pre -> stats all-gather -> post) with h read from DRAM
ONCE, on both NoCs, and kept in L1 between the two passes.

Why (measured, [source validation record], CONTROL dfr, 32k chunk): every decoder
norm is rms_norm_pre_all_gather 41.8 us + all_gather_async 16 us + rms_norm_post_all_gather 86.0 us, 128 norms per
2048-token chunk = 18.4 ms/chunk. Both image ops put one core on each of the 64 tile rows (64 of 110 cores) and their
readers barrier on every tile (reader_unary_interleaved_ln_rm_gb_{pre,post}_allgather.cpp), so post moves its 16 MB at
~190 GB/s and h ([M, 2560] bf16, 10.5 MB per chip) is read from DRAM twice.

What (two ttnn.generic_op programs on the same M/32 cores, JIT kernels in kernels/dnorm_fuse/, no image change):
  PRE  each core reads its tile row of h on BOTH RISCs/NoCs (chunk j on RISC j % 2, one barrier per chunk) straight
       into its shard of hs, an L1 HEIGHT-sharded [M, W] bf16 tensor (32 x W per core), and computes the row's stats
       with the image's rmsnorm_pre_allgather LLK sequence (x*x HiFi2 fp32 DEST -> fp32 CB, FPU row reduce -> bf16).
  AG   the same ttnn.experimental.all_gather_async of the stats the image path runs (unchanged).
  POST each core reads only the gathered stats tiles and gamma sticks; h comes from its own hs shard (no DRAM read).
       The image's rmsnorm_post_allgather LLK sequence (reduce AVG / eps / rsqrt, mul_bcast_cols, mul_bcast_rows).
Same LLK calls, CB data formats, fidelity and DEST mode as the image ops, row for row, so the same bytes (self-tested
bitwise on both chips). Moved cost: none expected; the stats all-gather and every other op are untouched. hs lives
from PRE to POST (160 KB per core at M=2048) and is freed right after POST.

Applies to: models/common/rmsnorm.py RMSNorm._distributed_rmsnorm (prefill attention_norm / ff_norm / final norm on
the TP mesh) whenever inp is [1, 1, M, W] bf16 TILE interleaved with M % 32 == 0, M / 32 <= cores, and that (M, W,
out dtype) passed the self-test this process. Everything else: today's three ops (logged once).

Self-test (fail closed, ensure_selftest, eager, before any trace): random per-chip h (DRAM) and a random gamma in the
norm weight's layout, the fused path vs the image's three ops through the SAME all-gather, bitwise on the output (and
the gathered stats) on every chip. Wrong-input branch at every boot: POST with gamma shifted by one tile (tile t gets
stick t + 1) MUST differ. QWEN36_PREFILL_DNORM_FUSE_MUSTFAIL=1 serves that shift: the comparison differs and the boot is REFUSED.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
import struct
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_PREFILL_DNORM_FUSE"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")
if _MF == "1" and _MODE != "1":
    raise RuntimeError(f"{MUSTFAIL_ENV}=1 needs {ENV}=1")

ADD_ENV = "QWEN36_PREFILL_DNORM_ADD"
ADD_MUSTFAIL_ENV = ADD_ENV + "_MUSTFAIL"
_ADD = _tt_public_getenv(ADD_ENV, "0")
if _ADD not in ("0", "1"):
    raise RuntimeError(f"{ADD_ENV} must be 0 or 1; got {_ADD!r}")
if _ADD == "1" and _MODE != "1":
    raise RuntimeError(f"{ADD_ENV}=1 needs {ENV}=1")
_ADD_MF = _tt_public_getenv(ADD_MUSTFAIL_ENV, "0")
if _ADD_MF not in ("0", "1"):
    raise RuntimeError(f"{ADD_MUSTFAIL_ENV} must be 0 or 1; got {_ADD_MF!r}")
if _ADD_MF == "1" and _ADD != "1":
    raise RuntimeError(f"{ADD_MUSTFAIL_ENV}=1 needs {ADD_ENV}=1")

# QWEN36_PREFILL_DNORM_XADD=1 (needs DNORM_ADD): the cross-layer residual x_next = h + ff_out folded the same way into
# the NEXT layer's attention-norm PRE pass (layer.py forward(fold_next=...)); same run_add program and self-test cells.
XADD_ENV = "QWEN36_PREFILL_DNORM_XADD"
_XADD = _tt_public_getenv(XADD_ENV, "0")
if _XADD not in ("0", "1"):
    raise RuntimeError(f"{XADD_ENV} must be 0 or 1; got {_XADD!r}")
if _XADD == "1" and _ADD != "1":
    raise RuntimeError(f"{XADD_ENV}=1 needs {ADD_ENV}=1")

SELFTEST_BANNER = "QWEN36_PREFILL_DNORM_FUSE self-test exact"
ADD_SELFTEST_BANNER = "QWEN36_PREFILL_DNORM_ADD self-test exact"
TILE = 32
CHUNK = 8  # tiles per reader barrier (PRE)
PRE_BLK = 4  # tiles per DEST acquire (PRE x**2)
POST_BLK = 4  # tiles per DEST acquire (POST), as the image factory picks for fp32 DEST (find_max_divisor(Wt, 4))

_KDIR = Path(__file__).resolve().parent / "kernels" / "dnorm_fuse"
_SRC = {}
_tested = {}  # (M, W, out_dtype) -> "exact"
_logged = set()


# Workbench-only bypass (tools/ttx/clients/wb_knobs.py), read per call so `ttx wb set` + re-capture gives the image
# path on the same boot for a same-process per-op comparison: "all" bypasses both levers, "add" only DNORM_ADD.
# Unset in every served arm; it never enables anything.
WB_BYPASS_ENV = "QWEN36_PREFILL_DNORM_WB_BYPASS"


def _bypass():
    return _tt_public_getenv(WB_BYPASS_ENV, "0")


def enabled():
    return _MODE == "1" and _bypass() != "all"


def mustfail():
    return _MF == "1"


def add_enabled():
    return _ADD == "1" and _bypass() not in ("all", "add")


def xadd_enabled():
    return _XADD == "1" and _bypass() not in ("all", "add", "xadd")


def add_mustfail():
    return _ADD_MF == "1"


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _shape(t):
    return [int(x) for x in t.shape]


def key(M, W, out_dtype):
    return (int(M), int(W), str(out_dtype))


def _grid(dev):
    g = dev.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def _check(inp):
    s = _shape(inp)
    if len(s) != 4 or s[0] != 1 or s[1] != 1:
        raise RuntimeError(f"{ENV}: input shape {s}, expected [1, 1, M, W]")
    M, W = s[2], s[3]
    if M % TILE or W % TILE:
        raise RuntimeError(f"{ENV}: M={M} W={W} not tile multiples")
    ps = [int(x) for x in inp.padded_shape]
    if ps != s:
        raise RuntimeError(f"{ENV}: padded shape {ps} != logical {s}")
    if inp.dtype != ttnn.bfloat16 or inp.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: input {inp.dtype} {inp.layout}, expected bf16 TILE")
    if inp.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: input must be interleaved, got {inp.memory_config()}")
    gx, gy = _grid(inp.device())
    if M // TILE > gx * gy:
        raise RuntimeError(f"{ENV}: M={M} needs {M // TILE} cores > {gx * gy}")
    return M, W


def usable(inp, out_dtype):
    """True when the flag is on, inp has the fused path's shape, and the self-test passed this (M, W, dtype)."""
    if not enabled():
        return False
    try:
        M, W = _check(inp)
    except RuntimeError as e:
        k = ("shape", str(e))
        if k not in _logged:
            _logged.add(k)
            logger.warning(f"QWEN36_PREFILL_DNORM_FUSE fallback: {e}; today's pre/all-gather/post serve it")
        return False
    if key(M, W, out_dtype) in _tested:
        return True
    k = ("untested", key(M, W, out_dtype))
    if k not in _logged:
        _logged.add(k)
        logger.warning(f"QWEN36_PREFILL_DNORM_FUSE fallback: M={M} W={W} out={out_dtype} not self-tested this process; today's ops serve it")
    return False


def _cores(dev, R):
    gx, gy = _grid(dev)
    crs = ttnn.num_cores_to_corerangeset(R, ttnn.CoreCoord(gx, gy), True)
    return [ttnn.CoreCoord(r % gx, r // gx) for r in range(R)], crs


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2  # RMSNorm.compute_kernel_config_hifi2
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def _cb(idx, n_pages, page, fmt, crs):
    return ttnn.CBDescriptor(total_size=n_pages * page, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])


def _tsize(dtype):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dtype]


def _kern(name, crs, ct, rt, cfg, defines=None):
    return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                 core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg,
                                 defines=list((defines or {}).items()))


def pre(inp, stats_mc):
    """PRE pass: returns (stats [1,1,M,32] bf16 in stats_mc, hs = h resident in L1, height-sharded one tile row/core)."""
    M, W = _check(inp)
    dev = inp.device()
    WT, R = W // TILE, M // TILE
    cores, crs = _cores(dev, R)
    hs_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                              ttnn.ShardSpec(crs, [TILE, W], ttnn.ShardOrientation.ROW_MAJOR))
    hs = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs_mc)
    stats = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, stats_mc)
    CB_H, CB_C0, CB_C1, CB_SC, CB_X2, CB_ST = 0, 1, 2, 3, 4, 5
    nchunk = (WT + CHUNK - 1) // CHUNK
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_H, hs),
           _cb(CB_C0, (nchunk + 1) // 2, 64, ttnn.bfloat16, crs),
           _cb(CB_C1, max(1, nchunk // 2), 64, ttnn.bfloat16, crs),
           _cb(CB_SC, 1, 4096, ttnn.float32, crs),  # image pre: scaler format = cb format (fp32 with fp32 DEST)
           _cb(CB_X2, WT, 4096, ttnn.float32, crs),
           _cb(CB_ST, 1, 2048, ttnn.bfloat16, crs)]
    acc = _acc(inp) + _acc(stats)
    rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    ha, sa = inp.buffer_address(), stats.buffer_address()
    for r, c in enumerate(cores):
        rt0[c.x][c.y] = [ha, sa, r]
        rt1[c.x][c.y] = [ha, sa, r]
    ct = lambda role, cnt: [CB_H, cnt, CB_SC, CB_ST, WT, CHUNK, role] + acc  # noqa: E731
    desc = ttnn.ProgramDescriptor(
        kernels=[_kern("pre_dm.cpp", crs, ct(0, CB_C0), rt0, ttnn.ReaderConfigDescriptor()),
                 _kern("pre_dm.cpp", crs, ct(1, CB_C1), rt1, ttnn.WriterConfigDescriptor()),
                 _kern("pre_compute.cpp", crs, [CB_H, CB_C0, CB_C1, CB_SC, CB_X2, CB_ST, WT, CHUNK, PRE_BLK], [],
                       _cfg())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([inp, hs, stats], desc)
    return stats, hs


def _check_b(a, b):
    if [int(x) for x in b.shape] != [int(x) for x in a.shape] or [int(x) for x in b.padded_shape] != _shape(a):
        raise RuntimeError(f"{ADD_ENV}: b shape {list(b.shape)} != a {list(a.shape)}")
    if b.layout != ttnn.TILE_LAYOUT or b.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
        raise RuntimeError(f"{ADD_ENV}: b {b.dtype} {b.layout}, expected bf16/bf8 TILE")
    if b.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ADD_ENV}: b must be interleaved, got {b.memory_config()}")


def pre_add(a, b, stats_mc, h):
    """PRE pass with the residual add folded in: h = a + b (written to the given interleaved bf16 h AND kept in L1 as
    hs), stats of h. Returns (stats, hs)."""
    M, W = _check(a)
    _check_b(a, b)
    dev = a.device()
    WT, R = W // TILE, M // TILE
    cores, crs = _cores(dev, R)
    hs_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                              ttnn.ShardSpec(crs, [TILE, W], ttnn.ShardOrientation.ROW_MAJOR))
    stats = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, stats_mc)
    hs = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, hs_mc)
    CB_A0, CB_B0, CB_A1, CB_B1, CB_H, CB_SC, CB_X2, CB_ST = 0, 1, 2, 3, 4, 5, 6, 7
    bsz = _tsize(b.dtype)
    cbs = [_cb(CB_A0, CHUNK, 2048, ttnn.bfloat16, crs), _cb(CB_B0, CHUNK, bsz, b.dtype, crs),
           _cb(CB_A1, CHUNK, 2048, ttnn.bfloat16, crs), _cb(CB_B1, CHUNK, bsz, b.dtype, crs),
           ttnn.cb_descriptor_from_sharded_tensor(CB_H, hs),
           _cb(CB_SC, 1, 4096, ttnn.float32, crs),
           _cb(CB_X2, WT, 4096, ttnn.float32, crs),
           _cb(CB_ST, 1, 2048, ttnn.bfloat16, crs)]
    acc = _acc(a) + _acc(b) + _acc(h) + _acc(stats)
    rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    adr = [a.buffer_address(), b.buffer_address(), h.buffer_address(), stats.buffer_address()]
    for r, c in enumerate(cores):
        rt0[c.x][c.y] = adr + [r]
        rt1[c.x][c.y] = adr + [r]
    ct = lambda role, ca, cb: [ca, cb, CB_H, CB_SC, CB_ST, WT, CHUNK, role] + acc  # noqa: E731
    desc = ttnn.ProgramDescriptor(
        kernels=[_kern("preadd_dm.cpp", crs, ct(0, CB_A0, CB_B0), rt0, ttnn.ReaderConfigDescriptor()),
                 _kern("preadd_dm.cpp", crs, ct(1, CB_A1, CB_B1), rt1, ttnn.WriterConfigDescriptor()),
                 _kern("preadd_compute.cpp", crs,
                       [CB_A0, CB_B0, CB_A1, CB_B1, CB_H, CB_SC, CB_X2, CB_ST, WT, CHUNK, PRE_BLK], [], _cfg())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([a, b, h, hs, stats], desc)
    return stats, hs


def post(hs, stats_g, gamma, eps, out_dtype, out_mc, num_devices, mf=False, y=None):
    """POST pass: y = hs * rsqrt(mean(stats_g) + eps) * gamma, [1,1,M,W] out_dtype TILE interleaved in out_mc."""
    dev = hs.device()
    M, W = _shape(hs)[2], _shape(hs)[3]
    WT, R = W // TILE, M // TILE
    SC = _shape(stats_g)[3] // TILE
    cores, crs = _cores(dev, R)
    if y is None:
        y = alloc_out(dev, M, W, out_dtype, out_mc)
    CB_H, CB_STATS, CB_G, CB_EPS, CB_SC, CB_VAR, CB_RECIP, CB_XN, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 7, 8
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_H, hs),
           _cb(CB_STATS, SC, 2048, ttnn.bfloat16, crs),
           _cb(CB_G, WT, 2048, ttnn.bfloat16, crs),
           _cb(CB_EPS, 1, 2048, ttnn.bfloat16, crs),
           _cb(CB_SC, 1, 2048, ttnn.bfloat16, crs),  # image post: scaler Float16_b for a bf16 input
           _cb(CB_VAR, 1, 4096, ttnn.float32, crs),
           _cb(CB_RECIP, 1, 4096, ttnn.float32, crs),
           _cb(CB_XN, WT, 4096, ttnn.float32, crs),
           _cb(CB_OUT, 4 * POST_BLK, _tsize(out_dtype), out_dtype, crs)]
    acc = _acc(stats_g) + _acc(gamma) + _acc(y)
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    sa, ga, ya = stats_g.buffer_address(), gamma.buffer_address(), y.buffer_address()
    for r, c in enumerate(cores):
        rt0[c.x][c.y] = [sa, ga, ya, r, 1 if mf else 0, eps_bits]
        rt1[c.x][c.y] = [sa, ga, ya, r, 1 if mf else 0, eps_bits]
    ct = lambda role: [CB_SC, CB_EPS, CB_STATS, CB_G, CB_OUT, WT, POST_BLK, SC, role] + acc  # noqa: E731
    dfs = {"REDUCE_FACTOR": str(W * int(num_devices))}
    desc = ttnn.ProgramDescriptor(
        kernels=[_kern("post_dm.cpp", crs, ct(0), rt0, ttnn.ReaderConfigDescriptor(), dfs),
                 _kern("post_dm.cpp", crs, ct(1), rt1, ttnn.WriterConfigDescriptor(), dfs),
                 _kern("post_compute.cpp", crs,
                       [CB_H, CB_STATS, CB_G, CB_EPS, CB_SC, CB_VAR, CB_RECIP, CB_XN, CB_OUT, WT, POST_BLK, SC], [],
                       _cfg())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([hs, stats_g, gamma, y], desc)
    return y


def alloc_out(dev, M, W, out_dtype, out_mc):
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), out_dtype, ttnn.TILE_LAYOUT, dev, out_mc)


def stats_all_gather(norm, stats, mc):
    """models/common/rmsnorm.py's stats all-gather, argument for argument."""
    return ttnn.experimental.all_gather_async(
        stats,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=norm.tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=1,
        topology=norm.ccl_topology,
        memory_config=mc,
        barrier_semaphore=norm.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


def _num_devices(dev):
    return int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1


def run(norm, inp, epsilon, weight, mc, out_dtype, mf=None, gather=None):
    """The fused distributed RMSNorm. Same contract as RMSNorm._distributed_rmsnorm (output in mc, dtype out_dtype or
    the input's). mf=None is the served call: refuses an untested (M, W, dtype)."""
    od = out_dtype if out_dtype is not None else inp.dtype
    M, W = _check(inp)
    if mf is None:
        k = key(M, W, od)
        if k not in _tested:
            raise RuntimeError(f"{ENV}: M={M} W={W} out={od} was not self-tested this process; refusing to serve it")
        mf = mustfail()
        if ("engaged", k) not in _logged:
            _logged.add(("engaged", k))
            logger.info(f"QWEN36_PREFILL_DNORM_FUSE engaged: M={M} W={W} out={od}: h read once on both NoCs into L1, "
                        f"{M // TILE} cores{' MUSTFAIL gamma tile +1' if mf else ''}")
    # y is allocated FIRST, so an L1 y lands where the image path puts it (above the transient hs/stats); allocated
    # after hs it sat below hs, and once hs was freed it reached into the next GDN op's static CB region
    # (2026-10-19 card boot: "Statically allocated circular buffers ... clash with L1 buffers", 852928 < 898048).
    y = alloc_out(inp.device(), M, W, od, mc)
    stats, hs = pre(inp, mc)
    g = (gather or (lambda s: stats_all_gather(norm, s, mc)))(stats)
    if g is not stats:
        ttnn.deallocate(stats)
    post(hs, g, weight, epsilon, od, mc, _num_devices(inp.device()), mf=bool(mf), y=y)
    ttnn.deallocate(g)
    ttnn.deallocate(hs)
    return y


def add_usable(a, b, out_dtype):
    """True when QWEN36_PREFILL_DNORM_ADD is on, (a, b) fit the folded path and its (M, W, dtype) passed the add self-test."""
    if not add_enabled():
        return False
    try:
        M, W = _check(a)
        _check_b(a, b)
    except RuntimeError as e:
        k = ("addshape", str(e))
        if k not in _logged:
            _logged.add(k)
            logger.warning(f"QWEN36_PREFILL_DNORM_ADD fallback: {e}; ttnn.add + the norm serve it")
        return False
    if ("add",) + key(M, W, out_dtype) + (str(b.dtype),) in _tested:
        return True
    k = ("adduntested", key(M, W, out_dtype), str(b.dtype))
    if k not in _logged:
        _logged.add(k)
        logger.warning(f"QWEN36_PREFILL_DNORM_ADD fallback: M={M} W={W} out={out_dtype} b={b.dtype} not self-tested "
                       f"this process; ttnn.add + the norm serve it")
    return False


def run_add(norm, a, b, epsilon, weight, mc, out_dtype, h_mc=None, mf=None, gather=None):
    """h = ttnn.add(a, b) and y = norm(h) in two programs (+ the stats all-gather). Returns (h, y); h is bf16
    interleaved in h_mc (default: a's memory config, as ttnn.add), y as run()."""
    od = out_dtype if out_dtype is not None else a.dtype
    M, W = _check(a)
    if mf is None:
        k = ("add",) + key(M, W, od) + (str(b.dtype),)
        if k not in _tested:
            raise RuntimeError(f"{ADD_ENV}: M={M} W={W} out={od} b={b.dtype} was not self-tested; refusing to serve it")
        mf = add_mustfail()
        if ("addengaged", k) not in _logged:
            _logged.add(("addengaged", k))
            logger.info(f"QWEN36_PREFILL_DNORM_ADD engaged: M={M} W={W} b={b.dtype} out={od}: residual add folded "
                        f"into the norm's PRE pass{' MUSTFAIL gamma tile +1' if mf else ''}")
    dev = a.device()
    # allocation order = today's (ttnn.add's h, then the norm's y, then its transients): see run()
    h = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                       h_mc if h_mc is not None else a.memory_config())
    y = alloc_out(dev, M, W, od, mc)
    stats, hs = pre_add(a, b, mc, h)
    g = (gather or (lambda s: stats_all_gather(norm, s, mc)))(stats)
    if g is not stats:
        ttnn.deallocate(stats)
    post(hs, g, weight, epsilon, od, mc, _num_devices(dev), mf=bool(mf), y=y)
    ttnn.deallocate(g)
    ttnn.deallocate(hs)
    return h, y


# ---------------------------------------------------------------------------------------------- self-test
def _chips(dev, t):
    return ttnn.get_device_tensors(t) if hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice) else [t]


def _bits(dev, t):
    import torch

    return [ttnn.to_torch(c).to(torch.bfloat16).contiguous().view(torch.int16) for c in _chips(dev, t)]


def _ndiff(a, b):
    return sum(int((x != y).sum()) for x, y in zip(a, b))


def selftest(norm, cells, seed=4242, gather=None):
    """Fused vs the image's three ops (norm._distributed_rmsnorm_image), same gather, random per-chip h and gamma.
    cells: [(M, W, out_dtype, mc)]. Returns {name: "exact" | "DIFF n", name_mustfail: n (must be > 0)}."""
    import torch

    dev = norm.device
    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    nd = _num_devices(dev)
    g = torch.Generator().manual_seed(seed)
    res = {}
    for M, W, od, mc in cells:
        name = f"M{M}_W{W}_{str(od).split('.')[-1]}_{'L1' if mc.buffer_type == ttnn.BufferType.L1 else 'DRAM'}"
        Wfull = W * nd
        h = torch.randn(1, 1, M, Wfull, generator=g) * 2 + torch.randn(1, 1, M, 1, generator=g)
        gm = torch.rand(1, 1, Wfull // TILE, TILE, generator=g) * 2
        kw_h = {"mesh_mapper": ttnn.ShardTensorToMesh(dev, dim=3)} if is_mesh else {}
        kw_g = {"mesh_mapper": ttnn.ShardTensor2dMesh(dev, dims=(None, 2), mesh_shape=list(dev.shape))} if is_mesh else {}
        x = ttnn.from_torch(h.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw_h)
        gamma = ttnn.from_torch(gm.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw_g)
        ref = norm._distributed_rmsnorm_image(x, epsilon=norm.eps, weight=gamma,
                                               compute_kernel_config=norm.compute_kernel_config_hifi2,
                                               output_memory_config=mc, output_dtype=od)
        cand = run(norm, x, norm.eps, gamma, mc, od, mf=mustfail(), gather=gather)
        bad = run(norm, x, norm.eps, gamma, mc, od, mf=True, gather=gather)
        r, c, w = _bits(dev, ref), _bits(dev, cand), _bits(dev, bad)
        n = _ndiff(r, c)
        res[name] = "exact" if n == 0 else f"DIFF {n}"
        res[name + "_mustfail"] = _ndiff(r, w)
        for t in (x, gamma, ref, cand, bad):
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return all(v == "exact" for k, v in res.items() if not k.endswith("_mustfail")) and \
        all(v > 0 for k, v in res.items() if k.endswith("_mustfail"))


def ensure_selftest(norm, cells, W):
    """Once per process per (M, W, dtype): fail closed, then record the cells the fused path may serve.
    cells: [(M, out_dtype, memory_config)]."""
    if not enabled():
        return
    gx, gy = _grid(norm.device)
    todo, seen = [], set()
    for M, od, mc in cells:
        k = (int(M), str(od), str(mc.buffer_type))
        if int(M) % TILE == 0 and int(M) // TILE <= gx * gy and k not in seen and key(M, W, od) not in _tested:
            seen.add(k)
            todo.append((int(M), int(W), od, mc))
    if not todo:
        return
    res = selftest(norm, todo)
    if not selftest_ok(res):
        raise RuntimeError(f"QWEN36_PREFILL_DNORM_FUSE self-test REFUSED{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}")
    for M, W_, od, _ in todo:
        _tested[key(M, W_, od)] = "exact"
    logger.info(f"{SELFTEST_BANNER}: {len(todo)} cells M={sorted(set(c[0] for c in todo))} W={W} "
                f"dtypes={sorted(set(str(c[2]).split('.')[-1] for c in todo))}, bitwise vs the image's pre/all-gather/post "
                f"on every chip; wrong-input branch (gamma tile +1) differs in every cell")


def selftest_add(norm, cells, seed=4343, gather=None):
    """run_add vs ttnn.add + the image's three norm ops, random per-chip a/b/gamma. cells: [(M, W, out_dtype, mc, b_dtype)].
    Returns {name_h: "exact"|"DIFF n", name_y: ..., name_mustfail: n (> 0 required)}."""
    import torch

    dev = norm.device
    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    nd = _num_devices(dev)
    g = torch.Generator().manual_seed(seed)
    res = {}
    for M, W, od, mc, bd in cells:
        name = (f"M{M}_W{W}_{str(od).split('.')[-1]}_b{str(bd).split('.')[-1]}_"
                f"{'L1' if mc.buffer_type == ttnn.BufferType.L1 else 'DRAM'}")
        Wf = W * nd
        kw = {"mesh_mapper": ttnn.ShardTensorToMesh(dev, dim=3)} if is_mesh else {}
        kwg = {"mesh_mapper": ttnn.ShardTensor2dMesh(dev, dims=(None, 2), mesh_shape=list(dev.shape))} if is_mesh else {}
        a = ttnn.from_torch((torch.randn(1, 1, M, Wf, generator=g) * 2).to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        b = ttnn.from_torch(torch.randn(1, 1, M, Wf, generator=g), dtype=bd, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        gamma = ttnn.from_torch((torch.rand(1, 1, Wf // TILE, TILE, generator=g) * 2).to(torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwg)
        rh = ttnn.add(a, b)
        ry = norm._distributed_rmsnorm_image(rh, epsilon=norm.eps, weight=gamma,
                                              compute_kernel_config=norm.compute_kernel_config_hifi2,
                                              output_memory_config=mc, output_dtype=od)
        ch, cy = run_add(norm, a, b, norm.eps, gamma, mc, od, mf=add_mustfail(), gather=gather)
        wh, wy = run_add(norm, a, b, norm.eps, gamma, mc, od, mf=True, gather=gather)
        res[name + "_dtype"] = "exact" if (ch.dtype == rh.dtype and ch.memory_config() == rh.memory_config()) else \
            f"DIFF {ch.dtype}/{rh.dtype} {ch.memory_config()}/{rh.memory_config()}"
        n = _ndiff(_bits(dev, rh), _bits(dev, ch))
        res[name + "_h"] = "exact" if n == 0 else f"DIFF {n}"
        n = _ndiff(_bits(dev, ry), _bits(dev, cy))
        res[name + "_y"] = "exact" if n == 0 else f"DIFF {n}"
        res[name + "_mustfail"] = _ndiff(_bits(dev, ry), _bits(dev, wy))
        for t in (a, b, gamma, rh, ry, ch, cy, wh, wy):
            ttnn.deallocate(t)
    return res


def ensure_selftest_add(norm, cells, W):
    """Once per process per (M, W, out dtype, b dtype); fail closed. cells: [(M, out_dtype, mc, b_dtype)]."""
    if not add_enabled():
        return
    gx, gy = _grid(norm.device)
    todo, seen = [], set()
    for M, od, mc, bd in cells:
        k = ("add",) + key(M, W, od) + (str(bd),)
        if int(M) % TILE == 0 and int(M) // TILE <= gx * gy and k not in seen and k not in _tested:
            seen.add(k)
            todo.append((int(M), int(W), od, mc, bd))
    if not todo:
        return
    res = selftest_add(norm, todo)
    if not selftest_ok(res):
        raise RuntimeError(f"QWEN36_PREFILL_DNORM_ADD self-test REFUSED"
                           f"{' (' + ADD_MUSTFAIL_ENV + '=1: expected)' if add_mustfail() else ''}: {res}")
    for M, W_, od, _, bd in todo:
        _tested[("add",) + key(M, W_, od) + (str(bd),)] = "exact"
    logger.info(f"{ADD_SELFTEST_BANNER}: {len(todo)} cells M={sorted(set(c[0] for c in todo))} W={W} "
                f"b={sorted(set(str(c[4]).split('.')[-1] for c in todo))}, h and y bitwise vs ttnn.add + the image norm "
                f"on every chip; wrong-input branch (gamma tile +1) differs in every cell")
