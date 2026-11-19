# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFILL_GDN_TAIL=1: the GDN prefill tail's per-head RMSNorm and head->token relayout as ONE program.

Why (measured, [source validation record], CONTROL dfr, 32k): after the fused
chunk scan, every GDN layer runs rms_norm on the head-major o [24, T, 128] fp32 (72.6 us), then nlp_concat_heads
(54.8 us) into the token-major [1, 1, T, 3072] fp32 the gate multiply reads; 48 layers = 6.1 ms per 2048-token chunk.
The relayout is pure addressing: head h, token tile r, column tile j of o goes to tile r * 96 + h * 4 + j.

What (one ttnn.generic_op, JIT kernels in kernels/gdn_tail/, no image change): each core takes a run of (head, token
tile) rows, reads the row's 4 fp32 tiles, runs the image layernorm.cpp RMSNORM+gamma LLK sequence on them (rms_norm's
default config: HiFi4, approx, fp32 DEST off), and its writer stores the 4 output tiles at their token-major places.
Same LLK calls, CB formats, fidelity and DEST mode as ttnn.rms_norm, so the same bytes; nlp_concat_heads moves tiles
unchanged. Moved cost: none expected (the output is the same L1 interleaved fp32 tensor the gate multiply reads today).

Applies to: gdn/tp.py forward_prefill's _gdn_fuse_out branch (head-major o, TILE fp32 interleaved) when (T, NH, D)
passed the self-test this process. Everything else: today's two ops (logged once).

Self-test (fail closed, ensure_selftest, eager, before any trace): random per-chip o and norm weight, the fused program
vs ttnn.rms_norm + nlp_concat_heads, fp32 bitwise on every chip. Wrong-input branch at every boot: gamma tile t <- tile
(t + 1) MUST differ. QWEN36_PREFILL_GDN_TAIL_MUSTFAIL=1 serves that shift: the comparison differs and the boot is REFUSED.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
import struct
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_PREFILL_GDN_TAIL"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")
if _MF == "1" and _MODE != "1":
    raise RuntimeError(f"{MUSTFAIL_ENV}=1 needs {ENV}=1")
# QWEN36_PREFILL_GDN_TAIL_DM=1 (pp-glue3): data-movement tuning of both programs, same compute, so bitwise the same
# outputs: the gate reader takes DM_GATE_RB tiles per read barrier (CBs 2 * DM_GATE_RB deep), the norm reader two rows
# per barrier (CB_IN four rows deep), and both writers flush instead of waiting for write acks per block (one ack
# barrier at the end). Off: no define, today's kernels and CB sizes exactly.
DM_ENV = ENV + "_DM"
_DM = _tt_public_getenv(DM_ENV, "0")
if _DM not in ("0", "1"):
    raise RuntimeError(f"{DM_ENV} must be 0 or 1; got {_DM!r}")
if _DM == "1" and _MODE != "1":
    raise RuntimeError(f"{DM_ENV}=1 needs {ENV}=1")
DM_GATE_RB = 4


def dm():
    return _DM == "1"

SELFTEST_BANNER = "QWEN36_PREFILL_GDN_TAIL self-test exact"
TILE = 32
_KDIR = Path(__file__).resolve().parent / "kernels" / "gdn_tail"
_SRC = {}
_tested = {}
_logged = set()

# Workbench-only bypass (tools/ttx/clients/wb_knobs.py), read per call: "1" serves today's ops on the same boot.
WB_BYPASS_ENV = "QWEN36_PREFILL_GDN_TAIL_WB_BYPASS"


def enabled():
    return _MODE == "1" and _tt_public_getenv(WB_BYPASS_ENV, "0") != "1"


def mustfail():
    return _MF == "1"


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _shape(t):
    return [int(x) for x in t.shape]


def key(T, NH, D):
    return (int(T), int(NH), int(D))


def _grid(dev):
    g = dev.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def _check(o, NH):
    s = _shape(o)
    if len(s) != 3 or s[0] != NH:
        raise RuntimeError(f"{ENV}: o shape {s}, expected [{NH}, T, D] head-major")
    T, D = s[1], s[2]
    if T % TILE or D % TILE or D // TILE > 8:
        raise RuntimeError(f"{ENV}: T={T} D={D} not tile multiples (or D > 256)")
    if o.dtype != ttnn.float32 or o.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: o {o.dtype} {o.layout}, expected fp32 TILE")
    if o.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: o must be interleaved, got {o.memory_config()}")
    return T, D


def usable(o, NH):
    """True when the flag is on, o has the fused path's shape, and its (T, NH, D) passed the self-test."""
    if not enabled():
        return False
    try:
        T, D = _check(o, NH)
    except RuntimeError as e:
        k = ("shape", str(e))
        if k not in _logged:
            _logged.add(k)
            logger.warning(f"QWEN36_PREFILL_GDN_TAIL fallback: {e}; rms_norm + nlp_concat_heads serve it")
        return False
    if key(T, NH, D) in _tested:
        return True
    k = ("untested", key(T, NH, D))
    if k not in _logged:
        _logged.add(k)
        logger.warning(f"QWEN36_PREFILL_GDN_TAIL fallback: T={T} NH={NH} D={D} not self-tested this process; "
                       f"rms_norm + nlp_concat_heads serve it")
    return False


def _split(dev, R):
    """[(core, row0, nrows)] over the whole compute grid, row-major, the first R % n cores one row longer."""
    gx, gy = _grid(dev)
    n = min(R, gx * gy)
    q, rem = divmod(R, n)
    out, r0 = [], 0
    for i in range(n):
        k = q + (1 if i < rem else 0)
        out.append((ttnn.CoreCoord(i % gx, i // gx), r0, k))
        r0 += k
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(gx, gy), True)
    return out, crs


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cfg_norm():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4  # ttnn rmsnorm_default_compute_config
    c.fp32_dest_acc_en = False
    c.math_approx_mode = True
    return c


def _cb(idx, n_pages, page, fmt, crs):
    return ttnn.CBDescriptor(total_size=n_pages * page, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])


def _kern(name, crs, ct, rt, cfg, defines=None):
    return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                 core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg,
                                 defines=list((defines or {}).items()))


def _dm_defs(v):
    return {"GTL_DM": v} if dm() else None


def norm_concat(o, gamma, eps, NH, out_mc, mf=False, out=None):
    """o [NH, T, D] fp32 head-major -> rms_norm(o, gamma, eps) relaid token-major as [1, 1, T, NH * D] fp32 in out_mc."""
    T, D = _check(o, NH)
    dev = o.device()
    WT, TT = D // TILE, T // TILE
    R = NH * TT
    plan, crs = _split(dev, R)
    if out is None:
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, NH * D]), ttnn.float32, ttnn.TILE_LAYOUT, dev, out_mc)
    CB_IN, CB_SC, CB_EPS, CB_G, CB_X2, CB_EX2, CB_EX2PE, CB_FU, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 7, 8
    gfmt = gamma.dtype
    gsz = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[gfmt]
    cbs = [_cb(CB_IN, (4 if dm() else 2) * WT, 4096, ttnn.float32, crs),
           _cb(CB_SC, 1, 2048, ttnn.bfloat16, crs),  # image: scaler Float16_b
           _cb(CB_EPS, 1, 2048, ttnn.bfloat16, crs),
           _cb(CB_G, WT, gsz, gfmt, crs),
           _cb(CB_X2, WT, 2048, ttnn.bfloat16, crs),  # interm Float16_b (fp32 DEST off)
           _cb(CB_EX2, 1, 2048, ttnn.bfloat16, crs),
           _cb(CB_EX2PE, 1, 2048, ttnn.bfloat16, crs),
           _cb(CB_FU, WT, 2048, ttnn.bfloat16, crs),
           _cb(CB_OUT, 2 * WT, 4096, ttnn.float32, crs)]
    acc = _acc(o) + _acc(gamma) + _acc(out)
    eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
    rt0, rt1, rtc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    oa, ga, ya = o.buffer_address(), gamma.buffer_address(), out.buffer_address()
    for c, r0, k in plan:
        a = [oa, ga, ya, r0, k, 1 if mf else 0, eps_bits]
        rt0[c.x][c.y] = a
        rt1[c.x][c.y] = a
        rtc[c.x][c.y] = [k]
    ct = lambda role: [CB_IN, CB_SC, CB_EPS, CB_G, CB_OUT, WT, TT, NH, role] + acc  # noqa: E731
    desc = ttnn.ProgramDescriptor(
        kernels=[_kern("norm_dm.cpp", crs, ct(0), rt0, ttnn.ReaderConfigDescriptor(), _dm_defs("1")),
                 _kern("norm_dm.cpp", crs, ct(1), rt1, ttnn.WriterConfigDescriptor(), _dm_defs("1")),
                 _kern("norm_compute.cpp", crs,
                       [CB_IN, CB_SC, CB_EPS, CB_G, CB_X2, CB_EX2, CB_EX2PE, CB_FU, CB_OUT, WT, D], rtc, _cfg_norm())],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([o, gamma, out], desc)
    return out


def _cfg_gate(cbs_fp32):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4  # binary_ng / unary defaults
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    modes = [ttnn.UnpackToDestMode.Default] * 64  # NUM_CIRCULAR_BUFFERS (the JIT build checks the length)
    for i in cbs_fp32:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    c.unpack_to_dest_mode = modes
    return c


GATE_NPER = 2  # tiles per DEST acquire (2 DEST slots each; fp32 DEST half-sync holds 4)


def _check_gate(a, z, z_c0):
    sa, sz = _shape(a), _shape(z)
    if len(sa) != 3 or len(sz) != 3 or sa[:2] != sz[:2] or sa[1] % TILE or sa[2] % TILE or sz[2] % TILE:
        raise RuntimeError(f"{ENV}: gate a {sa} / z {sz}, expected [1, T, C] / [1, T, >= C]")
    if sz[2] < z_c0 + sa[2]:
        raise RuntimeError(f"{ENV}: z {sz} too narrow for column offset {z_c0} + {sa[2]}")
    if a.dtype != ttnn.float32 or z.dtype != ttnn.bfloat16 or a.layout != ttnn.TILE_LAYOUT or z.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: gate a {a.dtype} z {z.dtype}, expected fp32 / bf16 TILE")
    for t in (a, z):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise RuntimeError(f"{ENV}: gate inputs must be interleaved, got {t.memory_config()}")
    return sa[1], sa[2]


def gate(a, z, out_mc, mf=False, z_c0=0, out=None):
    """out = a * silu(z) as [1, T, C] bf16 in out_mc, z read at column offset z_c0 (z may be wider than a)."""
    T, C = _check_gate(a, z, z_c0)
    dev = a.device()
    TT, C_T, Z_WT = T // TILE, C // TILE, _shape(z)[2] // TILE
    N = TT * C_T
    gx, gy = _grid(dev)
    n = min(N, gx * gy)
    q, rem = divmod(N, n)
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(gx, gy), True)
    if out is None:
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, out_mc)
    CB_A, CB_Z, CB_OUT = 0, 1, 2
    _d = 2 * (DM_GATE_RB if dm() else GATE_NPER)
    cbs = [_cb(CB_A, _d, 4096, ttnn.float32, crs),
           _cb(CB_Z, _d, 2048, ttnn.bfloat16, crs),
           _cb(CB_OUT, _d, 2048, ttnn.bfloat16, crs)]
    acc = _acc(a) + _acc(z) + _acc(out)
    rt0, rt1, rtc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    aa, za, ya = a.buffer_address(), z.buffer_address(), out.buffer_address()
    t0 = 0
    for i in range(n):
        k = q + (1 if i < rem else 0)
        c = ttnn.CoreCoord(i % gx, i // gx)
        rt0[c.x][c.y] = [aa, za, ya, t0, k]
        rt1[c.x][c.y] = [aa, za, ya, t0, k]
        rtc[c.x][c.y] = [k]
        t0 += k
    ct = lambda role: [CB_A, CB_Z, CB_OUT, GATE_NPER, C_T, Z_WT, z_c0 // TILE, role] + acc  # noqa: E731
    desc = ttnn.ProgramDescriptor(
        kernels=[_kern("gate_dm.cpp", crs, ct(0), rt0, ttnn.ReaderConfigDescriptor(), _dm_defs(str(DM_GATE_RB))),
                 _kern("gate_dm.cpp", crs, ct(1), rt1, ttnn.WriterConfigDescriptor(), _dm_defs(str(DM_GATE_RB))),
                 _kern("gate_compute.cpp", crs, [CB_A, CB_Z, CB_OUT, GATE_NPER], rtc, _cfg_gate([CB_A, CB_Z]),
                       {"GATE_MUSTFAIL": "1" if mf else "0"})],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([a, z, out], desc)
    return out


def reference_gate(a, z, out_mc):
    """Today's gdn/tp.py _silu_mul(out_f, z, L1, dtype=bf16) (QWEN36_FAST_MUL off for M > 32)."""
    s = ttnn.silu(z, memory_config=out_mc)
    y = ttnn.multiply(a, s, memory_config=out_mc, dtype=ttnn.bfloat16)
    ttnn.deallocate(s)
    return y


def _log_dm(T):
    """The DM banner, once per T, from both served entry points (log_engaged from gdn/tp.py, run)."""
    if dm() and ("dm", T) not in _logged:
        _logged.add(("dm", T))
        logger.info(f"QWEN36_PREFILL_GDN_TAIL_DM engaged: T={T}: gate reader {DM_GATE_RB} tiles per barrier, norm "
                    f"reader 2 rows per barrier, writers flush per block (one ack barrier at the end)")


def log_engaged(T, NH, D):
    k = ("engaged", key(T, NH, D))
    if k not in _logged:
        _logged.add(k)
        _log_dm(T)
        logger.info(f"QWEN36_PREFILL_GDN_TAIL engaged: T={T} NH={NH} D={D}: per-head rms_norm + head->token relayout "
                    f"in one program, silu(z) * out in one program{' MUSTFAIL' if mustfail() else ''}")


def gate_usable(a, z):
    """The fused gate serves only today's accurate (SFPU) multiply on a self-tested T: QWEN36_FAST_MUL's FPU
    multiply at this M, or any other shape / dtype, falls back to _silu_mul (logged once)."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    T = _shape(a)[-2]
    why = None
    if tpc.fast_mul("gdn_gate", T):
        why = f"QWEN36_FAST_MUL covers M={T}"
    else:
        try:
            _check_gate(a, z, 0)
        except RuntimeError as e:
            why = str(e)
    if why is None and ("gate", T) not in _tested:
        why = f"gate T={T} not self-tested this process"
    if why is None:
        return True
    if ("gate_fb", why) not in _logged:
        _logged.add(("gate_fb", why))
        logger.warning(f"QWEN36_PREFILL_GDN_TAIL gate fallback: {why}; silu + multiply serve it")
    return False


def reference(o, gamma, eps, NH, out_mc):
    """Today's ops, exactly as gdn/tp.py forward_prefill's _gdn_fuse_out branch runs them."""
    T, D = _shape(o)[1], _shape(o)[2]
    n = ttnn.rms_norm(o, weight=gamma, epsilon=eps, memory_config=out_mc)
    n = ttnn.reshape(n, (1, NH, T, D))
    y = ttnn.experimental.nlp_concat_heads(n, memory_config=out_mc)
    ttnn.deallocate(n)
    return y


def run(o, gamma, eps, NH, out_mc):
    """The served call: [1, 1, T, NH * D] fp32 in out_mc. Refuses an untested (T, NH, D)."""
    T, D = _check(o, NH)
    k = key(T, NH, D)
    if k not in _tested:
        raise RuntimeError(f"{ENV}: T={T} NH={NH} D={D} was not self-tested this process; refusing to serve it")
    mf = mustfail()
    if ("engaged", k) not in _logged:
        _logged.add(("engaged", k))
        logger.info(f"QWEN36_PREFILL_GDN_TAIL engaged: T={T} NH={NH} D={D}: per-head rms_norm + head->token relayout "
                    f"in one program{' MUSTFAIL gamma tile +1' if mf else ''}")
        _log_dm(T)
    return norm_concat(o, gamma, eps, NH, out_mc, mf=mf)


# ---------------------------------------------------------------------------------------------- self-test
def _chips(dev, t):
    return ttnn.get_device_tensors(t) if hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice) else [t]


def _bits32(dev, t):
    import torch

    return [ttnn.to_torch(c).to(torch.float32).contiguous().view(torch.int32) for c in _chips(dev, t)]


def _ndiff(a, b):
    return sum(int((x != y).sum()) for x, y in zip(a, b))


def selftest(dev, cells, eps=1e-6, seed=5151, out_mc=None):
    """Fused vs rms_norm + nlp_concat_heads, random per-chip o and gamma. cells: [(T, NH, D)].
    Returns {name: "exact" | "DIFF n", name_mustfail: n (must be > 0)}."""
    import torch

    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    nd = int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1
    # DRAM: placement does not change a byte (ttsim ran both), and at boot (set_paged_kv_caches) L1 already holds the
    # KV-time buffers: L1 outputs at T=2048 clashed with nlp_concat_heads' static CBs (card 2026-10-19 18:51Z)
    mc = out_mc or ttnn.DRAM_MEMORY_CONFIG
    g = torch.Generator().manual_seed(seed)
    res = {}
    for T, NH, D in cells:
        name = f"T{T}_NH{NH}_D{D}"
        # scan outputs: per-head scale spread (some heads near zero), per-chip different
        o = torch.randn(NH * nd, T, D, generator=g) * (torch.rand(NH * nd, 1, 1, generator=g) * 3 + 0.01)
        gm = torch.rand(1, 1, D, generator=g) * 2
        kw_o = {"mesh_mapper": ttnn.ShardTensorToMesh(dev, dim=0)} if is_mesh else {}
        kw_g = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if is_mesh else {}
        x = ttnn.from_torch(o, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw_o)
        gamma = ttnn.from_torch(gm.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw_g)
        ref = reference(x, gamma, eps, NH, mc)
        cand = norm_concat(x, gamma, eps, NH, mc, mf=mustfail())
        bad = norm_concat(x, gamma, eps, NH, mc, mf=True)
        r, c, w = _bits32(dev, ref), _bits32(dev, cand), _bits32(dev, bad)
        n = _ndiff(r, c)
        res[name] = "exact" if n == 0 else f"DIFF {n}"
        res[name + "_mustfail"] = _ndiff(r, w)
        for t in (x, gamma, cand, bad):
            ttnn.deallocate(t)
        # gate: out * silu(z) on the norm output (fp32 [1, T, C]) and a random bf16 z with saturating extremes
        C = NH * D
        a = ttnn.reshape(ref, (1, T, C))
        zz = torch.randn(1, T, C * nd, generator=g) * 3
        zz[0, :TILE, :8] = torch.tensor([-200., -90., -88., -30., 30., 88., 90., 200.])
        kw_z = {"mesh_mapper": ttnn.ShardTensorToMesh(dev, dim=2)} if is_mesh else {}
        z = ttnn.from_torch(zz.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw_z)
        gref = reference_gate(a, z, mc)
        gcand = gate(a, z, mc, mf=mustfail())
        gbad = gate(a, z, mc, mf=True)
        r, c, w = _bits32(dev, gref), _bits32(dev, gcand), _bits32(dev, gbad)
        n = _ndiff(r, c)
        res["gate_" + name] = "exact" if n == 0 else f"DIFF {n}"
        res["gate_" + name + "_mustfail"] = _ndiff(r, w)
        for t in (ref, z, gref, gcand, gbad):
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return all(v == "exact" for k, v in res.items() if not k.endswith("_mustfail")) and \
        all(v > 0 for k, v in res.items() if k.endswith("_mustfail"))


def ensure_selftest(dev, cells):
    """Once per process per (T, NH, D): fail closed, then record the cells the fused path may serve."""
    if _MODE != "1":
        return
    todo = sorted({key(*c) for c in cells if key(*c) not in _tested and int(c[0]) % TILE == 0})
    if not todo:
        return
    res = selftest(dev, todo)
    if not selftest_ok(res):
        raise RuntimeError(f"QWEN36_PREFILL_GDN_TAIL self-test REFUSED"
                           f"{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}")
    for k in todo:
        _tested[k] = "exact"
        _tested[("gate", k[0])] = "exact"
    logger.info(f"{SELFTEST_BANNER}: {len(todo)} cells T={sorted(set(c[0] for c in todo))} "
                f"NH={sorted(set(c[1] for c in todo))} D={sorted(set(c[2] for c in todo))}, fp32 bitwise vs rms_norm + "
                f"nlp_concat_heads and bf16 bitwise out-gate vs silu + multiply on every chip; wrong-input branches (gamma tile "
                f"+1, gate without silu) differ in every cell")
