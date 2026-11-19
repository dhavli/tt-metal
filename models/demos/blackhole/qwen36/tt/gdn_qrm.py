# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFILL_GDN_QKV_RM=1: the GDN prefill's qkv slice + to_layout(ROW_MAJOR) as ONE program.

Why (measured, [source validation record], 32k): every GDN layer slices qkv
[1, T, 5120] out of the fused qkvzab [1, T, 8248] (TILE, L1; one of the 144 slice calls, 4.3 ms/chunk) and the kda
conv then untilizes it to a row-major DRAM copy (to_layout, 101 us/call, 4.9 ms/chunk over 48 layers). The kda conv
only ever reads the row-major copy, so the TILE qkv is a pure intermediate.

What (one ttnn.generic_op, JIT kernels in kernels/gdn_qrm/, no image change): each core takes a run of (token tile
row, group of NB column tiles) blocks of the first C/32 tile columns of qkvzab, untilizes them with the image's
untilize helper (the pack_untilize LLK to_layout runs) and writes each of the 32 rows straight into its token row of
the row-major DRAM output. bf16 in, bf16 out: a data move, so the bytes equal slice + to_layout's.

Applies to: gdn/tp.py forward_prefill's QWEN36_GDN_CONV_KDA full-chunk path (no gdn_masks, valid_len None), when T
passed the self-test this process. The masked-bucket paths keep the TILE qkv (they read it). Everything else: today's.

Self-test (fail closed, ensure_selftest, eager, before any trace): random per-chip qkvzab in L1, the fused program
vs ttnn.slice + ttnn.to_layout, bitwise on every chip. Wrong-input branch at every boot: column group g reads group
(g + 1) MUST differ. QWEN36_PREFILL_GDN_QKV_RM_MUSTFAIL=1 serves that shift and the boot is REFUSED.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_PREFILL_GDN_QKV_RM"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
WB_BYPASS_ENV = ENV + "_WB_BYPASS"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")
if _MF == "1" and _MODE != "1":
    raise RuntimeError(f"{MUSTFAIL_ENV}=1 needs {ENV}=1")

SELFTEST_BANNER = "QWEN36_PREFILL_GDN_QKV_RM self-test exact"
TILE = 32
NB = 8  # tiles per untilize block (one pack_untilize pass at bf16)
_KDIR = Path(__file__).resolve().parent / "kernels" / "gdn_qrm"
_SRC = {}
_tested = {}
_logged = set()


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


def _check(src, C):
    s = _shape(src)
    if len(s) != 3 or s[0] != 1 or s[1] % TILE or C % (TILE * NB) or s[2] < C:
        raise RuntimeError(f"{ENV}: src {s}, C={C}: expected [1, T, >= C], T and C/{TILE * NB} whole")
    if src.dtype != ttnn.bfloat16 or src.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: src {src.dtype} {src.layout}, expected bf16 TILE")
    if src.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: src must be interleaved, got {src.memory_config()}")
    return s[1]


def usable(src, C):
    """True when the flag is on, src fits, and its T passed the self-test this process (else logged once)."""
    if not enabled():
        return False
    try:
        T = _check(src, C)
    except RuntimeError as e:
        if ("shape", str(e)) not in _logged:
            _logged.add(("shape", str(e)))
            logger.warning(f"QWEN36_PREFILL_GDN_QKV_RM fallback: {e}; slice + to_layout serve it")
        return False
    if (T, C) in _tested:
        return True
    if ("untested", T, C) not in _logged:
        _logged.add(("untested", T, C))
        logger.warning(f"QWEN36_PREFILL_GDN_QKV_RM fallback: T={T} C={C} not self-tested this process; slice + "
                       f"to_layout serve it")
    return False


def _kern(name, crs, ct, rt, cfg):
    return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                 core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)


def qkv_rm(src, C, mf=False, out=None):
    """Row-major [1, T, C] bf16 DRAM copy of src[:, :, :C] (src: [1, T, W >= C] bf16 TILE interleaved)."""
    T = _check(src, C)
    dev = src.device()
    TT, C_T = T // TILE, C // TILE
    G = C_T // NB
    W_T = -(-_shape(src)[2] // TILE)
    nblk = TT * G
    g = dev.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
    n = min(nblk, gx * gy)
    q, rem = divmod(nblk, n)
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(gx, gy), True)
    if out is None:
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, T, C]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev,
                                             ttnn.DRAM_MEMORY_CONFIG)
    CB_IN, CB_OUT = 0, 16
    cbs = [ttnn.CBDescriptor(total_size=2 * NB * 2048, core_ranges=crs, format_descriptors=[
               ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=ttnn.bfloat16, page_size=2048)]),
           ttnn.CBDescriptor(total_size=2 * NB * 2048, core_ranges=crs, format_descriptors=[
               ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.bfloat16, page_size=2048)])]
    acc = list(ttnn.TensorAccessorArgs(src).get_compile_time_args()) + \
        list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rt0, rt1, rtc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    sa, da = src.buffer_address(), out.buffer_address()
    b0 = 0
    for i in range(n):
        k = q + (1 if i < rem else 0)
        c = ttnn.CoreCoord(i % gx, i // gx)
        rt0[c.x][c.y] = [sa, da, b0, k, 1 if mf else 0]
        rt1[c.x][c.y] = [sa, da, b0, k, 0]
        rtc[c.x][c.y] = [k]
        b0 += k
    ct = lambda role: [CB_IN, CB_OUT, NB, G, W_T, role] + acc  # noqa: E731
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = False

    desc = ttnn.ProgramDescriptor(
        kernels=[_kern("qrm_dm.cpp", crs, ct(0), rt0, ttnn.ReaderConfigDescriptor()),
                 _kern("qrm_dm.cpp", crs, ct(1), rt1, ttnn.WriterConfigDescriptor()),
                 _kern("qrm_compute.cpp", crs, [CB_IN, CB_OUT, NB], rtc, cfg)],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([src, out], desc)
    return out


def reference(src, C, mid_mc=None):
    """Today's ops: gdn/tp.py _split_qkvzab's qkv slice (L1 when served) then _conv_kda_prefill's to_layout (DRAM row
    major). mid_mc: the slice's placement (the self-test uses DRAM: placement does not change a byte, and boot-time L1
    holds the KV buffers)."""
    T = _shape(src)[1]
    qkv = ttnn.slice(src, (0, 0, 0), (1, T, C), memory_config=mid_mc or ttnn.L1_MEMORY_CONFIG)
    y = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(qkv)
    return y


def run(src, C):
    """The served call; refuses an untested (T, C)."""
    T = _check(src, C)
    if (T, C) not in _tested:
        raise RuntimeError(f"{ENV}: T={T} C={C} was not self-tested this process; refusing to serve it")
    if ("engaged", T) not in _logged:
        _logged.add(("engaged", T))
        logger.info(f"QWEN36_PREFILL_GDN_QKV_RM engaged: T={T} C={C}: qkv slice + untilize in one program"
                    f"{' MUSTFAIL column group +1' if mustfail() else ''}")
    return qkv_rm(src, C, mf=mustfail())


def selftest(dev, cells, seed=4242):
    """cells: [(T, C, W)]. Random qkvzab per chip in L1; fused vs reference, bitwise. -> {name: 'exact'|'DIFF n',
    name_mustfail: n}."""
    import torch

    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    nd = int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1
    chips = (lambda t: ttnn.get_device_tensors(t)) if is_mesh else (lambda t: [t])  # noqa: E731
    bits = lambda t: [ttnn.to_torch(c).to(torch.bfloat16).contiguous().view(torch.int16) for c in chips(t)]  # noqa
    g = torch.Generator().manual_seed(seed)
    res = {}
    for T, C, W in cells:
        name = f"T{T}_C{C}_W{W}"
        x = (torch.randn(nd, T, W, generator=g) * 3).to(torch.bfloat16)
        kw = {"mesh_mapper": ttnn.ShardTensorToMesh(dev, dim=0)} if is_mesh else {}
        # src L1 (the served placement) up to T=512, DRAM above: both accessor paths are covered, the bytes do not
        # depend on placement, and a 34 MB L1 source at T=2048 trips ttsim's 2-chip SRAM model even for stock slice
        # ([source validation record]) and would compete with boot-time L1.
        src_mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        src = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=src_mc, **kw)
        ref = reference(src, C, ttnn.DRAM_MEMORY_CONFIG)
        cand = qkv_rm(src, C, mf=mustfail())
        bad = qkv_rm(src, C, mf=True)
        r, c, w = bits(ref), bits(cand), bits(bad)
        n = sum(int((a != b).sum()) for a, b in zip(r, c))
        ok_shape = all(_shape(t) == [1, T, C] and t.layout == ttnn.ROW_MAJOR_LAYOUT for t in (ref, cand))
        res[name] = "exact" if n == 0 and ok_shape else f"DIFF {n} shape_ok={ok_shape}"
        res[name + "_mustfail"] = sum(int((a != b).sum()) for a, b in zip(r, w))
        for t in (src, ref, cand, bad):
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return all(v == "exact" for k, v in res.items() if not k.endswith("_mustfail")) and \
        all(v > 0 for k, v in res.items() if k.endswith("_mustfail"))


def ensure_selftest(dev, cells):
    """Once per process per (T, C): fail closed. cells: [(T, C, W)]."""
    if not enabled():
        return
    todo = sorted({(int(T), int(C), int(W)) for T, C, W in cells if (int(T), int(C)) not in _tested
                   and int(T) % TILE == 0})
    if not todo:
        return
    res = selftest(dev, todo)
    if not selftest_ok(res):
        raise RuntimeError(f"QWEN36_PREFILL_GDN_QKV_RM self-test REFUSED"
                           f"{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}")
    for T, C, _ in todo:
        _tested[(T, C)] = "exact"
    logger.info(f"{SELFTEST_BANNER}: {len(todo)} cells T={sorted(set(c[0] for c in todo))} C={todo[0][1]} "
                f"W={todo[0][2]}, bitwise vs slice + to_layout(ROW_MAJOR) on every chip; wrong-input branch (column "
                f"group +1) differs in every cell")
