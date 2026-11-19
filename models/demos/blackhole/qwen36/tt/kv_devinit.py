# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_KV_DEVICE_INIT=1: allocate the paged KV pool on the device and zero it there.

Why (measured, docs/analysis/boot-speed-20261003.md item 3): on the BEST B=8 boot the KV pool takes 65.1 s
(c167 control server log: `QWEN36_KV_DTYPE engaged` 00:40:35.912 -> `MTP paged KV` 00:41:41.000), 34 calls of
`ttnn.as_tensor(torch.zeros([12297, 2, 64, 256]), bf8|bf4, ReplicateTensorToMesh)`: 16 attention layers x (K, V)
plus the MTP head's (K, V). Each call builds an 806 MB bf16 host tensor, converts, tilizes and packs it to bfp on
the 4 host cores and uploads it to both chips: ~1.9 s each, CPU-bound, and a multi-GB transient on a RAM-starved host.

Here: `ttnn.empty` with the same shape, dtype, TILE layout, DRAM interleaved memory config and mesh (no host tensor,
the same allocator calls in the same order, so the same addresses), then ONE generic_op program per cache whose
writer stores a 32-bit fill word over every page from L1 (kernels/kv_devinit/fill_writer.cpp). The fill word is 0:
host packing of torch.zeros gives all-zero bfp8/bfp4 tiles (shared exponent 0, mantissa/sign 0; proven in ttsim
and re-checked against the host packer at every boot by `selftest`), so the device bytes equal today's.

Self-test (fail closed, every boot, right after allocation and before any warm-up writes the pool): sampled whole
blocks (first, second, middle, last) of the first, middle and last attention layer's K and V and of the MTP K/V are
copied byte for byte into a uint32 buffer (kernels/kv_devinit/page_gather.cpp) and compared on EVERY chip with the
raw device bytes of a host-packed zero tile of the same dtype. The wrong-input branch compares the same pages with a
host-packed NON-zero tile; it must differ, or the boot is refused (the comparator could not fail).
QWEN36_KV_DEVICE_INIT_MUSTFAIL=1 fills with a non-zero word -> the zero check differs -> the boot is REFUSED.

Off (unset) = today's path, untouched. NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_KV_DEVICE_INIT"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
ZERO_WORD = 0x00000000
MUSTFAIL_WORD = 0x3C3C3C3C  # non-zero exponents and mantissas: finite, non-zero bfp values

_KDIR = Path(__file__).resolve().parent / "kernels" / "kv_devinit"
_SRC = {}
_CB = 0
_MAX_RUNS_PER_CORE = 64  # runtime args: 3 + 3 * runs words (cap ~341)

_banner_done = False


def enabled():
    return _tt_public_getenv(ENV, "0") == "1"


def mustfail():
    return _tt_public_getenv(MUSTFAIL_ENV, "0") == "1"


def fill_word():
    return MUSTFAIL_WORD if mustfail() else ZERO_WORD


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _is_mesh(dev):
    return hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)


def _check(t, what):
    mc = t.memory_config()
    if t.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: {what} must be TILE, got {t.layout}")
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: {what} must be DRAM interleaved, got {mc}")


def _page(t):
    p = int(t.buffer_aligned_page_size())
    if p % 4:
        raise RuntimeError(f"{ENV}: page size {p} is not a 4-byte multiple")
    return p


def _grid(device):
    g = device.compute_with_storage_grid_size()
    return g.x, g.y


def _split(device, n_work):
    """n_work units over the compute grid, row-wise -> (CoreRangeSet, [(core, start, count)])."""
    gx, gy = _grid(device)
    per = max(1, (n_work + gx * gy - 1) // (gx * gy))
    active = (n_work + per - 1) // per
    work = []
    for c in range(active):
        start = c * per
        work.append((ttnn.CoreCoord(c % gx, c // gx), start, min(per, n_work - start)))
    return _crs(gx, active), work


def _crs(gx, active):
    full, rem = divmod(active, gx)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def _cb(dtype, page, crs):
    return ttnn.CBDescriptor(
        total_size=page,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=_CB, data_format=dtype, page_size=page)],
    )


def _kernel(name, crs, ct, rt):
    return ttnn.KernelDescriptor(
        kernel_source=_src(name),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def fill_pages(t, word):
    """Every page of t (DRAM interleaved, TILE) <- the 32-bit `word` repeated, on every chip. In place."""
    _check(t, "fill target")
    page = _page(t)
    n = int(t.buffer_num_pages())
    crs, work = _split(t.device(), n)
    ct = [_CB, page, int(word) & 0xFFFFFFFF] + list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    addr = t.buffer_address()
    for core, start, cnt in work:
        rt[core.x][core.y] = [addr, start, cnt]
    desc = ttnn.ProgramDescriptor(
        kernels=[_kernel("fill_writer.cpp", crs, ct, rt)],
        semaphores=[],
        cbs=[_cb(t.dtype, page, crs)],
    )
    ttnn.generic_op([t, t], desc)  # generic_op wants >= 2 io tensors; t is both (only written)
    return t


def alloc_filled(shape, dtype, device, memory_config=None, word=None):
    """ttnn.empty(shape, dtype, TILE, device, memory_config) + fill_pages(word). On a mesh the buffer is
    allocated on every chip (as ReplicateTensorToMesh's upload does) and the program runs on every chip."""
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    t = ttnn.empty([int(s) for s in shape], dtype, ttnn.TILE_LAYOUT, device, mc)
    if _is_mesh(device):
        # ttnn.empty on a (1, 2) mesh records a 2-D replicate topology; the as_tensor(ReplicateTensorToMesh) upload
        # records a 1-D one. Same bytes and addresses either way; copy the upload's topology (from a HOST tensor, so
        # no device allocation shifts the pool) so every consumer sees exactly today's tensor metadata.
        t.update_tensor_topology(replicate_topology(device))
    return fill_pages(t, fill_word() if word is None else word)


def replicate_topology(device):
    """The tensor topology ttnn.as_tensor(..., mesh_mapper=ReplicateTensorToMesh(device)) gives (host-only build)."""
    import torch

    h = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return h.tensor_topology()


def gather_raw(t, runs):
    """Raw bytes of whole pages of t. runs = [(src_page, len), ...] -> per chip a list of `bytes`, one per page,
    in run order. Device copy page by page into a ROW_MAJOR uint32 [n, page/4] buffer, then a host read."""
    import numpy as np

    _check(t, "gather source")
    page = _page(t)
    n = sum(int(r[1]) for r in runs)
    dev = t.device()
    dst = ttnn.empty([n, page // 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    try:
        if int(dst.buffer_aligned_page_size()) != page:
            raise RuntimeError(f"{ENV}: gather buffer page {dst.buffer_aligned_page_size()} != source page {page}")
        gx, gy = _grid(dev)
        per_core = plan_gather(runs, gx * gy)
        ncores = len(per_core)
        crs = _crs(gx, ncores)
        ct = (
            [_CB, page]
            + list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        )
        rt = ttnn.RuntimeArgs()
        for c in range(ncores):
            args = [t.buffer_address(), dst.buffer_address(), len(per_core[c])]
            for s0, d0, ln in per_core[c]:
                args += [s0, d0, ln]
            rt[c % gx][c // gx] = args
        desc = ttnn.ProgramDescriptor(
            kernels=[_kernel("page_gather.cpp", crs, ct, rt)],
            semaphores=[],
            cbs=[_cb(t.dtype, page, crs)],
        )
        ttnn.generic_op([t, dst], desc)
        per_chip = ttnn.get_device_tensors(dst) if _is_mesh(dev) else [dst]
        out = []
        for dt_ in per_chip:
            a = ttnn.to_torch(dt_).numpy().astype(np.uint32, copy=False).reshape(n, page // 4)
            out.append([a[i].tobytes() for i in range(n)])
        return out
    finally:
        ttnn.deallocate(dst)


def plan_gather(runs, max_cores):
    """runs [(src_page, len)] -> per core a list of (src_page, dst_page, len) chunks: dst pages are the run pages in
    run order (0..n-1), long runs are split so up to max_cores cores share the work, <= _MAX_RUNS_PER_CORE each."""
    n = sum(int(r[1]) for r in runs)
    step = max(1, (n + max_cores - 1) // max_cores)
    chunks, d = [], 0
    for s0, ln in runs:
        s0, ln = int(s0), int(ln)
        for o in range(0, ln, step):
            chunks.append((s0 + o, d + o, min(step, ln - o)))
        d += ln
    if not chunks:
        raise RuntimeError(f"{ENV}: empty gather")
    ncores = min(max_cores, len(chunks))
    per_core = [[] for _ in range(ncores)]
    for i, ch in enumerate(chunks):
        per_core[i % ncores].append(ch)
    if max(len(p) for p in per_core) > _MAX_RUNS_PER_CORE:
        raise RuntimeError(f"{ENV}: gather of {len(chunks)} runs exceeds {_MAX_RUNS_PER_CORE} per core")
    return per_core


def reference_tile(dtype, device, value):
    """Raw device bytes of ONE host-packed tile of `value` (exactly today's path: torch -> from_torch(dtype, TILE)
    -> ReplicateTensorToMesh upload), read back through gather_raw from chip 0."""
    import torch

    rep = ttnn.ReplicateTensorToMesh(device) if _is_mesh(device) else None
    t = ttnn.from_torch(
        torch.full((1, 1, 32, 32), float(value), dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    try:
        return gather_raw(t, [(0, 1)])[0][0]
    finally:
        ttnn.deallocate(t)


def sample_runs(t):
    """Whole blocks (dim 0) to sample: first, second, middle, last -> [(page, len)]."""
    nb = int(t.shape[0])
    n = int(t.buffer_num_pages())
    if n % nb:
        raise RuntimeError(f"{ENV}: {n} pages do not split into {nb} blocks")
    ppb = n // nb
    blocks = sorted({0, min(1, nb - 1), nb // 2, nb - 1})
    return [(b * ppb, ppb) for b in blocks], blocks


def selftest(caches, wrong_value=1.0):
    """caches: [(name, tensor)]. Fail closed. -> summary str. Raises RuntimeError (boot REFUSED) when a sampled page
    of any chip differs from the host-packed zero tile of its dtype, or when the wrong-input branch (the same pages
    against a host-packed tile of `wrong_value`) does NOT differ."""
    if not caches:
        raise RuntimeError("QWEN36_KV_DEVICE_INIT self-test REFUSED: no caches to check")
    dev = caches[0][1].device()
    zero_ref, wrong_ref = {}, {}
    bad, wrong_equal, pages = [], [], 0
    for name, t in caches:
        dt = t.dtype
        if dt not in zero_ref:
            zero_ref[dt] = reference_tile(dt, dev, 0.0)
            wrong_ref[dt] = reference_tile(dt, dev, wrong_value)
            if zero_ref[dt] == wrong_ref[dt]:
                raise RuntimeError(
                    f"QWEN36_KV_DEVICE_INIT self-test REFUSED: zero and wrong reference tiles of {dt} are identical"
                )
        runs, blocks = sample_runs(t)
        per_chip = gather_raw(t, runs)
        for chip, got in enumerate(per_chip):
            for i, b in enumerate(got):
                pages += 1
                if b != zero_ref[dt]:
                    bad.append(f"{name} chip{chip} sample{i}")
                if b == wrong_ref[dt]:
                    wrong_equal.append(f"{name} chip{chip} sample{i}")
    # wrong-input branch: every sampled page must differ from the non-zero tile, else the comparator is blind
    if wrong_equal or pages == 0:
        raise RuntimeError(
            "QWEN36_KV_DEVICE_INIT self-test REFUSED: the wrong-input branch passed "
            f"({len(wrong_equal)} pages equal the non-zero tile, e.g. {wrong_equal[:3]})"
        )
    if bad:
        raise RuntimeError(
            f"QWEN36_KV_DEVICE_INIT self-test REFUSED: {len(bad)}/{pages} sampled pages differ from the "
            f"host-packed zero tile (e.g. {bad[:4]}; mustfail={mustfail()})"
        )
    zero_note = ",".join(f"{dt}:{'allzero' if not any(zero_ref[dt]) else 'nonzero-bytes'}" for dt in zero_ref)
    return (
        f"{len(caches)} caches x {len(blocks)} blocks, {pages} pages on all chips equal the host-packed zero tile "
        f"({zero_note}); wrong-input branch differs"
    )


def log_engaged(k_dtype, v_dtype, shape, n_caches, what):
    global _banner_done
    if mustfail():
        logger.warning(
            f"QWEN36_KV_DEVICE_INIT_MUSTFAIL engaged: fill word {MUSTFAIL_WORD:#010x} "
            "(the self-test must REFUSE this boot)"
        )
    if not _banner_done:
        logger.info(
            "QWEN36_KV_DEVICE_INIT engaged: paged KV via ttnn.empty + device fill (no host tensor), "
            f"K={k_dtype} V={v_dtype} shape {list(shape)}"
        )
        _banner_done = True
    logger.info(f"{ENV}: {what}: {n_caches} caches allocated and filled on device")
