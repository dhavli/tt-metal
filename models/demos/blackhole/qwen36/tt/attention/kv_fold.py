# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_KV_UPDATE_FOLD=1: the paged K and V cache update of one attention layer in ONE program.

Why (desk, docs/analysis/sdpa-decode-upstream-20261006.md): about 5 ms of the 6.09 / 12.55 ms B8 attention class
per verify (c154) is fixed cost outside the SDPA op. Part of it is launches. With QWEN36_ATTN_PREP the
MTP-verify KV update of one layer is, per cache (K, V), one sharded_to_interleaved, then per candidate j < T a
ttnn.slice, a to_memory_config reshard to width Bw and a paged_update_cache: 2 * (1 + 3T) programs per layer
(T=3: 20, T=4: 26; 320 / 416 per verify over 16 layers). Plain decode (T=1) issues 2 paged_update_cache per layer.
This path issues 1 per layer in both cases.

What: one ttnn.generic_op over the k/v height shard that attn_decode_prep already emits at width B = T*Bw (row
u = j*Bw + b on core u, candidate-major). Each core runs paged_update_cache's per-row work (kernels/kv_fold/: reader,
writer and compute follow reader/writer_update_cache_interleaved_start_id.cpp and update_cache.cpp) for K and then V,
with update index fold_pos[u] (the [T*Bw] cur_pos the folded verify SDPA already takes: QWEN36_SDPA_SPEC_FOLD's
host-written persistent input `v_posf`, no new tensor) and page-table row u % Bw. Rows of one user at consecutive
positions share cache tiles, which is why the served path ran the T candidates as T programs: here row u waits
(semaphore, per cache) until row u - Bw has written its tiles back, so each tile sees the same read-modify-writes in
the same order as the T sequential calls. Same bytes, bitwise (ttsim + the boot self-test check exactly that).

Moved cost: none on the host or in the step inputs (fold_pos already exists; the page table is shared; no index
tensor is built). On the device, the T read-modify-writes of one user are serial inside one program instead of
across T programs; the S2I, the T slices and the T reshards per cache are gone.

Applies to: the grouped (MTP verify) path with QWEN36_ATTN_PREP + QWEN36_SDPA_SPEC_FOLD, a FoldGroupIO carrying
fold_pos and one shared page table; and the plain paged decode path with QWEN36_ATTN_PREP (T=1). Anything else
(no prep, no fold_pos, e.g. the MTP head's KV reseed) keeps today's ops. Unset: today's path, untouched.

Self-test (fail closed, once per process at the first set_paged_kv_cache, before any trace): for every (T, Bw) the
server can run (T 1..4 x Bw in 1, 2, 4, 8 and the max batch, plus every speculating bucket's (K_b+1, Bw) from
mtp_batch_host.served_verify_cells, which reaches T 16; always T*Bw <= 32) on small scratch caches of the live caches'
dtypes, block size and head dim with random contents, random inputs and edge positions (tile and page
boundaries, a skipped row), the fold against the served op sequence (S2I, slice, reshard, paged_update_cache per
candidate; for T=1 the two plain calls), bitwise on every chip. Wrong-input branch at every boot: the fold with the
must-fail row offset must DIFFER from the reference, else the boot is refused. QWEN36_KV_UPDATE_FOLD_MUSTFAIL=1
serves the must-fail (input row on tile row idx+1): the comparison differs and the boot is REFUSED.
update() refuses a (T, Bw, dtypes) the self-test did not pass.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_KV_UPDATE_FOLD"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")

SELFTEST_BANNER = "QWEN36_KV_UPDATE_FOLD self-test exact"
# Verify depth cap. Nothing in the program depends on T: row u = j*Bw + b runs on core u and waits on row u - Bw, so
# T only sets the chain length; the row cap (one row per core of the k/v height shard, rows in tile-row 0) is
# MAX_ROWS. Was 4 (the K <= 3 MTP shapes); 16 = mtp_batch_host.MAX_K + 1 (verify-depth-gt4, the drafter's K 7-15).
MAX_T = 16
MAX_ROWS = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))
BASE_T = 4  # the T range every boot self-tests at every width (today's cells); deeper T only where a bucket serves it

CARD_GRID = (11, 10)  # the P150 compute grid (x, y) the served shard configs must fit (ttsim's is 13 x 10)


def kv_shard_grid(B, grid_x, grid_y):
    """(cols, rows) of the width-B KV-update height shard (one row per core, attention/tp.py _kv_shard_cfg and
    attn_decode_prep's k/v output). Today's rule, the largest divisor of B that is <= 8, whenever its rows fit the
    grid (every width served before verify-depth-gt4, unchanged). Otherwise (B = 11, 22: a T = 11 verify) the
    smallest divisor c <= grid_x with B / c <= grid_y. Raises when B has no rectangle on the grid (B = 13, 26 on the
    card's 11 x 10: a T = 13 verify at Bw <= 2), so such a width refuses at the boot self-test instead of asking the
    allocator for a core that does not exist."""
    B, grid_x, grid_y = int(B), int(grid_x), int(grid_y)
    cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
    if B // cols <= grid_y:
        return cols, B // cols
    for c in range(1, grid_x + 1):
        if B % c == 0 and B // c <= grid_y:
            return c, B // c
    raise RuntimeError(f"{ENV}: no {grid_x}x{grid_y} core rectangle holds a width-{B} KV-update shard (one row per "
                       f"core); this verify width (T*Bw) is not servable")


_KDIR = Path(__file__).resolve().parent.parent / "kernels" / "kv_fold"
_SRC = {}

# CB ids (mirror paged_update_cache's where they have a counterpart)
CB_CK, CB_INK, CB_IDX, CB_PT, CB_INV, CB_CV = 0, 1, 2, 3, 4, 5
CB_OK, CB_OV = 16, 17
CB_UC, CB_UC2, CB_UINK, CB_UINV = 24, 25, 26, 27
SEM_K, SEM_V = 0, 1

_tested = {}  # (T, Bw, k dtype, v dtype, block, HD, NH) -> summary
_logged = set()


def enabled():
    return _MODE == "1"


def mustfail():
    return _MF == "1"


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _is_mesh(dev):
    return hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)


def _cores(t):
    """The ordered cores of t's height shard (row u on core u): ranges in order, each row-major, as the C++
    corerange_to_cores(grid, n, row_wise=True) that paged_update_cache uses."""
    spec = t.memory_config().shard_spec
    if spec is None or t.memory_config().memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        raise RuntimeError(f"{ENV}: k/v must be height sharded, got {t.memory_config()}")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise RuntimeError(f"{ENV}: k/v shard orientation must be ROW_MAJOR, got {spec.orientation}")
    cores = []
    for r in spec.grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores, spec


def key(T, Bw, k_cache, v_cache):
    s = [int(x) for x in k_cache.shape]
    return (int(T), int(Bw), str(k_cache.dtype), str(v_cache.dtype), s[2], s[3], s[1])


def _check(k_cache, v_cache, k_sh, v_sh, pos, pt, T, Bw):
    B = T * Bw
    ks, vs = [int(x) for x in k_cache.shape], [int(x) for x in v_cache.shape]
    if ks != vs:
        raise RuntimeError(f"{ENV}: K cache {ks} and V cache {vs} differ in shape")
    if not 1 <= T <= MAX_T or B > MAX_ROWS:
        raise RuntimeError(f"{ENV}: T={T} Bw={Bw} outside T<={MAX_T}, T*Bw<={MAX_ROWS}")
    for n, t in (("k", k_sh), ("v", v_sh)):
        sh = [int(x) for x in t.shape]
        if sh[1] != B or sh[3] != ks[3] or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{ENV}: {n} input {sh} {t.dtype} {t.layout}; expected [1, {B}, *, {ks[3]}] bf16 TILE")
    # like paged_update_cache, entries / rows past the first B / Bw are never read
    if [int(x) for x in pos.shape][-1] < B:
        raise RuntimeError(f"{ENV}: update index has {list(pos.shape)} entries, expected >= {B}")
    if int(pt.shape[0]) < Bw:
        raise RuntimeError(f"{ENV}: page table {list(pt.shape)} has {pt.shape[0]} rows, expected >= Bw={Bw}")


def usable(T, Bw, k_cache, v_cache):
    """True when the flag is on and the boot self-test passed this (T, Bw, dtypes, geometry). The call sites keep
    today's ops otherwise (a width the self-test did not cover is served exactly as before, never folded)."""
    if not enabled():
        return False
    if key(T, Bw, k_cache, v_cache) in _tested:
        return True
    k = ("fallback",) + key(T, Bw, k_cache, v_cache)
    if k not in _logged:
        _logged.add(k)
        logger.warning(f"QWEN36_KV_UPDATE_FOLD fallback: T={T} Bw={Bw} not self-tested this process; today's per-candidate ops serve it")
    return False


def update(k_cache, v_cache, k_sh, v_sh, pos, pt, T, Bw, mf=None):
    """K and V paged-cache update of all T*Bw rows (row u = j*Bw + b) in one program. pos: [T*Bw] int32 update
    indices (-1 skips the row); pt: the [Bw, n] page table the T candidates share. In place on the caches.
    Refuses a (T, Bw, dtypes) the self-test did not pass (mf=None, the served call)."""
    _check(k_cache, v_cache, k_sh, v_sh, pos, pt, T, Bw)
    if mf is None:
        k = key(T, Bw, k_cache, v_cache)
        if k not in _tested:
            raise RuntimeError(
                f"{ENV}: (T={T}, Bw={Bw}, K={k_cache.dtype}, V={v_cache.dtype}) was not self-tested this process "
                f"(tested: {sorted(_tested)}); refusing to serve it"
            )
        mf = mustfail()
        if k not in _logged:
            _logged.add(k)
            logger.info(
                f"QWEN36_KV_UPDATE_FOLD engaged: T={T} Bw={Bw}: K+V paged update in 1 program per layer (was "
                f"{2 * (1 + 3 * T) if T > 1 else 2}), K={k_cache.dtype} V={v_cache.dtype}, fold_pos index"
                f"{' MUSTFAIL row offset +1' if mf else ''}"
            )
    _run(k_cache, v_cache, k_sh, v_sh, pos, pt, T, Bw, bool(mf))


def _run(k_cache, v_cache, k_sh, v_sh, pos, pt, T, Bw, mf):
    dev = k_sh.device()
    cores, spec = _cores(k_sh)
    vcores, _ = _cores(v_sh)
    B = T * Bw
    if len(cores) != B or [(c.x, c.y) for c in cores] != [(c.x, c.y) for c in vcores]:
        raise RuntimeError(f"{ENV}: k/v shards on {len(cores)} / {len(vcores)} cores (must be the same {B})")
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    HD = int(k_cache.shape[3])
    NH = int(k_cache.shape[1])
    block = int(k_cache.shape[2])
    if shard_w != HD or shard_h % 32 or shard_h < NH:
        raise RuntimeError(f"{ENV}: shard {shard_h}x{shard_w} does not hold {NH} heads x {HD}")
    WT = HD // 32
    n_in = shard_h * shard_w // 1024
    if n_in != WT:
        raise RuntimeError(f"{ENV}: shard of {n_in} tiles, expected one tile row ({WT})")
    crs = spec.grid
    ck_page = int(k_cache.buffer_aligned_page_size())
    cv_page = int(v_cache.buffer_aligned_page_size())
    ix_page = int(pos.buffer_aligned_page_size())
    pt_page = int(pt.buffer_aligned_page_size())
    bf16_tile = 2048

    def cb(idx, dtype, page, n):
        return ttnn.CBDescriptor(
            total_size=page * n,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    cbs = [
        cb(CB_CK, k_cache.dtype, ck_page, 2 * WT),
        cb(CB_CV, v_cache.dtype, cv_page, 2 * WT),
        ttnn.cb_descriptor_from_sharded_tensor(CB_INK, k_sh),
        ttnn.cb_descriptor_from_sharded_tensor(CB_INV, v_sh),
        cb(CB_IDX, ttnn.int32, ix_page, 1),
        cb(CB_PT, ttnn.int32, pt_page, 1),
        # untilized cache block, patched in place: c_24 and c_25 alias one buffer, as in paged_update_cache
        ttnn.CBDescriptor(
            total_size=2 * WT * bf16_tile,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_UC, data_format=ttnn.bfloat16, page_size=bf16_tile),
                ttnn.CBFormatDescriptor(buffer_index=CB_UC2, data_format=ttnn.bfloat16, page_size=bf16_tile),
            ],
        ),
        cb(CB_UINK, ttnn.bfloat16, bf16_tile, WT),
        cb(CB_UINV, ttnn.bfloat16, bf16_tile, WT),
        cb(CB_OK, k_cache.dtype, ck_page, 2 * WT),
        cb(CB_OV, v_cache.dtype, cv_page, 2 * WT),
    ]
    sems = [
        ttnn.SemaphoreDescriptor(id=SEM_K, core_ranges=crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_V, core_ranges=crs, initial_value=0),
    ]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    r_ct = [CB_CK, CB_CV, CB_INK, CB_INV, CB_IDX, CB_PT, WT, NH, block, block // 32, ix_page, pt_page, SEM_K, SEM_V,
            n_in] + acc(k_cache) + acc(v_cache) + acc(pos) + acc(pt)
    w_ct = [CB_OK, CB_OV, CB_UC, CB_UC2, CB_UINK, CB_UINV, CB_IDX, CB_PT, WT, HD * 2, NH, block, block // 32,
            SEM_K, SEM_V, 1 if mf else 0] + acc(k_cache) + acc(v_cache)
    c_ct = [CB_CK, CB_CV, CB_INK, CB_INV, CB_UC, CB_UC2, CB_UINK, CB_UINV, CB_OK, CB_OV, WT, NH]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    kc, vc, ia, pa = k_cache.buffer_address(), v_cache.buffer_address(), pos.buffer_address(), pt.buffer_address()
    for u, c in enumerate(cores):
        nxt = cores[u + Bw] if u + Bw < B else c
        phys = dev.worker_core_from_logical_core(nxt)
        r_rt[c.x][c.y] = [kc, vc, ia, pa, u, u % Bw, 1 if u >= Bw else 0]
        w_rt[c.x][c.y] = [kc, vc, u, 1 if u + Bw < B else 0, phys.x, phys.y]

    def kern(name, ct, rt, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=_src(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )

    desc = ttnn.ProgramDescriptor(
        kernels=[
            kern("kv_fold_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            kern("kv_fold_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
            kern("kv_fold_compute.cpp", c_ct, [], ttnn.ComputeConfigDescriptor()),
        ],
        semaphores=sems,
        cbs=cbs,
    )
    ttnn.generic_op([k_sh, v_sh, pos, pt, k_cache, v_cache], desc)


# ---------------------------------------------------------------------------------------------- reference + self-test


def served_reference(k_cache, v_cache, k_sh, v_sh, pos_j, pt, T, Bw, shard_cfg):
    """Today's ops, op for op: T == 1 -> the plain path's two paged_update_cache calls on the width-B shards;
    T > 1 -> _forward_decode_grouped's QWEN36_ATTN_PREP branch: per cache one sharded_to_interleaved (L1), then per
    candidate j a slice of rows j*Bw.., a reshard to shard_cfg(Bw) and paged_update_cache(pos_j, pt)."""
    L1 = ttnn.L1_MEMORY_CONFIG
    if T == 1:
        ttnn.experimental.paged_update_cache(k_cache, k_sh, update_idxs_tensor=pos_j[0], page_table=pt)
        ttnn.experimental.paged_update_cache(v_cache, v_sh, update_idxs_tensor=pos_j[0], page_table=pt)
        return
    kv_rows, HD = int(k_sh.shape[2]), int(k_sh.shape[3])
    cfg = shard_cfg(Bw)
    for sh, cache in ((k_sh, k_cache), (v_sh, v_cache)):
        il = ttnn.sharded_to_interleaved(sh, L1)
        for j in range(T):
            lo, hi = j * Bw, (j + 1) * Bw
            part = ttnn.slice(il, (0, lo, 0, 0), (1, hi, kv_rows, HD), memory_config=L1)
            part_sh = ttnn.to_memory_config(part, cfg)
            ttnn.deallocate(part)
            ttnn.experimental.paged_update_cache(cache, part_sh, update_idxs_tensor=pos_j[j], page_table=pt)
            ttnn.deallocate(part_sh)
        ttnn.deallocate(il)


def selftest_cells(B_max, served=()):
    """Today's cells (T 1..BASE_T x widths 1, 2, 4, 8, B_max) plus every served (T, Bw) (the speculating buckets'
    verify depths, mtp_batch_host.served_verify_cells), in a stable order: the base cells first, unchanged."""
    widths = sorted({w for w in (1, 2, 4, 8, int(B_max)) if 1 <= w <= int(B_max)})
    base = [(T, w) for w in widths for T in range(1, BASE_T + 1) if T * w <= MAX_ROWS]
    extra = sorted({(int(T), int(w)) for T, w in served if (int(T), int(w)) not in base})
    for T, w in extra:
        if not (1 <= T <= MAX_T and 1 <= w and T * w <= MAX_ROWS):
            raise RuntimeError(f"{ENV}: served verify cell T={T} Bw={w} outside T<={MAX_T}, T*Bw<={MAX_ROWS}")
    return base + extra


def plan_cell(T, Bw, block, seed, ctx=None):
    """-> (page table rows [Bw][n], positions [T][Bw], n physical blocks). Per user b a base position p_b; candidate
    j writes p_b + j. Edge bases first: a tile boundary inside the T rows (30), a page boundary (block - 2), position
    0, a tile row start + 1 inside a block, the last position of a block, then random. ctx (e.g. 32768): the page
    table is ctx // block + 1 columns wide and the bases sit at the end of that context (ctx - 2 crosses into the
    last page). User 1's last candidate is SKIPPED (-1) when Bw >= 2, as fold_pos carries for a padded row. Every
    page a written position lands in has its own physical block (no two users share one)."""
    import random

    rnd = random.Random(seed)
    span = 3 * block if ctx is None else int(ctx) + block
    cols = span // block
    top = span - block  # the last base keeps base + T - 1 < span
    edge = [32 - 2, block - 2, 0, block - 32 + 1, 2 * block - 1] if ctx is None else [
        top - 2, top - block - 1, top - T, top - 32 - 1, top - block - 2]
    base = [(edge[b] if b < len(edge) else rnd.randrange(0, top - T)) for b in range(Bw)]
    pos = [[base[b] + j for b in range(Bw)] for j in range(T)]
    if Bw >= 2:
        pos[T - 1][1] = -1
    both = pos + shift_positions(pos, T)  # a second chained step writes the next T positions
    touched = sorted({(b, p // block) for row in both for b, p in enumerate(row) if p >= 0})
    n = max(len(touched) + 2, cols)  # paged_update_cache validates page-table width <= cache blocks
    phys = list(range(n))
    rnd.shuffle(phys)
    own = {tb: phys[i] for i, tb in enumerate(touched)}
    pt = [[own.get((b, c), rnd.randrange(0, n)) for c in range(cols)] for b in range(Bw)]
    return pt, pos, n


def shift_positions(pos, d):
    return [[(p + d if p >= 0 else -1) for p in row] for row in pos]


def selftest(k_like, v_like, shard_cfg, cells, seed=4321, ctx=None, chain=False):
    """Fail closed. k_like / v_like: the live caches (only dtype, block size, heads, head dim, device are read).
    shard_cfg(B): the attention module's _kv_shard_cfg. Returns {cell: "exact"|...}; raises RuntimeError
    (boot REFUSED) on any differing cell, or when the wrong-input branch (must-fail fold) does NOT differ."""
    import torch

    dev = k_like.device()
    mesh = _is_mesh(dev)
    rep = ttnn.ReplicateTensorToMesh(dev) if mesh else None
    NH, block, HD = int(k_like.shape[1]), int(k_like.shape[2]), int(k_like.shape[3])
    DR, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    res = {}
    g = torch.Generator().manual_seed(seed)

    def up(t, dtype, layout=ttnn.TILE_LAYOUT, mc=DR):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, mesh_mapper=rep, memory_config=mc)

    def chips(t):
        return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)] if mesh else [ttnn.to_torch(t)]

    def bits(x):
        return x.view(torch.int16) if x.dtype == torch.bfloat16 else x

    def same(a, b):
        return len(a) == len(b) and all(x.shape == y.shape and torch.equal(bits(x), bits(y)) for x, y in zip(a, b))

    for (T, Bw) in cells:
        B = T * Bw
        pt_rows, pos, nblk = plan_cell(T, Bw, block, seed + 7 * T + Bw, ctx=ctx)
        cache_k = torch.randn(nblk, NH, block, HD, generator=g).to(torch.bfloat16)
        cache_v = torch.randn(nblk, NH, block, HD, generator=g).to(torch.bfloat16)
        pt_t = torch.tensor(pt_rows, dtype=torch.int32)
        kin = torch.randn(1, B, 32, HD, generator=g).to(torch.bfloat16)
        vin = torch.randn(1, B, 32, HD, generator=g).to(torch.bfloat16)
        keep = []
        try:
            pt = up(pt_t, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            pos_j = [up(torch.tensor(p, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for p in pos]
            posf = up(torch.tensor([x for p in pos for x in p], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            keep += [pt, posf] + pos_j
            cfg = shard_cfg(B)

            steps = [(pos_j, posf, kin, vin)]
            if chain:
                pos2 = shift_positions(pos, T)
                p2j = [up(torch.tensor(p, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for p in pos2]
                p2f = up(torch.tensor([x for p in pos2 for x in p], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                keep += [p2f] + p2j
                steps.append((p2j, p2f, torch.randn(1, B, 32, HD, generator=g).to(torch.bfloat16),
                              torch.randn(1, B, 32, HD, generator=g).to(torch.bfloat16)))

            def shards(ki, vi):
                ks = ttnn.to_memory_config(up(ki, ttnn.bfloat16, mc=L1), cfg)
                vs = ttnn.to_memory_config(up(vi, ttnn.bfloat16, mc=L1), cfg)
                keep.extend([ks, vs])
                return ks, vs

            def path(kind):
                kc, vc = up(cache_k, k_like.dtype), up(cache_v, v_like.dtype)
                keep.extend([kc, vc])
                for pj, pf, ki, vi in steps:
                    ks, vs = shards(ki, vi)
                    if kind == "ref":
                        served_reference(kc, vc, ks, vs, pj, pt, T, Bw, shard_cfg)
                    else:
                        _run(kc, vc, ks, vs, pf, pt, T, Bw, kind == "wrong" or mustfail())
                return chips(kc), chips(vc)

            ref, got, wrong = path("ref"), path("fold"), path("wrong")
            # the reference must actually have written something (else equal caches would prove nothing)
            ik = up(cache_k, k_like.dtype)
            keep.append(ik)
            wrote = not same(chips(ik), ref[0])
            ok_k, ok_v = same(ref[0], got[0]), same(ref[1], got[1])
            mf_differs = not (same(ref[0], wrong[0]) and same(ref[1], wrong[1]))
            res[f"T{T}_B{Bw}"] = ("exact" if ok_k and ok_v else f"differs(K={ok_k},V={ok_v})") + (
                "" if wrote else ",ref-wrote-nothing") + ("" if mf_differs else ",wrong-input-passed")
        finally:
            for t in keep:
                ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return bool(res) and all(v == "exact" for v in res.values())


def ensure_selftest(k_cache, v_cache, B_max, shard_cfg, served=()):
    """Once per process per (dtypes, geometry): run the self-test for every cell, fail closed, record the cells.
    served: the (T, Bw) verify cells the MTP buckets run (mtp_batch_host.served_verify_cells), added to the base."""
    if not enabled():
        return
    cells = selftest_cells(B_max, served)
    todo = [c for c in cells if key(c[0], c[1], k_cache, v_cache) not in _tested]
    if not todo:
        return
    res = selftest(k_cache, v_cache, shard_cfg, todo)
    if not selftest_ok(res):
        raise RuntimeError(
            f"QWEN36_KV_UPDATE_FOLD self-test REFUSED (K={k_cache.dtype} V={v_cache.dtype})"
            f"{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}"
        )
    for T, Bw in todo:
        _tested[key(T, Bw, k_cache, v_cache)] = res[f"T{T}_B{Bw}"]
    logger.info(
        f"{SELFTEST_BANNER}: {len(todo)} cells (T, Bw) {todo} K={k_cache.dtype} V={v_cache.dtype} "
        f"block={int(k_cache.shape[2])} HD={int(k_cache.shape[3])}, bitwise vs the served ops on every chip; "
        f"wrong-input branch differs"
    )
