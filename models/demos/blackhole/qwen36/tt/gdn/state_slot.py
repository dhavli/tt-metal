# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_STATE_SLOT=1: per-row GDN state commit by slot index (docs/MTP-BATCH.md section 10).

The MTP-at-batch commit used to pick, per row b, the state after its accepted length m_b out of T
candidate states with a where-chain (T-1 full-state `ttnn.where` passes) followed by a full copy into
the persistent state: about 11 state-sized DRAM passes per GDN layer, 16.04 ms per step at B=8 K=3
(c115 timing). Here the host writes one uint32 page `idx` (entry b = m_b, the commit depth) and ONE
program per state kind reads, per row, only the selected candidate and writes it straight into the
persistent buffer:

  commit_rec   rec_state[b] <- hists[idx[b]][b]           (TILE [B, Nv, Dk, Dv], page-granular)
  commit_conv  conv_states[i][r] <- cands_i[idx[r]][r]    (TILE [1, Bmax, C], row-in-tile compose)

Both are ttnn.generic_op programs with JIT dataflow kernels (kernels/state_slot/*.cpp): no ttnn C++
rebuild, no image change. Buffer addresses are runtime args (fixed under a trace: the candidates are
the verify trace's own outputs, the destinations the fixed-address decode state), and idx is read by
the kernels at run time, so the traces replay with new accepted lengths like the where-chain's masks.
An idx entry >= T leaves that row untouched.

Bit-exact by construction: the kernels move whole pages (rec) or whole 16-element face rows (conv)
with no arithmetic and no format conversion; the source and destination dtypes must match, which the
wrapper checks (the where path's typecast is refused, not emulated).
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn

ENV = "QWEN36_MTP_STATE_SLOT"
SKIP = 0xFFFF  # idx entry: leave this row as it is
IDX_LEN = 32  # one ROW_MAJOR uint32 page, one entry per decode row (Bw <= 32)
# Deepest verify depth the commits take (verify-depth-gt4; was an assert T <= 8). The readers index T candidate
# addresses from runtime args and select candidates with a 32-bit mask (conv_reader.cpp `need`), so nothing in the
# kernels caps T below 32; 16 = mtp_batch_host.MAX_K + 1.
MAX_T = 16

_KDIR = Path(__file__).resolve().parent / "kernels" / "state_slot"
_SRC = {}

# CB indices (low and dense, BLACKHOLE-PERF section 9)
_CB_DATA, _CB_IDX, _CB_SCR = 0, 1, 2


def enabled():
    return _tt_public_getenv(ENV, "0") == "1"


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _dt(t):
    return t.dtype


def _shape(t):
    return [int(t.shape[i]) for i in range(len(t.shape))]


def _check(t, what):
    mc = t.memory_config()
    if t.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: {what} must be TILE, got {t.layout}")
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: {what} must be DRAM interleaved, got {mc}")


def _tile_bytes(dtype):
    if dtype == ttnn.float32:
        return 4096
    if dtype == ttnn.bfloat16:
        return 2048
    raise RuntimeError(f"{ENV}: unsupported state dtype {dtype} (fp32 or bf16 only)")


def _cores(device, n_work):
    """Split n_work units over the compute grid, row-wise. -> (CoreRangeSet, [(core, start, count)])."""
    g = device.compute_with_storage_grid_size()
    ncores = g.x * g.y
    per = (n_work + ncores - 1) // ncores
    active = (n_work + per - 1) // per
    work = []
    for c in range(active):
        start = c * per
        work.append((ttnn.CoreCoord(c % g.x, c // g.x), start, min(per, n_work - start)))
    full, rem = divmod(active, g.x)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges), work


def _cb(idx, dtype, page, n, crs):
    return ttnn.CBDescriptor(
        total_size=page * n,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
    )


def _kernel(name, crs, ct, rt, cfg):
    return ttnn.KernelDescriptor(
        kernel_source=_src(name),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=ct,
        runtime_args=rt,
        config=cfg,
    )


def _idx_page(idx):
    if idx.layout != ttnn.ROW_MAJOR_LAYOUT or idx.dtype != ttnn.uint32 or _shape(idx)[-1] != IDX_LEN:
        raise RuntimeError(f"{ENV}: idx must be ROW_MAJOR uint32 [1, {IDX_LEN}], got {idx.layout} {idx.dtype} {list(idx.shape)}")
    try:
        return int(idx.buffer_aligned_page_size())
    except Exception:
        return IDX_LEN * 4


def commit_rec(dst, cands, idx, bw, nv, dk, dv):
    """dst[b] <- cands[idx[b]][b] for b < bw (idx[b] >= T: keep). dst [Bmax, nv, dk, dv], cands T x
    [bw, nv, dk, dv], same dtype, TILE, DRAM interleaved. In place into dst; returns dst."""
    T = len(cands)
    for i, c in enumerate(cands):
        _check(c, f"rec candidate {i}")
        if _dt(c) != _dt(dst):
            raise RuntimeError(f"{ENV}: rec candidate dtype {_dt(c)} != state dtype {_dt(dst)}")
        if _shape(c) != [bw, nv, dk, dv]:
            raise RuntimeError(f"{ENV}: rec candidate {i} shape {list(c.shape)} != {[bw, nv, dk, dv]}")
    _check(dst, "rec state")
    if _shape(dst)[0] < bw or _shape(dst)[1:] != [nv, dk, dv]:
        raise RuntimeError(f"{ENV}: rec state shape {list(dst.shape)} vs bucket {bw}x{nv}x{dk}x{dv}")
    assert dk % 32 == 0 and dv % 32 == 0 and bw <= IDX_LEN and 1 <= T <= MAX_T
    page = _tile_bytes(_dt(dst))
    kv = (dk // 32) * (dv // 32)
    ipage = _idx_page(idx)
    crs, work = _cores(dst.device(), bw * nv)
    st_ct = ttnn.TensorAccessorArgs(dst).get_compile_time_args()
    ix_ct = ttnn.TensorAccessorArgs(idx).get_compile_time_args()
    r_ct = [T, kv, nv, _CB_DATA, _CB_IDX, page, ipage] + list(st_ct) + list(ix_ct)
    w_ct = [T, kv, nv, _CB_DATA, _CB_IDX, page] + list(st_ct)
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    caddr = [c.buffer_address() for c in cands]
    for core, start, n in work:
        r_rt[core.x][core.y] = [idx.buffer_address(), start, n] + caddr
        w_rt[core.x][core.y] = [dst.buffer_address(), start, n]
    desc = ttnn.ProgramDescriptor(
        kernels=[
            _kernel("rec_reader.cpp", crs, r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            _kernel("rec_writer.cpp", crs, w_ct, w_rt, ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[],
        cbs=[_cb(_CB_DATA, _dt(dst), page, 2 * kv, crs), _cb(_CB_IDX, ttnn.uint32, ipage, 1, crs)],
    )
    return ttnn.generic_op([idx] + list(cands) + [dst], desc)


def commit_conv(lives, snaps, idx, bw):
    """Per tap i: lives[i][0, r] <- cands_i[idx[r]][0, r] for r < bw, with cands_i = [snaps[0][i], ...,
    snaps[T-2][i], lives[i]] (idx[r] == T-1 or >= T: keep the live row). lives: NTAP x [1, Bmax, C],
    snaps: (T-1) x NTAP x [1, Bmax, C], one dtype, TILE, DRAM interleaved. In place; returns lives[-1]."""
    ntap = len(lives)
    T = len(snaps) + 1
    shp = _shape(lives[0])
    for i, t in enumerate(list(lives) + [x for sn in snaps for x in sn]):
        _check(t, f"conv tap {i}")
        if _dt(t) != _dt(lives[0]) or _shape(t) != shp:
            raise RuntimeError(f"{ENV}: conv taps must share dtype/shape ({_dt(t)} {list(t.shape)} vs {shp})")
    assert len(shp) == 3 and shp[0] == 1 and bw <= min(32, shp[1]) and shp[2] % 32 == 0 and 1 <= T <= MAX_T
    if any(len(sn) != ntap for sn in snaps):
        raise RuntimeError(f"{ENV}: every snapshot must hold {ntap} taps")
    page = _tile_bytes(_dt(lives[0]))
    ct = shp[2] // 32
    ipage = _idx_page(idx)
    crs, work = _cores(lives[0].device(), ntap * ct)
    st_ct = ttnn.TensorAccessorArgs(lives[0]).get_compile_time_args()
    ix_ct = ttnn.TensorAccessorArgs(idx).get_compile_time_args()
    r_ct = [T, ntap, ct, bw, _CB_DATA, _CB_SCR, _CB_IDX, page, ipage] + list(st_ct) + list(ix_ct)
    w_ct = [ntap, ct, _CB_DATA, _CB_IDX, page] + list(st_ct)
    live_a = [t.buffer_address() for t in lives]
    snap_a = [snaps[s][i].buffer_address() for s in range(T - 1) for i in range(ntap)]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, n in work:
        r_rt[core.x][core.y] = [idx.buffer_address(), start, n] + live_a + snap_a
        w_rt[core.x][core.y] = [start, n] + live_a
    desc = ttnn.ProgramDescriptor(
        kernels=[
            _kernel("conv_reader.cpp", crs, r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            _kernel("conv_writer.cpp", crs, w_ct, w_rt, ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[],
        cbs=[
            _cb(_CB_DATA, _dt(lives[0]), page, 2, crs),
            _cb(_CB_IDX, ttnn.uint32, ipage, 1, crs),
            _cb(_CB_SCR, _dt(lives[0]), page, 1, crs),
        ],
    )
    return ttnn.generic_op([idx] + [x for sn in snaps for x in sn] + list(lives), desc)


def commit_conv_multi(lives, wins, idx, bw, T):
    """QWEN36_MTP_CONV_MULTI commit (gdn/conv_multi.py, docs/MTP-BATCH.md section 14). wins: the NTAP=4 verify window
    tensors [1, T*bw, C] after the one batch-M gdn_decode_conv_gates: row j*bw + b = row b's tap after j+1 tokens.
    Writes, per tap k and for every row of lives[k] ([1, Bmax, C]), what the sequential verify + commit_conv leaves:
      r < bw : wins[k] row min(idx[r], T-1)*bw + r  (idx >= T keeps the live row = the tap after T tokens)
      r >= bw: the row shifted T times with zero input (lives[k+T] row r when k+T < 4, else zero)
    Same dtype everywhere, TILE, DRAM interleaved. In place; returns lives[-1]."""
    ntap = len(lives)
    if ntap != 4 or len(wins) != 4:
        raise RuntimeError(f"{ENV}: conv multi commit needs 4 live taps and 4 windows ({ntap}, {len(wins)})")
    shp = _shape(lives[0])
    M = T * bw
    for i, t in enumerate(lives):
        _check(t, f"conv tap {i}")
        if _dt(t) != _dt(lives[0]) or _shape(t) != shp:
            raise RuntimeError(f"{ENV}: conv taps must share dtype/shape ({_dt(t)} {list(t.shape)} vs {shp})")
    for i, t in enumerate(wins):
        _check(t, f"conv window {i}")
        if _dt(t) != _dt(lives[0]) or _shape(t) != [1, M, shp[2]]:
            raise RuntimeError(f"{ENV}: conv window {i} {_dt(t)} {list(t.shape)} != {_dt(lives[0])} {[1, M, shp[2]]}")
    assert len(shp) == 3 and shp[0] == 1 and bw <= shp[1] and shp[2] % 32 == 0 and 1 <= T and M <= 32
    page = _tile_bytes(_dt(lives[0]))
    ct = shp[2] // 32
    bt = (shp[1] + 31) // 32
    ipage = _idx_page(idx)
    crs, work = _cores(lives[0].device(), bt * ct)
    lv_ct = ttnn.TensorAccessorArgs(lives[0]).get_compile_time_args()
    p_ct = ttnn.TensorAccessorArgs(wins[0]).get_compile_time_args()
    ix_ct = ttnn.TensorAccessorArgs(idx).get_compile_time_args()
    r_ct = [T, bw, ct, _CB_DATA, _CB_SCR, _CB_IDX, page, ipage] + list(lv_ct) + list(p_ct) + list(ix_ct)
    w_ct = [ct, _CB_DATA, _CB_IDX, page] + list(lv_ct)
    live_a = [t.buffer_address() for t in lives]
    win_a = [t.buffer_address() for t in wins]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, start, n in work:
        r_rt[core.x][core.y] = [idx.buffer_address(), start, n] + live_a + win_a
        w_rt[core.x][core.y] = [start, n] + live_a
    desc = ttnn.ProgramDescriptor(
        kernels=[
            _kernel("conv_multi_reader.cpp", crs, r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            _kernel("conv_multi_writer.cpp", crs, w_ct, w_rt, ttnn.WriterConfigDescriptor()),
        ],
        semaphores=[],
        cbs=[
            _cb(_CB_DATA, _dt(lives[0]), page, 2 * ntap, crs),
            _cb(_CB_IDX, ttnn.uint32, ipage, 1, crs),
            _cb(_CB_SCR, _dt(lives[0]), page, 1, crs),
        ],
    )
    return ttnn.generic_op([idx] + list(wins) + list(lives), desc)
