# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_SPEC_SAMPLE_FUSED=1: the exact-sampling candidate block, cut, pick and greedy id in 3 programs, was ~63.

Why (an internal record, [source validation record]): the sampled MTP step runs ~298
extra small device programs against greedy (verify 109, each chain depth 63), and the card step is +21..27% slower
(B1/4k 37.1 -> 45.3 ms, B4/32k 43.8 -> 52.9, B8/32k 52.6 -> 66.1).  By the ttsim work proxy the wide top-K
(topk_large_indices over the 124160/chip shard, on an untilized copy) is ~42% of the extra work, the stable TILE cut
~30% (30 programs per call), the merge 13%; the greedy argmax and the selects add their own chain.

What (spec_sample.candidates + the greedy argmax + dev_sample.select_ids, per sampler call):
  1. kernels/fused_sampler/fs_topk.cpp, ONE multi-core program on the whole grid: every core reads only the face
     rows of its tile range straight from the TILE logits (no untilize), keeps a per-row top-K min-heap in the RISC's
     local RAM (both data-movement RISCs, rows split), writes its sorted partial lists, and core m merges row m's
     partial lists (NC-way) into the chip's K best (value descending, local id ascending: a TOTAL order).
  2. ONE all-gather of the per-chip [M, 2K] block (values and ids together; the served path gathers twice).
  3. kernels/fused_sampler/fs_cut.cpp, one row per RISC: the two-list merge (value descending, global id
     ascending), temperature -> top_k with ties -> softmax -> strict top_p -> renormalised (vLLM's order, the
     served spec_sample.cut), the Gumbel-max pick over the host's pre-scaled noise in the merged column order, the
     greedy id (entry 0 of the merged order IS the served lowest-id argmax), the per-row select (flag) and, in the
     verify, the draft override (v_sx / v_sxf).  Outputs: probs [M, 2K] fp32, ids [M, 2K] fp32 (global), sel [M, 1]
     uint32: exactly what spec_sample.rule and the MTP plumbing take.
The rule (spec_sample.rule) is unchanged.

Numerics: the temperature and the shift (fl(v / T), fl(vt - vt0)) are IEEE float32 like the served ops; exp is
64-bit fixed point (rel err < 2e-9), the sums and the top_p test are exact integers / double, the probability one
correctly rounded division.  The served stable cut is within ~1e-7 of fp64, so the two agree to ~1e-7 (gate <= 1e-6).
Divergences, all where the served path is itself ill-defined: (a) a tie at a shard's K-th value: the served wide
top-K's tie order is unspecified, this one keeps the lowest ids; (b) a tie inside the cut's sorted order (equal
values): the served ttnn.topk order is unspecified, this one is id ascending (a different top_p boundary only when
the boundary candidates tie exactly); (c) the served cut masks the non-top_k candidates to -1e4 before its softmax,
so a kept logit/T below ~-1e4 lets masked candidates in; this one masks exactly; (d) the top_p test: the served
cut compares an fp32 cumsum (before = cum - pr < p), this one the exact mass (integers / double), so when the mass
above a candidate lies within ~1e-7 of p the kept sets can differ by that one candidate (and the pick with it).  The emitted distribution is the same
(the Gumbel noise is i.i.d. per merged column).

Boot self-test (fail closed, once per process before any capture): at every served (M, per) cell the fused block, cut,
pick and emitted ids are compared with the served chain (spec_sample.candidates + argmax_fold greedy + selects, op for
op) on device logits with planted distinct candidates and a mixed batch (exact rows at T=1 / 0.6, greedy rows, top_k
20 / 8 / none, top_p 0.95 / 0.5 / 1, the verify's draft override): ids and emitted ids bitwise on every chip,
probabilities <= 1e-6; plus a planted-tie greedy check (the T=0 identity: cross-shard and in-shard exact ties at the
max); the wrong-input branch (emitted ids + 1) must differ.  QWEN36_MTP_SPEC_SAMPLE_FUSED_MUSTFAIL=1 serves the +1:
the comparison differs and the boot is REFUSED.

QWEN36_MTP_SPEC_SAMPLE_SSP=1 (sampled-step parity, an internal record build 1; needs FUSED, SERVE and
QWEN36_MTP_DEVACCEPT): the three pieces of the sampled step that FUSED left as separate programs or host work go into
fs_cut.cpp (compile switches; the S7 program is the switches at 0):
  * the exact rule (spec_sample.rule, ~46 programs per verify) on each verify row, in IEEE fp32 (= spec_sample.host_rule);
    fs_cut then writes accept [M, 1] fp32 and resid [M, 1] uint32 ROW_MAJOR, which the device accept reads as is;
  * the noise (v_gum, v_su, v_snr per verify row; d_sgum{k} per chain row) is MADE in fs_cut from the host's own hash
    (kernels/fused_sampler/fs_noise.h, pure integer C++, the host builders' values bit for bit on the gated keys) for
    only the columns the row reads, keyed from one 1 KiB noise plan page per step (noise_plan_page) and, in the chain,
    the device accept's m page: the host no longer builds or uploads the noise blocks (~80 KB per step at B8 under
    DEVACCEPT, whose chain tables hold the noise for every possible m);
  * the draft-vocab id map (mtp_draft_vocab.map_ids, ~11 programs per depth) is one table lookup in the chain's fs_cut.
Boot self-test (fail closed): at every served cell the SSP program against the S7 composition it replaces (fs_cut with
the host-built noise blocks, spec_sample.rule, map_ids), every output bitwise on every chip; the wrong-input branch (the
noise keyed one position late) must differ.  QWEN36_MTP_SPEC_SAMPLE_SSP_MUSTFAIL=noise|accept|resid_p serves a wrong
branch: the self-test refuses the boot.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_MTP_SPEC_SAMPLE_FUSED"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
SELFTEST_BANNER = "QWEN36_MTP_SPEC_SAMPLE_FUSED self-test exact"
ENGAGED_BANNER = "QWEN36_MTP_SPEC_SAMPLE_FUSED engaged"
MAX_ROWS = 32  # fs_topk: one tile row
NDEV = 2
PROB_TOL = 1e-6
# fs_topk MF / fs_cut MF codes
MF_NONE, MF_IDS, MF_DIST = 0, 1, 2
MF_MODES = {"0": None, "1": "ids", "ids": "ids", "topk": "topk", "dist": "dist"}

_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF_RAW = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF_RAW not in MF_MODES:
    raise RuntimeError(f"{MUSTFAIL_ENV} must be one of {sorted(MF_MODES)}; got {_MF_RAW!r}")
_MF = MF_MODES[_MF_RAW]

SSP_ENV = "QWEN36_MTP_SPEC_SAMPLE_SSP"
SSP_MUSTFAIL_ENV = SSP_ENV + "_MUSTFAIL"
SSP_SELFTEST_BANNER = "QWEN36_MTP_SPEC_SAMPLE_SSP self-test exact"
SSP_ENGAGED_BANNER = "QWEN36_MTP_SPEC_SAMPLE_SSP engaged"
# fs_cut MF codes of the SSP branches (fs_cut.cpp)
MF_ACCEPT, MF_RESID_P, MF_NOISE = 3, 4, 5
SSP_MF_MODES = {"0": None, "accept": MF_ACCEPT, "resid_p": MF_RESID_P, "noise": MF_NOISE}
_SSP = _tt_public_getenv(SSP_ENV, "0")
if _SSP not in ("0", "1"):
    raise RuntimeError(f"{SSP_ENV} must be 0 or 1; got {_SSP!r}")
_SSP_MF_RAW = _tt_public_getenv(SSP_MUSTFAIL_ENV, "0")
if _SSP_MF_RAW not in SSP_MF_MODES:
    raise RuntimeError(f"{SSP_MUSTFAIL_ENV} must be one of {sorted(SSP_MF_MODES)}; got {_SSP_MF_RAW!r}")
_SSP_MF = SSP_MF_MODES[_SSP_MF_RAW]
if _SSP_MF and _SSP != "1":
    raise RuntimeError(f"{SSP_MUSTFAIL_ENV}={_SSP_MF_RAW} needs {SSP_ENV}=1")

# the noise plan page (fs_cut.cpp NP_*): int32 words, one 1 KiB ROW_MAJOR page per step, written before the verify
NP_LEN = 256
NP_SEED_DS, NP_VPOS, NP_TDS_LO, NP_TDS_HI, NP_SEED_SP = 0, 32, 64, 96, 128  # per verify row r < 32
NP_CSEED, NP_CBASE, NP_CT_LO, NP_CT_HI = 160, 176, 192, 208  # per chain row b < 16
NP_MAX_VROWS, NP_MAX_CROWS = 32, 16
STREAM_CHAIN = 20  # spec_sample.STREAM_CHAIN (fs_cut keys stream 20 + depth)

_KDIR = Path(__file__).resolve().parent / "kernels" / "fused_sampler"
_SRC = {}
_tested = {}  # (M, per, ndev, ovr) -> "exact"
_ssp_tested = {}  # (M, per, ndev, ovr) -> "exact" (QWEN36_MTP_SPEC_SAMPLE_SSP)
_logged = set()

CB_DATA, CB_LIST, CB_RED = 0, 1, 2
CB_SCR = 0
SEM = 0


def enabled():
    """The flag (the served wiring also needs QWEN36_MTP_SPEC_SAMPLE_SERVE; mtp_batch refuses otherwise)."""
    return _MODE == "1"


def mustfail():
    return _MF


def ssp():
    """QWEN36_MTP_SPEC_SAMPLE_SSP (needs FUSED; mtp_batch refuses it without SERVE and DEVACCEPT)."""
    return _SSP == "1"


def ssp_mustfail():
    return _SSP_MF


def _src(name):
    """A kernel's source with its local headers inlined (SOURCE_CODE kernels are compiled from the string, so a
    relative #include has no directory to resolve against)."""
    if name not in _SRC:
        out = []
        for line in (_KDIR / name).read_text().splitlines():
            st = line.strip()
            if st.startswith('#include "') and st.endswith('"') and (_KDIR / st[10:-1]).is_file():
                out.append(_src(st[10:-1]))
            else:
                out.append(line)
        _SRC[name] = "\n".join(out) + "\n"
    return _SRC[name]


def _f64_words(t):
    """(lo, hi) uint32 words of float(t) as an IEEE float64 (the host multiplies g by the Python float T)."""
    import struct

    b = struct.unpack("<Q", struct.pack("<d", float(t)))[0]
    return b & 0xFFFFFFFF, b >> 32


def _i32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def noise_plan_page(v_seeds_ds=(), v_pos=(), v_temps=(), v_seeds_sp=(), c_seeds=(), c_base=(), c_temps=()):
    """The per-step noise plan page (NP_LEN int32 words) fs_cut keys its made noise with.
    Verify rows r (j-major, r = j*bw + b): the target pick's dev_sample key (seed, position = verify_pos + 1, T) and the
    spec seed of the accept uniform / residual noise (same position).  Chain rows b: the spec seed, the base position
    p_b + 2 (fs_cut adds m_b from the device accept's m page and the depth) and T (0 for a T == 0 row: no noise)."""
    import torch

    if len(v_pos) > NP_MAX_VROWS or len(c_base) > NP_MAX_CROWS:
        raise RuntimeError(f"{SSP_ENV}: noise plan for {len(v_pos)} verify / {len(c_base)} chain rows "
                           f"(max {NP_MAX_VROWS} / {NP_MAX_CROWS})")
    w = [0] * NP_LEN
    for r, (sd, p, t, ss) in enumerate(zip(v_seeds_ds, v_pos, v_temps, v_seeds_sp)):
        lo, hi = _f64_words(t if float(t) > 0.0 else 0.0)
        w[NP_SEED_DS + r], w[NP_VPOS + r], w[NP_TDS_LO + r], w[NP_TDS_HI + r] = int(sd), int(p), lo, hi
        w[NP_SEED_SP + r] = int(ss)
    for b, (sd, base, t) in enumerate(zip(c_seeds, c_base, c_temps)):
        lo, hi = _f64_words(t if float(t) > 0.0 else 0.0)
        w[NP_CSEED + b], w[NP_CBASE + b], w[NP_CT_LO + b], w[NP_CT_HI + b] = int(sd), int(base), lo, hi
    for i, v in enumerate(w):
        if not (-(1 << 31) <= _i32(v) < (1 << 31)):
            raise RuntimeError(f"{SSP_ENV}: noise plan word {i} = {v} is not 32-bit")
    return torch.tensor([_i32(v) for v in w], dtype=torch.int32).reshape(1, NP_LEN)


def id_table(ids, mesh, shift=0):
    """The draft vocab's id table for fs_cut's MAP: [ceil(N/16), 16] uint32 ROW_MAJOR DRAM (one 64-byte page per 16 ids),
    entry s = ids[(s + shift) % N] (shift as mtp_draft_vocab.digit_table: 1 is the draft vocab's must-fail map)."""
    import numpy as np
    import torch

    ids = np.asarray(ids, dtype=np.int64)
    n = int(ids.size)
    src = ids[(np.arange(n) + int(shift)) % n]
    rows = (n + 15) // 16
    t = np.zeros(rows * 16, dtype=np.int64)
    t[:n] = src
    tt = torch.from_numpy(t.astype(np.int32)).reshape(rows, 16)
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if _ndev(mesh) > 1 else {}
    return ttnn.from_torch(tt, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep)


def _shape(t):
    return [int(x) for x in t.shape]


def _ndev(dev):
    return int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1


def key(M, per, ndev, ovr=False):
    return (int(M), int(per), int(ndev), bool(ovr))


def _grid(dev):
    g = dev.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def _crs(cores):
    """CoreRangeSet over logical cores [(x, y)], one range per run along x."""
    ranges = []
    for x, y in cores:
        c = ttnn.CoreCoord(x, y)
        ranges.append(ttnn.CoreRange(c, c))
    return ttnn.CoreRangeSet(ranges)


def _rows_crs(gx, n):
    """The first n cores of the grid in row-major logical order, as full-row rectangles plus one partial row."""
    full, rem = divmod(int(n), int(gx))
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def _noc_xy(dev, gx, gy):
    """NoC coordinates of the grid's logical columns and rows; refuses a grid whose map is not separable."""
    px = [int(dev.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(gx)]
    py = [int(dev.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(gy)]
    for y in range(gy):
        for x in range(gx):
            c = dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            if int(c.x) != px[x] or int(c.y) != py[y]:
                raise RuntimeError(f"{ENV}: the logical->NoC map is not separable at ({x}, {y}): {c}")
    return px, py


def _spans(r0, r1):
    """fs_topk.cpp's per-tile L1 bytes for rows [r0, r1) (64-byte aligned face-row spans, both faces of a half)."""
    ts = 0
    for f in (0, 1):
        a, b = max(r0, 16 * f), min(r1, 16 * f + 16)
        if a < b:
            a_al = (a - 16 * f) & ~1
            b_al = (b - 16 * f + 1) & ~1
            ts += 2 * (b_al - a_al) * 32
    return ts


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _check_dram_interleaved(t, what):
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: {what} must be DRAM interleaved; got {mc}")


def _check_rm_rows(t, name, M, w, dtype=ttnn.float32):
    s = _shape(t)
    if len(s) != 4 or s[:3] != [1, 1, M] or s[3] != w:
        raise RuntimeError(f"{ENV}: {name} {s}; expected [1, 1, {M}, {w}]")
    if t.layout != ttnn.ROW_MAJOR_LAYOUT or t.dtype != dtype:
        raise RuntimeError(f"{ENV}: {name} must be ROW_MAJOR {dtype}; got {t.layout} {t.dtype}")
    _check_dram_interleaved(t, name)
    page = int(t.buffer_aligned_page_size())
    if page < w * 4:
        raise RuntimeError(f"{ENV}: {name} page {page} B < one row ({w * 4} B): not one page per row")
    return page


# ---------------------------------------------------------------------------------------------- device programs
def topk_block(logits, per, mesh, K, mf_topk=False):
    """The per-chip top-K block [1, 1, M, 2K] float32 ROW_MAJOR (bits: K values, then K local ids) of the
    vocab-shard logits [1, 1, M, per] bf16 TILE DRAM interleaved.  ONE program (fs_topk.cpp).  logits NOT freed."""
    M, per, K = int(logits.shape[2]), int(per), int(K)
    s = _shape(logits)
    if len(s) != 4 or s[:3] != [1, 1, M] or s[3] != per or logits.dtype != ttnn.bfloat16:
        raise RuntimeError(f"{ENV}: logits {s} {logits.dtype}; expected [1, 1, {M}, {per}] bf16")
    if logits.layout != ttnn.TILE_LAYOUT:
        raise RuntimeError(f"{ENV}: logits must be TILE (the kernel reads the faces); got {logits.layout}")
    _check_dram_interleaved(logits, "logits")
    if not 1 <= M <= MAX_ROWS:
        raise RuntimeError(f"{ENV}: M={M} outside 1..{MAX_ROWS}")
    if per % 32 or per < 32:
        raise RuntimeError(f"{ENV}: per={per} must be a positive tile multiple")
    if not 1 <= K <= 64:
        # fs_cut sums up to 2K values of at most 2^56 in a uint64: 2K <= 128 keeps it < 2^63
        raise RuntimeError(f"{ENV}: K={K} outside 1..64")
    tile_page = int(logits.buffer_aligned_page_size())
    if tile_page != 2048:
        raise RuntimeError(f"{ENV}: logits page {tile_page} B; expected one bf16 tile (2048 B)")
    gx, gy_full = _grid(mesh)
    NT = per // 32
    NC = min(gx * gy_full, NT, 255)
    if NC < M:
        raise RuntimeError(f"{ENV}: {NC} cores < M={M} rows (core m reduces row m)")
    if (NT + NC - 1) // NC * 32 > 0xFFFF:
        raise RuntimeError(f"{ENV}: {NT} tiles over {NC} cores exceeds the 16-bit column index")
    gy = (NC + gx - 1) // gx
    px, py = _noc_xy(mesh, gx, gy)
    base, rem = divmod(NT, NC)
    maxt = base + (1 if rem else 0)
    h = (M + 1) // 2
    roles = [(0, h, ttnn.WriterConfigDescriptor())]
    if h < M:
        roles.append((h, M, ttnn.ReaderConfigDescriptor()))
    ts = [_spans(a, b) for a, b, _ in roles]
    data_bytes = maxt * sum(ts)
    list_bytes = M * K * 8
    red_bytes = NC * K * 8 + 2 * K * 4 + 2 * NC  # the NC lists, the output row, the merge cursors + heap
    crs = _rows_crs(gx, NC)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, 2 * K]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh,
                                         ttnn.DRAM_MEMORY_CONFIG)
    out_page = int(out.buffer_aligned_page_size())
    if out_page < 2 * K * 4:
        raise RuntimeError(f"{ENV}: out page {out_page} < {2 * K * 4}")

    def cb(idx, nbytes):
        nbytes = max(64, (int(nbytes) + 63) // 64 * 64)
        return ttnn.CBDescriptor(total_size=nbytes, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=nbytes)])

    rts = []
    la, oa = logits.buffer_address(), out.buffer_address()
    for _ in roles:
        rt = ttnn.RuntimeArgs()
        for q in range(NC):
            x, y = q % gx, q // gx
            t0 = q * base + min(q, rem)
            nt = base + (1 if q < rem else 0)
            rt[x][y] = [la, oa, q, t0, nt] + px + py
        rts.append(rt)
    kernels = []
    off = 0
    for (a, b, cfg), t_s, rt in zip(roles, ts, rts):
        ct = [CB_DATA, CB_LIST, CB_RED, M, K, NC, a, b, SEM, off, gx, gy, tile_page, out_page,
              1 if mf_topk else 0] + _acc(logits) + _acc(out)
        kernels.append(ttnn.KernelDescriptor(kernel_source=_src("fs_topk.cpp"),
                                             source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                             core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg))
        off += maxt * t_s
    desc = ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[ttnn.SemaphoreDescriptor(id=SEM, core_ranges=crs, initial_value=0)],
        cbs=[cb(CB_DATA, data_bytes), cb(CB_LIST, list_bytes), cb(CB_RED, red_bytes)])
    ttnn.generic_op([logits, out], desc)
    return out


def cut(g, blk, M, K, per, mesh, override=None, mf=MF_NONE, debug=False):
    """fs_cut.cpp (the S7 program): (probs, ids, sel, aux) as before the SSP switches; see cut_x."""
    o = cut_x(g, blk, M, K, per, mesh, override=override, mf=mf, debug=debug)
    return o["probs"], o["ids"], o["sel"], o.get("aux")


def cut_x(g, blk, M, K, per, mesh, override=None, mf=MF_NONE, debug=False, rule=None, noise=None, table=None):
    """fs_cut.cpp over the gathered block g [1, 1, M, 4K] (bits).  blk: role-keyed noise/scale/rank [1,1,M,2K],
    topp/flag [1,1,M,1] fp32 ROW_MAJOR.  override: (x, xf) [1,1,M,1] fp32 (the verify's draft and its flag) or None.
    QWEN36_MTP_SPEC_SAMPLE_SSP switches (fs_cut.cpp):
      rule   {"qp", "qi"} [1,1,M,2K] fp32 (the carried q block) and, without made noise, {"u" [1,1,M,1], "nres"
             [1,1,M,2K]}: the exact rule on every row (needs override); adds "accept" / "resid" to the outputs
      noise  {"np": the noise plan page [1, 256] int32, "mode": 1 (verify) | 2 (chain), "depth": k, "m": the device
             accept's m page (mode 2)}: the noise is made in the kernel; blk's "noise" is not read
      table  the draft vocab's id table (id_table): the emitted id is mapped sub-id -> full id
    Returns a dict: probs [1,1,M,2K] fp32, ids [1,1,M,2K] fp32, sel [1,1,M,1] uint32, aux [1,1,M,4] uint32 (debug),
    accept [1,1,M,1] fp32 / resid [1,1,M,1] uint32 (rule), ndump [1,1,M,W] fp32 (debug with made noise: the made pick
    noise per merged column, the made nres per sorted column, then u)."""
    M, K, per = int(M), int(K), int(per)
    C2 = 2 * K
    nz_mode = int(noise["mode"]) if noise is not None else 0
    if nz_mode not in (0, 1, 2):
        raise RuntimeError(f"{SSP_ENV}: noise mode {nz_mode}")
    if rule is not None and override is None:
        raise RuntimeError(f"{SSP_ENV}: the rule runs in the verify (it needs the draft override)")
    if nz_mode == 1 and override is None or nz_mode == 2 and (override is not None or rule is not None):
        raise RuntimeError(f"{SSP_ENV}: noise mode {nz_mode} with override={override is not None} rule={rule is not None}")
    if table is not None and override is not None:
        raise RuntimeError(f"{SSP_ENV}: the id map is the chain's")
    if nz_mode and M > (NP_MAX_VROWS if nz_mode == 1 else NP_MAX_CROWS):
        raise RuntimeError(f"{SSP_ENV}: M={M} rows exceed the noise plan page")
    g_page = _check_rm_rows(g, "gathered block", M, 2 * C2)
    pages = {}
    for name, w in (("scale", C2), ("rank", C2), ("topp", 1), ("flag", 1)):
        t = blk.get(name)
        if t is None:
            raise RuntimeError(f"{ENV}: the block has no '{name}' tensor (keys {sorted(blk)})")
        pages[name] = _check_rm_rows(t, f"block '{name}'", M, w)
    dummy = blk["topp"]
    if nz_mode == 0:
        if blk.get("noise") is None:
            raise RuntimeError(f"{ENV}: the block has no 'noise' tensor (keys {sorted(blk)})")
        pages["noise"] = _check_rm_rows(blk["noise"], "block 'noise'", M, C2)
        nz_t = blk["noise"]
    else:
        nz_t = dummy
        pages["noise"] = pages["topp"]
    if override is not None:
        x, xf = override
        pages["x"] = _check_rm_rows(x, "draft x", M, 1)
        pages["xf"] = _check_rm_rows(xf, "draft flag", M, 1)
    else:
        x = xf = dummy
        pages["x"] = pages["xf"] = pages["topp"]

    def _rm1(t, name, shape, dtype):
        s_ = _shape(t)
        if s_[-len(shape):] != list(shape) or any(v != 1 for v in s_[:-len(shape)]) or t.layout != ttnn.ROW_MAJOR_LAYOUT \
                or t.dtype != dtype:
            raise RuntimeError(f"{SSP_ENV}: {name} {s_} {t.layout} {t.dtype}; expected ROW_MAJOR {dtype} {list(shape)}")
        _check_dram_interleaved(t, name)
        return int(t.buffer_aligned_page_size())

    ex = {}  # the SSP tensors: name -> (tensor, page)
    if nz_mode:
        npt = noise["np"]
        ex["np"] = (npt, _rm1(npt, "noise plan", (1, NP_LEN), ttnn.int32))
        if ex["np"][1] < 4 * NP_LEN:
            raise RuntimeError(f"{SSP_ENV}: noise plan page {ex['np'][1]} B < {4 * NP_LEN}")
        if nz_mode == 2:
            mt = noise["m"]
            ex["mp"] = (mt, _rm1(mt, "m page", (1, 64), ttnn.uint32))
    if rule is not None:
        ex["qp"] = (rule["qp"], _check_rm_rows(rule["qp"], "q probs", M, C2))
        ex["qi"] = (rule["qi"], _check_rm_rows(rule["qi"], "q ids", M, C2))
        if nz_mode == 0:
            ex["u"] = (rule["u"], _check_rm_rows(rule["u"], "u", M, 1))
            ex["nr"] = (rule["nres"], _check_rm_rows(rule["nres"], "nres", M, C2))
    if table is not None:
        ex["tb"] = (table, _rm1(table, "id table", (int(table.shape[-2]), 16), ttnn.uint32))
    alloc = lambda w, dt: ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, w]), dt, ttnn.ROW_MAJOR_LAYOUT,  # noqa: E731
                                                         mesh, ttnn.DRAM_MEMORY_CONFIG)
    probs, ids, sel = alloc(C2, ttnn.float32), alloc(C2, ttnn.float32), alloc(1, ttnn.uint32)
    aux = alloc(4, ttnn.uint32) if debug else None
    acc = res = nd = None
    if rule is not None:
        acc, res = alloc(1, ttnn.float32), alloc(1, ttnn.uint32)
        ex["ac"] = (acc, int(acc.buffer_aligned_page_size()))
        ex["rs"] = (res, int(res.buffer_aligned_page_size()))
    if debug and nz_mode:
        nd = alloc(2 * C2 + 16, ttnn.float32)
        ex["nd"] = (nd, int(nd.buffer_aligned_page_size()))
    aux_t = aux if aux is not None else sel
    p_pr, p_id, p_sel = (int(t.buffer_aligned_page_size()) for t in (probs, ids, sel))
    p_aux = int(aux_t.buffer_aligned_page_size())

    def al(v):
        return (int(v) + 63) // 64 * 64

    def pg(n):
        return ex[n][1] if n in ex else pages["topp"]

    scr = sum(al(v) for v in (g_page, pages["scale"], pages["rank"], pages["topp"], pages["noise"], pages["flag"],
                              pages["x"], pages["xf"], p_pr, p_id, p_sel, p_aux)) + 3 * al(C2 * 4) + al(C2 * 8)
    # the SSP pieces, in fs_cut.cpp's order (O_NP .. O_END)
    scr += (al(pg("np")) if nz_mode else 0) + (al(pg("qp")) + al(pg("qi")) if rule is not None else 0)
    scr += (al(pg("u")) + al(pg("nr")) if rule is not None and nz_mode == 0 else 0)
    scr += (al(pg("ac")) + al(pg("rs")) if rule is not None else 0) + (al(pg("mp")) if nz_mode == 2 else 0)
    scr += (al(pg("tb")) if table is not None else 0) + (al(pg("nd")) if debug and nz_mode else 0)
    scr += al(C2 * 4) if nz_mode else 0
    gx, gy = _grid(mesh)
    ncores = min(gx * gy, (M + 1) // 2)
    nchunks = 2 * ncores
    per_chunk, extra = divmod(M, nchunks)
    bounds, r = [], 0
    for i in range(nchunks):
        n = per_chunk + (1 if i < extra else 0)
        bounds.append((r, r + n))
        r += n
    crs = _rows_crs(gx, ncores)
    tens = [g, blk["scale"], blk["rank"], blk["topp"], nz_t, blk["flag"], x, xf, probs, ids, sel, aux_t]
    tens += [ex[n][0] if n in ex else dummy for n in ("np", "qp", "qi", "u", "nr", "ac", "rs", "mp", "tb", "nd")]
    addrs = [t.buffer_address() for t in tens]
    accs = []
    for t in tens:
        accs += _acc(t)
    mf_code = {None: MF_NONE, False: MF_NONE, MF_NONE: MF_NONE, "ids": MF_IDS, MF_IDS: MF_IDS,
               "dist": MF_DIST, MF_DIST: MF_DIST, "accept": MF_ACCEPT, MF_ACCEPT: MF_ACCEPT,
               "resid_p": MF_RESID_P, MF_RESID_P: MF_RESID_P, "noise": MF_NOISE, MF_NOISE: MF_NOISE}[mf]
    if mf_code in (MF_ACCEPT, MF_RESID_P) and rule is None or mf_code == MF_NOISE and not nz_mode:
        raise RuntimeError(f"{SSP_ENV}: must-fail code {mf_code} needs {'the rule' if mf_code != MF_NOISE else 'made noise'}")
    depth = int(noise.get("depth", 0)) if noise is not None else 0
    kernels = []
    for role, cfg in ((0, ttnn.WriterConfigDescriptor()), (1, ttnn.ReaderConfigDescriptor())):
        rt = ttnn.RuntimeArgs()
        for c in range(ncores):
            r0, r1 = bounds[2 * c + role]
            rt[c % gx][c // gx] = addrs + [r0, r1]
        ct = [CB_SCR, K, per, mf_code, 1 if override is not None else 0, 1 if debug else 0, role * scr,
              g_page, pages["scale"], pages["rank"], pages["topp"], pages["noise"], pages["flag"], pages["x"],
              pages["xf"], p_pr, p_id, p_sel, p_aux,
              1 if rule is not None else 0, nz_mode, 1 if table is not None else 0, depth]
        ct += [pg(n) for n in ("np", "qp", "qi", "u", "nr", "ac", "rs", "mp", "tb", "nd")] + accs
        kernels.append(ttnn.KernelDescriptor(kernel_source=_src("fs_cut.cpp"),
                                             source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                             core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg))
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[ttnn.CBDescriptor(
        total_size=2 * scr, core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.float32, page_size=2 * scr)])])
    io = [g, blk["scale"], blk["rank"], blk["topp"], blk["flag"]] + ([nz_t] if nz_mode == 0 else [])
    if override is not None:
        io += [x, xf]
    io += [ex[n][0] for n in ("np", "qp", "qi", "u", "nr", "mp", "tb") if n in ex]
    io += [probs, ids] + ([aux] if aux is not None else []) + ([acc, res] if acc is not None else [])
    io += ([nd] if nd is not None else []) + [sel]
    ttnn.generic_op(io, desc)
    out = {"probs": probs, "ids": ids, "sel": sel}
    if aux is not None:
        out["aux"] = aux
    if acc is not None:
        out["accept"], out["resid"] = acc, res
    if nd is not None:
        out["ndump"] = nd
    return out


def candidates(logits, per, mesh, tt_ccl, topology, blk, K, override=None, mf=None, debug=False, rule=None, noise=None,
               table=None):
    """The fused sampler call: {"probs", "ids", "sel"} (+ "aux" when debug, + "accept"/"resid" with the rule, + "ndump"
    when debug with made noise) from the vocab-shard TILE logits.  logits are NOT freed.  mf: None -> the env
    (QWEN36_MTP_SPEC_SAMPLE_FUSED_MUSTFAIL, then QWEN36_MTP_SPEC_SAMPLE_SSP_MUSTFAIL where it applies); else a mode or
    False.  rule / noise / table: the SSP switches (cut)."""
    from models.tt_transformers.tt.ccl import tt_all_gather

    M, K = int(logits.shape[2]), int(K)
    nd = _ndev(mesh)
    if nd != NDEV:
        raise RuntimeError(f"{ENV}: needs the {NDEV}-chip vocab-sharded mesh; got {nd} device(s)")
    if mf is None:
        mf = mustfail()
        if not mf and ssp_mustfail():
            code = ssp_mustfail()
            if (code in (MF_ACCEPT, MF_RESID_P) and rule is not None) or (code == MF_NOISE and noise is not None):
                mf = code
    mf = mf or None
    blockt = topk_block(logits, per, mesh, K, mf_topk=(mf == "topk"))
    gat = tt_all_gather(blockt, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    if _shape(gat) != [1, 1, M, 4 * K]:
        raise RuntimeError(f"{ENV}: gathered block {_shape(gat)}; expected [1, 1, {M}, {4 * K}]")
    cmf = {"ids": MF_IDS, "dist": MF_DIST}.get(mf, mf if mf in (MF_ACCEPT, MF_RESID_P, MF_NOISE, "accept", "resid_p",
                                                                "noise") else MF_NONE)
    out = cut_x(gat, blk, M, K, per, mesh, override=override, mf=cmf, debug=debug, rule=rule, noise=noise, table=table)
    if gat is not blockt:
        ttnn.deallocate(blockt)
    ttnn.deallocate(gat)
    return out


def serve_candidates(logits, per, mesh, tt_ccl, topology, blk, K, override=None, rule=None, noise=None, table=None):
    """The served call: refuses a (M, per) cell the boot self-test did not pass; logs the engaged banner per cell."""
    M = int(logits.shape[2])
    k = key(M, per, _ndev(mesh), override is not None)
    if k not in _tested:
        raise RuntimeError(f"{ENV}: (M={M}, per={per}, verify={override is not None}) was not self-tested this "
                           f"process (tested: {sorted(_tested)}); refusing to serve it")
    sw = (rule is not None, int(noise["mode"]) if noise is not None else 0, table is not None)
    if any(sw):
        if not ssp():
            raise RuntimeError(f"{SSP_ENV}: the SSP switches {sw} without the flag")
        if k not in _ssp_tested:
            raise RuntimeError(f"{SSP_ENV}: (M={M}, per={per}, verify={override is not None}) was not SSP self-tested "
                               f"this process (tested: {sorted(_ssp_tested)}); refusing to serve it")
    if k + sw not in _logged:
        _logged.add(k + sw)
        logger.info(f"QWEN36_MTP_SPEC_SAMPLE_FUSED engaged: M={M} per={per} {'verify' if override is not None else 'chain'}: fs_topk "
                    f"(TILE logits, {K}/chip) + 1 all-gather + fs_cut (cut, pick, greedy, select) = 3 programs"
                    f"{' MUSTFAIL ' + str(mustfail()) if mustfail() else ''}")
        if any(sw):
            logger.info(f"{SSP_ENGAGED_BANNER}: M={M} per={per} {'verify' if override is not None else 'chain'}: fs_cut "
                        f"{'+ the exact rule ' if sw[0] else ''}+ noise made on device (mode {sw[1]})"
                        f"{' + the draft-vocab id map' if sw[2] else ''}"
                        f"{' MUSTFAIL ' + _SSP_MF_RAW if ssp_mustfail() else ''}")
    return candidates(logits, per, mesh, tt_ccl, topology, blk, K, override=override, rule=rule, noise=noise,
                      table=table)


# ---------------------------------------------------------------------------------------------- reference + self-test
def served_reference(logits, per, mesh, tt_ccl, topology, blk, K, override=None):
    """The served chain, op for op (mtp_batch._verify_sample / _spec_candidates with the fused flag off, DRAFTPICK
    off): spec_sample.candidates, the greedy argmax (argmax_fold's combine, the served QWEN36_ARGMAX_FOLD path), the
    flag select and, in the verify, the draft override select.  logits NOT freed.  Returns {"probs", "ids", "sel",
    "pick", "greedy"}."""
    from models.demos.blackhole.qwen36.tt import argmax_fold, dev_sample, spec_sample

    c = spec_sample.candidates(logits, per, mesh, tt_ccl, topology, blk, K)
    greedy = argmax_fold._run(ttnn.clone(logits), int(logits.shape[2]), int(per), mesh, tt_ccl, topology, False)
    sel = dev_sample.select_ids(greedy, c["pick"], blk["flag"])
    if override is not None:
        x, xf = override
        xu = ttnn.typecast(x, ttnn.uint32)
        sel2 = dev_sample.select_ids(sel, xu, xf)
        ttnn.deallocate(xu)
        ttnn.deallocate(sel)
        sel = sel2
    return {"probs": c["probs"], "ids": c["ids"], "sel": sel, "pick": c["pick"], "greedy": greedy}


def planted_logits(M, ndev, per, K, g, ties=False, ktie=False):
    """bf16 [1, 1, M, ndev*per]: a low background (randn - 6, max ~ -1) and, per row, 2K DISTINCT planted values
    8 + n/16 (n = 0..2K-1, exactly representable in bf16) interleaved over the shards (shard d holds the n = d mod 2
    ones) at random columns, so every shard's top-K is unique and the merged order is unambiguous.  ties=True also
    plants exact ties at the row max: a cross-shard tie (row 0), an in-shard 3-way tie (row 1) and a 2-way tie at
    the max in shard 0 (row 2): the greedy (lowest id) rule; the probabilities then tie too (served order undefined).
    ktie=True: the values at global ranks 16..23 (both shards) are set EQUAL (the rank-18 value): a top_k=20 row must
    keep all 24 (vLLM keeps the ties at the k-th value); the order among them is undefined in the served path, so it is
    compared id-keyed (compare_keyed)."""
    import torch

    x = (torch.randn(1, 1, M, ndev * per, generator=g) - 6.0)
    C2 = 2 * K
    for m in range(M):
        for d in range(ndev):
            vals = torch.tensor([8.0 + v / 16.0 for v in range(C2) if v % ndev == d])
            if ktie:
                tie = 8.0 + (C2 - 19) / 16.0
                vals = torch.tensor([tie if C2 - 24 <= v <= C2 - 17 else 8.0 + v / 16.0
                                     for v in range(C2) if v % ndev == d])
            pos = torch.randperm(per, generator=g)[:vals.numel()] + d * per
            x[0, 0, m, pos] = vals
    if ties:
        top = 8.0 + (2 * K - 1) / 16.0 + 1.0
        if M >= 1 and ndev > 1:
            j = int(torch.randint(0, per, (1,), generator=g))
            for d in range(ndev):
                x[0, 0, 0, d * per + j] = top
        if M >= 2:
            js = torch.randperm(per, generator=g)[:3] + (ndev - 1) * per
            x[0, 0, 1, js] = top
        if M >= 3:
            js = torch.randperm(per, generator=g)[:2]
            x[0, 0, 2, js] = top
    return x.to(torch.bfloat16)


def _mixed_params(M):
    """The self-test's mixed batch: (temps, topks, topps) per row."""
    temps, topks, topps = [], [], []
    for m in range(M):
        kind = m % 4
        temps.append([1.0, 0.6, 0.0, 1.0][kind])
        topks.append([20, 8, 20, 0][kind])
        topps.append([0.95, 0.5, 0.95, 1.0][kind])
    return temps, topks, topps


def _up(mesh, t, dtype=None):
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if _ndev(mesh) > 1 else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype or ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep)


def _chips(mesh, t):
    import torch

    cs = ttnn.get_device_tensors(t) if _ndev(mesh) > 1 else [t]
    out = []
    for c in cs:
        v = ttnn.to_torch(c)
        out.append(v.to(torch.int64) if v.dtype in (torch.uint32, torch.int32) else v)
    return out


def compare(mesh, M, K, fused, ref):
    """Fused vs served outputs, every chip: ids bitwise, sel bitwise, probs max abs.  Returns a dict."""
    import torch

    C2 = 2 * int(K)
    res = {"ids_diff": 0, "sel_diff": 0, "probs_max_abs": 0.0, "probs_rows_over_tol": 0}
    for a, b in zip(_chips(mesh, fused["ids"]), _chips(mesh, ref["ids"])):
        res["ids_diff"] += int((a.reshape(M, C2).float() != b.reshape(M, C2).float()).sum())
    for a, b in zip(_chips(mesh, fused["sel"]), _chips(mesh, ref["sel"])):
        res["sel_diff"] += int((a.reshape(-1)[:M].to(torch.int64) != b.reshape(-1)[:M].to(torch.int64)).sum())
    for a, b in zip(_chips(mesh, fused["probs"]), _chips(mesh, ref["probs"])):
        d = (a.reshape(M, C2).double() - b.reshape(M, C2).double()).abs()
        res["probs_max_abs"] = max(res["probs_max_abs"], float(d.max()))
        res["probs_rows_over_tol"] += int((d.max(dim=1).values > PROB_TOL).sum())
    return res


def compare_keyed(mesh, M, K, fused, ref, rows=None):
    """Fused vs served where the served ORDER among equal values is undefined: per row the id SETS of the support
    must be equal and every id's probability within PROB_TOL; sel compared only on the given rows (greedy rows)."""
    C2 = 2 * int(K)
    res = {"support_diff_rows": 0, "probs_max_abs": 0.0, "sel_diff": 0}
    for fi, fp_, ri, rp in zip(_chips(mesh, fused["ids"]), _chips(mesh, fused["probs"]), _chips(mesh, ref["ids"]),
                               _chips(mesh, ref["probs"])):
        fi, fp_, ri, rp = (t.reshape(M, C2) for t in (fi, fp_, ri, rp))
        for m in range(M):
            fd = {int(i): float(p) for i, p in zip(fi[m].tolist(), fp_[m].tolist()) if p > 0}
            rd = {int(i): float(p) for i, p in zip(ri[m].tolist(), rp[m].tolist()) if p > 0}
            if set(fd) != set(rd):
                res["support_diff_rows"] += 1
                continue
            res["probs_max_abs"] = max(res["probs_max_abs"], max(abs(fd[i] - rd[i]) for i in fd))
    for a, b in zip(_chips(mesh, fused["sel"]), _chips(mesh, ref["sel"])):
        a, b = a.reshape(-1)[:M], b.reshape(-1)[:M]
        for m in (rows if rows is not None else range(M)):
            res["sel_diff"] += int(int(a[m]) != int(b[m]))
    return res


def _blk_host(M, K, temps, topks, topps, seed):
    from models.demos.blackhole.qwen36.tt import dev_sample

    blk = dev_sample.row_block([seed + m for m in range(M)], [100 + m for m in range(M)], temps, topks=topks,
                               topps=topps, K=K, M=M)
    return blk


def selftest(mesh, tt_ccl, topology, cells, K, seed=20261021):
    """Per (M, per, verify) cell: the fused path vs served_reference on planted device logits (mixed batch), every
    chip; the planted-tie greedy check (flag 0 everywhere: sel must equal the served greedy); the must-fail branch
    (+1) must differ.  Returns {tag: "exact" | "DIFF ...", tag + "_mustfail": n, tag + "_detail": {...}}."""
    import torch

    from models.demos.blackhole.qwen36.tt import argmax_fold

    ndev = _ndev(mesh)
    shard = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=3)} if ndev > 1 else {}
    res = {}
    for M, per, ovr in sorted({(int(a), int(b), bool(c)) for a, b, c in cells}):
        g = torch.Generator().manual_seed(seed + 131 * M + per + (7 if ovr else 0))
        tag = f"M{M}_P{per}_{'v' if ovr else 'c'}"
        temps, topks, topps = _mixed_params(M)
        bh = _blk_host(M, K, temps, topks, topps, seed + M)
        blk = {n: _up(mesh, v.reshape(1, 1, M, -1)) for n, v in bh.items()}
        override = None
        if ovr:
            xs = torch.tensor([[float((7919 * m) % (ndev * per))] for m in range(M)]).reshape(1, 1, M, 1)
            xf = torch.tensor([[1.0 if m % 3 == 1 else 0.0] for m in range(M)]).reshape(1, 1, M, 1)
            override = (_up(mesh, xs), _up(mesh, xf))
        x = planted_logits(M, ndev, per, K, g)
        lg = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard)
        ref = served_reference(lg, per, mesh, tt_ccl, topology, blk, K, override=override)
        fus = candidates(lg, per, mesh, tt_ccl, topology, blk, K, override=override)
        cmp_ = compare(mesh, M, K, fus, ref)
        bad = candidates(lg, per, mesh, tt_ccl, topology, blk, K, override=override, mf="ids")
        nm = sum(int((a.reshape(-1)[:M].to(torch.int64) != b.reshape(-1)[:M].to(torch.int64)).sum())
                 for a, b in zip(_chips(mesh, bad["sel"]), _chips(mesh, ref["sel"])))
        ok = cmp_["ids_diff"] == 0 and cmp_["sel_diff"] == 0 and cmp_["probs_rows_over_tol"] == 0
        res[tag] = "exact" if ok else f"DIFF {cmp_}"
        res[tag + "_mustfail"] = nm
        res[tag + "_detail"] = cmp_
        for t in (*fus.values(), *bad.values(), *ref.values(), lg):
            ttnn.deallocate(t)
        # top_k ties: ranks 16..23 equal; T=1 top_k=20 top_p=1 rows (all 24 kept) and greedy rows (flag 0)
        xk = planted_logits(M, ndev, per, K, g, ktie=True)
        lk = ttnn.from_torch(xk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard)
        kt = [1.0 if m % 2 == 0 else 0.0 for m in range(M)]
        bk_ = _blk_host(M, K, kt, [20] * M, [1.0] * M, seed + 77)
        blkk = {n: _up(mesh, v.reshape(1, 1, M, -1)) for n, v in bk_.items()}
        refk = served_reference(lk, per, mesh, tt_ccl, topology, blkk, K)
        fusk = candidates(lk, per, mesh, tt_ccl, topology, blkk, K)
        ck = compare_keyed(mesh, M, K, fusk, refk, rows=[m for m in range(M) if kt[m] == 0.0])
        sup = [int((p > 0).sum()) for p in _chips(mesh, fusk["probs"])[0].reshape(M, -1)]
        ck["support_t1_rows"] = sorted({sup[m] for m in range(M) if kt[m] > 0})
        okk = (ck["support_diff_rows"] == 0 and ck["probs_max_abs"] <= PROB_TOL and ck["sel_diff"] == 0
               and ck["support_t1_rows"] == [24])
        res[tag + "_ktie"] = "exact" if okk else f"DIFF {ck}"
        res[tag + "_ktie_detail"] = ck
        for t in (*fusk.values(), *refk.values(), lk, *blkk.values()):
            ttnn.deallocate(t)
        # the T=0 identity on planted exact ties at the max: every row greedy (flag 0), no override
        xt = planted_logits(M, ndev, per, K, g, ties=True)
        lt = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard)
        b0 = _blk_host(M, K, [0.0] * M, [0] * M, [1.0] * M, seed)
        blk0 = {n: _up(mesh, v.reshape(1, 1, M, -1)) for n, v in b0.items()}
        greedy = argmax_fold._run(ttnn.clone(lt), M, per, mesh, tt_ccl, topology, False)
        f0 = candidates(lt, per, mesh, tt_ccl, topology, blk0, K)
        nt = sum(int((a.reshape(-1)[:M].to(torch.int64) != b.reshape(-1)[:M].to(torch.int64)).sum())
                 for a, b in zip(_chips(mesh, f0["sel"]), _chips(mesh, greedy)))
        host = torch.argmax(xt[0, 0].float(), dim=-1)  # torch: the first (lowest) index of the max
        nh = sum(int((a.reshape(-1)[:M].to(torch.int64) != host).sum()) for a in _chips(mesh, f0["sel"]))
        res[tag + "_t0ties"] = "exact" if nt == 0 and nh == 0 else f"DIFF served={nt} torch={nh}"
        for t in (*f0.values(), greedy, lt, *blk0.values(), *blk.values()):
            ttnn.deallocate(t)
        if override is not None:
            for t in override:
                ttnn.deallocate(t)
    return res


def selftest_ok(res):
    checks = {k: v for k, v in res.items() if not k.endswith(("_mustfail", "_detail"))}
    return (bool(checks) and all(v == "exact" for v in checks.values())
            and all(int(v) > 0 for k, v in res.items() if k.endswith("_mustfail")))


def ensure_selftest(mesh, tt_ccl, topology, cells, K):
    """Once per process per cell, before any capture: fail closed.  cells: (M, per, verify) triples."""
    if not enabled():
        return
    ndev = _ndev(mesh)
    todo = sorted({(int(a), int(b), bool(c)) for a, b, c in cells if key(a, b, ndev, c) not in _tested})
    if not todo:
        return
    res = selftest(mesh, tt_ccl, topology, todo, K)
    served_mf = mustfail()
    if served_mf:
        raise RuntimeError(f"{MUSTFAIL_ENV}={_MF_RAW}: refusing the boot after the self-test ({res}): the served "
                           f"sampler is the wrong branch under this flag")
    if not selftest_ok(res):
        raise RuntimeError(f"{ENV} self-test REFUSED: {res}")
    for M, per, ovr in todo:
        _tested[key(M, per, ndev, ovr)] = "exact"
    worst = max((v["probs_max_abs"] for k, v in res.items() if k.endswith("_detail")), default=0.0)
    logger.info(f"QWEN36_MTP_SPEC_SAMPLE_FUSED self-test exact: cells (M, per, verify) {todo} on {ndev} chip(s), K={K}: ids and emitted ids "
                f"bitwise vs the served chain (spec_sample.candidates + argmax_fold + selects), probs max abs "
                f"{worst:.2e} (<= {PROB_TOL}), planted exact ties at the max take the served greedy id; the +1 branch "
                f"differs")
    logger.info(f"QWEN36_MTP_SPEC_SAMPLE_FUSED engaged: the sampled step's candidate block, cut, pick, greedy id and "
                f"selects run as fs_topk + 1 all-gather + fs_cut (3 programs per sampler call) at {todo}")


# ---------------------------------------------------------------------------------------------- SSP self-test
def _q_block(probs, ids, M, C2, extra_id, g):
    """A drafter block q on the SAME ids as the target block p (the exactness gate's rule: a q from another cell shares no
    ids with p and every draft is rejected): q ~ sqrt(p) on p's support plus one id OUTSIDE it (extra_id + m), and the
    draft x per row: q's argmax (m % 3 == 0), q's smallest support id (m % 3 == 1), the outside id (m % 3 == 2, p(x) = 0:
    a sure rejection).  Returns qp, qi [M, C2] fp32 and x [M, 1] fp32 (host)."""
    import torch

    qp = torch.zeros(M, C2, dtype=torch.float32)
    qi = torch.zeros(M, C2, dtype=torch.float32)
    xs = torch.zeros(M, 1, dtype=torch.float32)
    for m in range(M):
        sup = [j for j in range(C2) if float(probs[m, j]) > 0.0][: C2 - 1]  # room for the outside id
        w = torch.tensor([float(probs[m, j]) ** 0.5 for j in sup] + [0.15], dtype=torch.float64)
        w = w / w.sum()
        perm = torch.randperm(len(sup) + 1, generator=g).tolist()  # q's own column order differs from p's
        cid = [float(ids[m, j]) for j in sup] + [float(extra_id + m)]
        for c, src in enumerate(perm):
            qp[m, c] = float(w[src])
            qi[m, c] = cid[src]
        kind = m % 3
        if kind == 0:
            xs[m, 0] = cid[int(torch.argmax(w[:-1]))]
        elif kind == 1:
            xs[m, 0] = cid[int(torch.argmin(w[:-1]))]
        else:
            xs[m, 0] = cid[-1]
    return qp, qi, xs


def _np_verify(M, K, temps, seed):
    """Verify-side noise keys + the host noise blocks the served path would upload for them (row_block_fast's v_gum,
    noise_group's u / nres): (page, v_gum [M, 2K], u [M, 1], nres [M, 2K])."""
    from models.demos.blackhole.qwen36.tt import dev_sample, spec_sample

    C2 = 2 * K
    sds = [(seed * 7 + 13 * m) & 0x7FFFFFFF for m in range(M)]
    ssp_ = [(seed * 11 + 29 * m + 5) & 0x7FFFFFFF for m in range(M)]
    vpos = [(97 * m + seed) % 40000 - 1 for m in range(M)]  # includes position -1 (an inactive row's 0 - 1)
    page = noise_plan_page(sds, vpos, temps, ssp_)
    v_gum = dev_sample.row_block_fast(sds, vpos, temps, [0] * M, [1.0] * M, K=K, M=M)["noise"].reshape(M, C2)
    gv = spec_sample.noise_group([
        {"seeds": ssp_, "positions": vpos, "stream": spec_sample.STREAM_U, "width": 1, "kind": "u01"},
        {"seeds": ssp_, "positions": vpos, "stream": spec_sample.STREAM_RES, "width": C2, "kind": "expo"}])
    return page, v_gum, gv[0].reshape(M, 1), gv[1].reshape(M, C2)


def _np_chain(bw, K, temps, depth, mvals, seed):
    """Chain-side noise keys (+ the m page) and the host d_sgum{depth} block the served path would upload: (page, m page
    [1, 64] uint32 values, d_sgum [bw, 2K])."""
    import torch

    from models.demos.blackhole.qwen36.tt import spec_sample

    C2 = 2 * K
    cs = [(seed * 5 + 17 * b + 1) & 0x7FFFFFFF for b in range(bw)]
    p = [(331 * b + seed) % 30000 - 1 for b in range(bw)]  # p = -1 for an inactive row
    ct = [t if t > 0.0 else 0.0 for t in temps]
    page = noise_plan_page(c_seeds=cs, c_base=[x + 2 for x in p], c_temps=ct)
    mp = torch.zeros(1, 64, dtype=torch.int32)
    for b in range(bw):
        mp[0, b] = int(mvals[b])
    g = spec_sample.noise_group([{"seeds": cs, "positions": [p[b] + 2 + int(mvals[b]) + depth for b in range(bw)],
                                  "stream": spec_sample.STREAM_CHAIN + depth, "width": C2, "kind": "gumbel",
                                  "temps": ct}])[0]
    return page, mp, g.reshape(bw, C2)


def _bits_diff(mesh, a, b, n=None):
    import torch

    d = 0
    for x, y in zip(_chips(mesh, a), _chips(mesh, b)):
        x = x.reshape(-1)
        y = y.reshape(-1)
        if n is not None:
            x, y = x[:n], y[:n]
        if x.dtype == torch.float32:
            x, y = x.contiguous().view(torch.int32), y.contiguous().view(torch.int32)
        d += int((x.to(torch.int64) != y.to(torch.int64)).sum())
    return d


def ssp_selftest(mesh, tt_ccl, topology, cells, K, depths, table=None, map_ref=None, seed=20261022,
                 mutants=("noise",), dump=False):
    """QWEN36_MTP_SPEC_SAMPLE_SSP: per cell the SSP fs_cut against the S7 composition it replaces, every output bitwise
    on every chip.
      verify (M, per, True):  fs_cut(rule + made noise) vs fs_cut(host-built v_gum) + spec_sample.rule(host-built u,
                              nres) and vs spec_sample.host_rule (the rule's fp32 definition): probs, ids, sel, accept,
                              resid; the accept must take both values (both branches of the rule exercised)
      chain (bw, per, False): per depth k in depths, fs_cut(made noise from the m page [+ the id map]) vs fs_cut(host-
                              built d_sgum{k}) [+ map_ref, the served map]: probs, ids, sel
    The wrong-input branch (the noise keyed one position late) must differ; mutants adds the rule's wrong branches
    (verify: "accept" flips every accept, "resid_p" draws the residual from p), each of which must differ too.
    dump (the gate probe): the made noise itself (fs_cut's debug dump of EVERY column) bitwise vs the host blocks.
    Returns {tag: "exact" | "DIFF ...", tag_<mode>_mustfail: differing outputs, tag_detail: ...}."""
    import torch

    from models.demos.blackhole.qwen36.tt import spec_sample

    ndev = _ndev(mesh)
    shard = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=3)} if ndev > 1 else {}
    C2 = 2 * int(K)
    res = {}
    for M, per, ovr in sorted({(int(a), int(b), bool(c)) for a, b, c in cells}):
        g = torch.Generator().manual_seed(seed + 131 * M + per + (7 if ovr else 0))
        temps, topks, topps = _mixed_params(M)
        x = planted_logits(M, ndev, per, K, g)
        lg = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG, **shard)
        bh = _blk_host(M, K, temps, topks, topps, seed + M)
        if ovr:
            tag = f"M{M}_P{per}_v"
            page, v_gum, u, nres = _np_verify(M, K, temps, seed + M)
            blk = {n: _up(mesh, v.reshape(1, 1, M, -1)) for n, v in bh.items() if n != "noise"}
            blk_h = dict(blk, noise=_up(mesh, v_gum.reshape(1, 1, M, C2)))
            xf = torch.tensor([[1.0 if m % 3 == 1 else 0.0] for m in range(M)])
            # a first call for p's support (the q block shares its ids)
            xs0 = _up(mesh, torch.zeros(1, 1, M, 1))
            xf_t = _up(mesh, xf.reshape(1, 1, M, 1))
            p0 = candidates(lg, per, mesh, tt_ccl, topology, blk_h, K, override=(xs0, xf_t), mf=False)
            pr0, id0 = _chips(mesh, p0["probs"])[0].reshape(M, C2), _chips(mesh, p0["ids"])[0].reshape(M, C2)
            qp, qi, xs = _q_block(pr0, id0, M, C2, ndev * per + 1000, g)
            # x from the q block on drafted rows (xf), 0 elsewhere (as the host writes v_sx)
            xs = torch.where(xf > 0, xs, torch.zeros_like(xs))
            xs_t = _up(mesh, xs.reshape(1, 1, M, 1))
            dev = {"qp": _up(mesh, qp.reshape(1, 1, M, C2)), "qi": _up(mesh, qi.reshape(1, 1, M, C2)),
                   "u": _up(mesh, u.reshape(1, 1, M, 1)), "nres": _up(mesh, nres.reshape(1, 1, M, C2))}
            npt = _up(mesh, page, dtype=ttnn.int32)
            ref = candidates(lg, per, mesh, tt_ccl, topology, blk_h, K, override=(xs_t, xf_t), mf=False)
            rr = spec_sample.rule(ref["probs"], ref["ids"], dev["qp"], dev["qi"], xs_t, dev["u"], dev["nres"], mf=False)
            ha, hr = spec_sample.host_rule(_chips(mesh, ref["probs"])[0].reshape(M, C2), _chips(mesh, ref["ids"])[0].reshape(M, C2),
                                           qp, qi, xs, u, nres)
            cand = candidates(lg, per, mesh, tt_ccl, topology, blk, K, override=(xs_t, xf_t), mf=False,
                              rule={"qp": dev["qp"], "qi": dev["qi"]}, noise={"np": npt, "mode": 1})
            cand_dbg = None
            if dump:
                cand_dbg = candidates(lg, per, mesh, tt_ccl, topology, blk, K, override=(xs_t, xf_t), mf=False,
                                      debug=True, rule={"qp": dev["qp"], "qi": dev["qi"]}, noise={"np": npt, "mode": 1})
            det = {"probs": _bits_diff(mesh, cand["probs"], ref["probs"]), "ids": _bits_diff(mesh, cand["ids"], ref["ids"]),
                   "sel": _bits_diff(mesh, cand["sel"], ref["sel"], M),
                   "accept_vs_rule": _bits_diff(mesh, cand["accept"], rr["accept"], M),
                   "resid_vs_rule": _bits_diff(mesh, cand["resid"], rr["resid"], M)}
            hd_a = hd_r = 0
            for ca, cr in zip(_chips(mesh, cand["accept"]), _chips(mesh, cand["resid"])):
                hd_a += int((ca.reshape(-1)[:M].float() != ha.float()).sum())
                hd_r += int((cr.reshape(-1)[:M].to(torch.int64) != hr.to(torch.int64)).sum())
            det["accept_vs_host"], det["resid_vs_host"] = hd_a, hd_r
            acc_vals = sorted({float(v) for v in _chips(mesh, cand["accept"])[0].reshape(-1)[:M].tolist()})
            det["accept_values"] = acc_vals
            mfs = {}
            for mode in mutants:
                bad = candidates(lg, per, mesh, tt_ccl, topology, blk, K, override=(xs_t, xf_t), mf=mode,
                                 rule={"qp": dev["qp"], "qi": dev["qi"]}, noise={"np": npt, "mode": 1})
                mfs[mode] = (_bits_diff(mesh, bad["sel"], ref["sel"], M) + _bits_diff(mesh, bad["resid"], rr["resid"], M)
                             + _bits_diff(mesh, bad["accept"], rr["accept"], M))
                for t in bad.values():
                    ttnn.deallocate(t)
            if cand_dbg is not None:
                hv = torch.cat([v_gum.reshape(M, C2), nres.reshape(M, C2), u.reshape(M, 1)], dim=1)
                det["dump"] = _dump_diff(mesh, cand_dbg["ndump"], hv, M, temps, C2)
                det["dump_vs_cand"] = _bits_diff(mesh, cand_dbg["sel"], cand["sel"], M)
                for t in cand_dbg.values():
                    ttnn.deallocate(t)
            ok = all(det[k] == 0 for k in ("probs", "ids", "sel", "accept_vs_rule", "resid_vs_rule", "accept_vs_host",
                                           "resid_vs_host")) and acc_vals == [0.0, 1.0]
            if cand_dbg is not None:
                ok = ok and det["dump"]["diff"] == 0 and det["dump_vs_cand"] == 0
            res[tag] = "exact" if ok else f"DIFF {det}"
            res[tag + "_detail"] = det
            for mode, nm in mfs.items():
                res[f"{tag}_{mode}_mustfail"] = nm
            for t in (*p0.values(), *ref.values(), *rr.values(), *cand.values(), *dev.values(), xs0,
                      xs_t, xf_t, npt, blk_h["noise"]):
                ttnn.deallocate(t)
            for t in blk.values():
                ttnn.deallocate(t)
        else:
            blk = {n: _up(mesh, v.reshape(1, 1, M, -1)) for n, v in bh.items() if n != "noise"}
            for k in sorted({int(d) for d in depths}):
                tag = f"M{M}_P{per}_c_d{k}"
                mv = [int(v) for v in torch.randint(0, int(K) if int(K) > 0 else 1, (M,), generator=g)]
                page, mp, d_sgum = _np_chain(M, K, temps, k, mv, seed + 3 * M + k)
                npt = _up(mesh, page, dtype=ttnn.int32)
                mpt = _up(mesh, mp, dtype=ttnn.uint32)
                blk_h = dict(blk, noise=_up(mesh, d_sgum.reshape(1, 1, M, C2)))
                ref = candidates(lg, per, mesh, tt_ccl, topology, blk_h, K, mf=False)
                ref_sel = map_ref(ref["sel"]) if (table is not None and map_ref is not None) else ref["sel"]
                cand = candidates(lg, per, mesh, tt_ccl, topology, blk, K, mf=False, table=table,
                                  noise={"np": npt, "mode": 2, "depth": k, "m": mpt})
                bad = candidates(lg, per, mesh, tt_ccl, topology, blk, K, mf="noise", table=table,
                                 noise={"np": npt, "mode": 2, "depth": k, "m": mpt})
                dbg = None
                if dump:
                    dbg = candidates(lg, per, mesh, tt_ccl, topology, blk, K, mf=False, debug=True, table=table,
                                     noise={"np": npt, "mode": 2, "depth": k, "m": mpt})
                det = {"probs": _bits_diff(mesh, cand["probs"], ref["probs"]), "ids": _bits_diff(mesh, cand["ids"], ref["ids"]),
                       "sel": _bits_diff(mesh, cand["sel"], ref_sel, M), "map": table is not None and map_ref is not None}
                nm = _bits_diff(mesh, bad["sel"], ref_sel, M)
                ok = det["probs"] == 0 and det["ids"] == 0 and det["sel"] == 0
                if dbg is not None:
                    det["dump"] = _dump_diff(mesh, dbg["ndump"], d_sgum.reshape(M, C2), M, temps, C2)
                    ok = ok and det["dump"]["diff"] == 0
                res[tag] = "exact" if ok else f"DIFF {det}"
                res[tag + "_detail"] = det
                res[tag + "_noise_mustfail"] = nm
                to_free = [*ref.values(), *cand.values(), *bad.values(), npt, mpt, blk_h["noise"]]
                if dbg is not None:
                    to_free += list(dbg.values())
                if ref_sel is not ref["sel"]:
                    to_free.append(ref_sel)
                for t in to_free:
                    ttnn.deallocate(t)
            for t in blk.values():
                ttnn.deallocate(t)
        ttnn.deallocate(lg)
    return res


def _dump_diff(mesh, nd, host, M, temps, C2):
    """fs_cut's debug dump (the made noise, every column) vs the host block [M, W], bitwise per chip; a T == 0 row's
    zero noise compares by value (the host's g * 0.0 keeps g's sign, -0.0; the device writes +0.0: the same pick)."""
    import torch

    W = int(host.shape[1])
    out = {"n": 0, "diff": 0, "signed_zero_rows": 0}
    h = host.float().contiguous()
    for c in _chips(mesh, nd):
        d = c.reshape(-1)[: M * (2 * C2 + 16)].reshape(M, 2 * C2 + 16)[:, :W].float().contiguous()
        hz = h.clone()
        dz = d.clone()
        for m in range(M):
            if not float(temps[m]) > 0.0:
                n0 = int(((hz[m, :C2] == 0) & (hz[m, :C2].view(torch.int32) != 0)).sum())
                out["signed_zero_rows"] += 1 if n0 else 0
                hz[m, :C2] = hz[m, :C2] + 0.0
                dz[m, :C2] = dz[m, :C2] + 0.0
        out["n"] += M * W
        out["diff"] += int((dz.view(torch.int32) != hz.view(torch.int32)).sum())
    return out


def ssp_selftest_ok(res):
    checks = {k: v for k, v in res.items() if not k.endswith(("_mustfail", "_detail"))}
    return (bool(checks) and all(v == "exact" for v in checks.values())
            and all(int(v) > 0 for k, v in res.items() if k.endswith("_mustfail")))


def ensure_ssp_selftest(mesh, tt_ccl, topology, cells, K, depths, table=None, map_ref=None):
    """Once per process per cell, before any capture: fail closed.  cells: (M, per, verify) triples; depths: the chain
    depths served (0..K_draft-1); table / map_ref: the draft vocab's id table and its served map (dv.map_ids)."""
    if not ssp():
        return
    if not enabled():
        raise RuntimeError(f"{SSP_ENV}=1 needs {ENV}=1 (it extends the fused sampler's fs_cut)")
    ndev = _ndev(mesh)
    todo = sorted({(int(a), int(b), bool(c)) for a, b, c in cells if key(a, b, ndev, c) not in _ssp_tested})
    if not todo:
        return
    res = ssp_selftest(mesh, tt_ccl, topology, todo, K, depths, table=table, map_ref=map_ref)
    if ssp_mustfail():
        # the served branch is the wrong one: the comparison above ran the right one, so refuse explicitly (the probe's
        # MUSTFAIL cell serves it and must see this refusal)
        raise RuntimeError(f"{SSP_ENV} self-test REFUSED: {SSP_MUSTFAIL_ENV}={_SSP_MF_RAW} serves a wrong branch ({res})")
    if not ssp_selftest_ok(res):
        raise RuntimeError(f"{SSP_ENV} self-test REFUSED: {res}")
    for M, per, ovr in todo:
        _ssp_tested[key(M, per, ndev, ovr)] = "exact"
    logger.info(f"{SSP_SELFTEST_BANNER}: cells (M, per, verify) {todo}, chain depths {sorted(set(depths))}, on {ndev} "
                f"chip(s): fs_cut with the rule, the noise made on device{' and the id map' if table is not None else ''} "
                f"bitwise vs the S7 composition (host-built noise, spec_sample.rule, host_rule"
                f"{', map_ids' if table is not None else ''}); the late-noise branch differs "
                f"{ {k: v for k, v in res.items() if k.endswith('_mustfail')} }")
