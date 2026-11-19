# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_DEVICE_SAMPLE=1: device-side sampled decode over the TP-sharded vocab, MTP kept.

Why (desk, docs/analysis/agentic-ideas-20261020.md; the 2026-10-20 agentic bench): any step with
temperature > 0 leaves the device argmax (model_runner.py:2110 sends the WHOLE step to the host
sampler) AND leaves the MTP step (mtp_batch.py refuses temperature > 0).  Measured: 6.4 tok/s per
user at B=8 against ~63-67 greedy.  The host sampler reads B x 248320 bf16 back and runs a CPU
softmax + sort.

What: the pick is a Gumbel-max over a CANDIDATE SET, so it needs no sort and no CDF:
  per shard  the WIDE top-K over the row-major logits -> K candidates (values + local ids)
  gather    ONE all-gather per tensor (values, ids): the union of the two shards' top-K
  order     a top-K over the 2K merged candidates: the GLOBAL descending order (the top_k and the
            top_p cuts are prefixes of it)
  cut       top_k (a rank mask from the host) and top_p (a sorted cumulative cut inside the
            candidates), both as a per-row mask
  pick      argmax_i (logit_i + s_i) over the eligible candidates -> an exact sample from
            softmax(logits / T) over the candidate set, where s_i = T * gumbel_i
The Gumbel noise is supplied by the HOST as a [1, 1, M, 2K+1] fp32 block: columns 0..2K-1 are the
SCALED noise T*g (drawn from the request's own seed and the row's position, so the seed stays per
request and per position) and column 2K is the SELECT FLAG (1.0 = sample this row, 0.0 = greedy).
Why the noise is pre-scaled: argmax_i (logit_i / T + g_i) == argmax_i (logit_i + T * g_i) for
T > 0 (the scale is positive, so the argmax is unchanged), so the DEVICE never divides and never
needs a per-row broadcast of the temperature: the host folds T into the noise it sends.

THE WIDE TOP-K (the s-sample2 review's blocker 1): the served per-chip vocab is 248320/2 = 124160,
which is wider than the uint16 index path of ttnn.topk, so the WIDE op is used.
ttnn.experimental.topk_large_indices (tt-metal 0.77.0-dev20260813-76983d0,
ttnn/cpp/ttnn/operations/experimental/topk_large_indices/) takes a ROW_MAJOR BFLOAT16 tensor,
returns INDICES ONLY (a ROW_MAJOR UINT32 tensor, sorted descending) and is Blackhole-only; the
VALUES are therefore gathered back out of the same row-major logits with ttnn.gather.  The op
needs k in [16, 2048] and a multiple of 16, so K is a multiple of 16.

THE CUTS (the s-sample2 review's blocker 3): the device sampler implements the temperature, top_k
(top_k <= K) and top_p (a sorted cumulative cut inside the candidate set), so a request with
Qwen's default top_k=20 / top_p=0.95 stays on device.  The penalties (presence / frequency /
repetition) are NOT implemented: a step that asks for them falls back to the host sampler (the
plugin gate decides, per row).

THE ORDER OF THE CUTS (the s-sample3 review's blocker 2): vLLM samples in this order (vLLM 0.26,
vllm/v1/sample/sampler.py + vllm/v1/sample/ops/topk_topp_sampler.py):
    1. temperature:  logits / T
    2. top_k:        the logits below the k-th largest become -inf (the ties at the k-th value are
                     kept), so the softmax below sees ONLY the top_k survivors
    3. softmax:      over the survivors (the -inf tokens carry no mass)
    4. top_p:        the logits are sorted ASCENDING, probs_sum = cumsum(softmax(sorted)), and every
                     token whose probs_sum <= 1 - p is removed (the largest logit is never removed).
                     In descending rank terms: a candidate of rank r survives iff the cumulative
                     mass of the candidates ABOVE it is < p.
The device does exactly that: the host sends the per-row 1/T ("scale"), the device tempers the
candidate values, masks the non-top_k candidates to -3e38 (so they carry no mass in the softmax),
softmaxes, and cuts with the STRICT rule (the mass above < p).  The old code softmaxed the
UN-tempered values over ALL the candidates and cut with <=, which kept 20 of 20 where vLLM keeps
15 of 20 (Qwen's defaults top_k=20 / top_p=0.95 / T=0.6 on 0.1-spaced logits).  The host reference
(host_reference) computes vLLM's order INDEPENDENTLY (the full-vocabulary ascending-sort
formulation), never a mirror of the device formula.

THE SEED (the s-sample3 review's blocker 1): the seed the plugin hands over is an int32 (the
plugin's persistent batch keeps it in an int32 tensor), so it is folded to 31 bits
(fold_seed31) and never left unwritten: a request with no seed gets a fresh random 31-bit one, so
two identical unseeded prompts do not share the Gumbel noise.  The metal fallback (a row the
plugin did not seed) draws a random 31-bit seed per row too (it used to be 0, i.e. SHARED).

The candidate set is the union of the two shards' top-K, i.e. the top-K of each half of the vocab:
the sample is exact for softmax over that set and truncated against the full 248320 softmax (the
truncation is the documented tier-2 deviation; for an LM step the mass outside the top-2K is
negligible, but it is a deviation and it is stated as one).

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os

import ttnn
from loguru import logger

ENV = "QWEN36_DEVICE_SAMPLE"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
PROFILE_ENV = ENV + "_TEMP"
DEFAULT_K = 64  # per shard; a multiple of 16 (the wide top-K op's k constraint)
MAX_ROWS = 32
NEG = -3.0e38  # the "ineligible" value: far below any logit + noise
# the top_p threshold floor: vLLM removes every token whose cumulative mass reaches p, and for
# p <= 0 that is every token but the largest one.  The device's rule is the STRICT one (the mass
# above < p), so a p <= 0 would leave nothing; the host clamps the threshold to this floor, which
# keeps exactly the top candidate (the same set vLLM keeps).
TINY_P = 1e-30
# the seed mask: the device path's seed is an int32 (the plugin's persistent batch keeps it in an
# int32 tensor), so it is folded to 31 bits.  31 bits, not 32: the sign bit stays clear.
SEED_MASK31 = 0x7FFFFFFF
# the vocabulary width of the FLAT self-test scenario (the review's 0.1-spaced case): narrow, so the
# scenario is cheap on the card, and wide enough for the 20 planted candidates.
FLAT_PER = 1024

_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")

SELFTEST_BANNER = "QWEN36_DEVICE_SAMPLE self-test exact"
_logged = set()

# The WIDE TOP-K ORACLE (ttsim only).  ttnn.experimental.topk_large_indices is an LLK kernel and
# the simulator does NOT implement the SFPCONFIG the LLK emits ("tensix_sfpconfig: instr_mod1=8",
# tt-sim's SFPCONFIG supports instr_mod1 <= 1), so the op cannot be EXECUTED under ttsim at any
# width: the launch succeeds and the kernel aborts the simulator at the next sync.  The op is
# upstream tt-metal (not this lever's code) and its contract is a ROW_MAJOR UINT32 index tensor,
# sorted descending, of the per-row top-k of the ROW_MAJOR bf16 input.
#
# So the ttsim gate injects that tensor from the host (the oracle) and gates EVERYTHING ELSE at
# the SERVED per-chip vocab (124160): the values gather out of the row-major logits (the code the
# s-sample2 review flagged), the all-gather, the global sort, the top_k / top_p cuts, the
# Gumbel-max pick, the select and the MUSTFAIL.  The oracle is None on the card: the boot
# self-test there runs the REAL op.
#
# The device code does not depend on the wide op's tie-break: the merged candidates are re-sorted
# on device (ttnn.topk over the merged block), so only the CANDIDATE SET (which tokens) comes
# from the wide op, and a tie at the K-th value only changes the set, not the algorithm.
WIDE_TOPK_ORACLE = None
WIDE_TOPK_HOST = None

# The device-sample inputs of the MTP verify.  They live in the "v_" namespace ON PURPOSE: the
# pre-verify write is {n: v for n, v in vals.items() if n.startswith("v_")} (mtp_batch.py), so a
# name outside it would be written AFTER the verify and the verify would read the PREVIOUS step's
# noise and select flag (the s-sample2 review's blocker 2).
# v_gs is the per-row 1/T (the temperature, for vLLM's order: the device tempers the candidate
# values BEFORE the softmax).
INPUT_NAMES = ("v_gum", "v_guf", "v_gkr", "v_gp", "v_gs")


def set_wide_topk_oracle(fn):
    """Install the wide top-K oracle (ttsim only); None restores the real op."""
    global WIDE_TOPK_ORACLE
    WIDE_TOPK_ORACLE = fn


def set_wide_topk_host(logits_torch):
    """The ttsim oracle's host copy of the logits (the self-test sets it per cell)."""
    global WIDE_TOPK_HOST
    WIDE_TOPK_HOST = logits_torch


def enabled():
    return _MODE == "1"


def mustfail():
    return _MF == "1"


def k():
    """The per-shard top-K width of the candidate set."""
    try:
        kk = int(_tt_public_getenv(ENV + "_K", str(DEFAULT_K)))
    except ValueError as e:
        raise RuntimeError(f"{ENV}_K must be an int; got {_tt_public_getenv(ENV + '_K')!r}") from e
    if not 1 <= kk <= 1024:
        raise RuntimeError(f"{ENV}_K must be in 1..1024; got {kk}")
    if kk % 16:
        # the wide top-K op: k in [16, 2048] and a multiple of 16
        raise RuntimeError(f"{ENV}_K must be a multiple of 16 (the wide top-K op's k constraint); got {kk}")
    return kk


def profile_temperature():
    """The temperature this lever serves. 0.0 means 'greedy only' (never sample)."""
    try:
        t = float(_tt_public_getenv(PROFILE_ENV, "0.6"))
    except ValueError as e:
        raise RuntimeError(f"{PROFILE_ENV} must be a float; got {_tt_public_getenv(PROFILE_ENV)!r}") from e
    if not (0.0 < t <= 100.0):
        raise RuntimeError(f"{PROFILE_ENV} must be in (0, 100]; got {t}")
    return t


def _ndev(dev):
    return int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1


def _shape(t):
    return [int(x) for x in t.shape]


def _rm(t):
    """ROW_MAJOR view of t (a no-op when it already is).  The gather and the argmax take dim=-1 on
    a ROW_MAJOR tensor in the SERVED form (argmax_fold._run); some SFPU ops (softmax) return TILE."""
    if t.layout == ttnn.ROW_MAJOR_LAYOUT:
        return t
    return ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)


def check_rows(M):
    if not 1 <= int(M) <= MAX_ROWS:
        raise RuntimeError(f"{ENV}: M={M} outside 1..{MAX_ROWS}")


# ---------------------------------------------------------------------------------------------- host noise
_M1 = -4658895280553007687  # 0xBF58476D1CE4E5B9 (signed)
_M2 = -7697172588593208085  # 0x94D049BB133111EB (signed)
_M3 = 2685821657736338717   # 0x2545F4914F6CDD1D
_GOLD = -7046029254386353131  # 0x9E3779B97F4A7C15 (signed)


def fold_seed31(seed):
    """Fold any integer seed into a 31-bit seed (the int32 the device path carries).

    Why: the plugin's persistent batch keeps the seed in an int32 tensor (SEED_NONE_SENTINEL = -1
    for "no seed"), so a 63-bit seed raises "value cannot be converted to type int32 without
    overflow" on the write.  The s-sample3 plugin code drew a 63-bit seed and SWALLOWED that
    error, so the seed stayed the sentinel and every unseeded row shared seed 0 (two identical
    prompts, the same noise, the same tokens).

    Deterministic: the same 64-bit seed always folds to the same 31-bit value, so a seeded request
    is reproducible across boots and the fold is a pure function (no RNG state).
    """
    x = int(seed) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return int(x & SEED_MASK31)


def random_seed31():
    """A fresh random 31-bit seed (the int32 the device path carries)."""
    import random

    return int(random.getrandbits(31))


def _mix64(x):
    """splitmix64's finalizer on an int64 tensor (torch int64 arithmetic wraps like C)."""
    x = x ^ ((x >> 30) & 0x3FFFFFFFF)
    x = x * _M1
    x = x ^ ((x >> 27) & 0x1FFFFFFFFFF)
    x = x * _M2
    x = x ^ ((x >> 31) & 0x1FFFFFFFFFF)
    return x


def _u01(h):
    """The top 53 bits of h as a float64 in [0, 1)."""
    import torch

    return ((h >> 11) & 0x1FFFFFFFFFFFFF).to(torch.float64) * (2.0 ** -53)


def noise_matrix(seeds, positions, K=DEFAULT_K, temps=None, M=None):
    """The host's per-row noise block [M, 2K+1] fp32.

    seeds / positions: per-row (the request's seed and the row's absolute position).  temps: per-row
    temperature (a scalar or a per-row list); the noise is pre-scaled by it (T * gumbel) and the last
    column is the select flag (1.0 when T > 0).

    Deterministic: the same (seed, position) always gives the same noise block, so a request replayed
    with the same seed reproduces the same tokens.  Different rows of the same request differ (their
    positions differ), so the sampled tokens are not a repetition of one draw.
    """
    import torch

    K = int(K)
    seeds = [int(s) for s in (seeds if seeds is not None else [0])]
    positions = [int(p) for p in (positions if positions is not None else [0])]
    if len(seeds) != len(positions):
        raise RuntimeError(f"{ENV}: {len(seeds)} seeds but {len(positions)} positions")
    if M is None:
        M = len(seeds)
    M = int(M)
    if M != len(seeds):
        seeds = (seeds + [seeds[-1]] * M)[:M]
        positions = (positions + [positions[-1]] * M)[:M]
    if temps is None:
        temps = [profile_temperature()] * M
    elif isinstance(temps, (int, float)):
        temps = [float(temps)] * M
    else:
        temps = [float(t) for t in temps]
        if len(temps) != M:
            raise RuntimeError(f"{ENV}: {len(temps)} temperatures for {M} rows")

    s = torch.tensor(seeds, dtype=torch.int64)
    p = torch.tensor(positions, dtype=torch.int64)
    base = (s * _M3) ^ (p * _GOLD)
    cols = torch.arange(2 * K, dtype=torch.int64)
    h = _mix64(base.reshape(-1, 1) + cols.reshape(1, -1) * _M3)
    u = _u01(h).clamp_(1e-12, 1.0 - 1e-9)
    g = -torch.log(-torch.log(u))  # the Gumbel(0,1) noise
    t = torch.tensor(temps, dtype=torch.float64).reshape(-1, 1)
    noise = (g * t).to(torch.float32)  # T * g: the device never divides
    flag = torch.tensor([1.0 if tv > 0.0 else 0.0 for tv in temps], dtype=torch.float32).reshape(-1, 1)
    return torch.cat([noise, flag], dim=1).reshape(1, 1, M, 2 * K + 1)


def rank_mask(topks, K=DEFAULT_K, M=None):
    """The per-row top_k RANK mask [M, 2K] fp32: 1.0 for the ranks below the row's top_k, else 0.0.

    top_k <= 0 (or -1) means "no truncation": the row keeps every candidate (all ones).  The mask
    is built on the HOST because the RANK of a candidate is only known after the device's global
    sort, but the RULE (rank < top_k) is a host-side constant.

    It is a PREFIX mask (1.0 for the first top_k columns, 0.0 for the rest), and it is the same in
    the MERGED order (shard 0's K then shard 1's K) and in the GLOBAL descending order: the top_k
    cut is a prefix cut in both, so the device multiplies it with the sorted nucleus mask
    elementwise without permuting it.  (The candidate set contains the global top-k only when
    top_k <= K, the per-shard width; the plugin gate refuses a larger top_k.)
    """
    import torch

    K = int(K)
    if M is None:
        M = len(topks)
    M = int(M)
    tk = [int(t) for t in topks]
    if len(tk) != M:
        tk = (tk + [0] * M)[:M]
    out = torch.zeros((M, 2 * K), dtype=torch.float32)
    rank = torch.arange(2 * K, dtype=torch.float32)
    for m in range(M):
        out[m, :] = 1.0 if tk[m] <= 0 else (rank < float(tk[m])).to(torch.float32)
    return out.reshape(1, 1, M, 2 * K)


def row_block(seeds, positions, temps, topks=None, topps=None, K=DEFAULT_K, M=None):
    """The host's per-row block for one step: the tensors the device sampler reads.

    Returns a dict of torch tensors (each [1, 1, M, w] fp32):
      noise  [M, 2K]  the SCALED Gumbel noise T*g
      rank   [M, 2K]  the top_k rank mask (1.0 = keep)
      scale  [M, 2K]  the per-row 1/T (the temperature, for vLLM's order: the device tempers the
                      candidate values BEFORE the softmax).  A greedy row (T == 0) carries 1.0.
      topp   [M, 1]   the top_p threshold (1.0 = keep everything), floored at TINY_P (vLLM's rule
                      with p <= 0 keeps exactly the top candidate)
      flag   [M, 1]   the select flag (1.0 = sample this row, 0.0 = greedy)
    """
    import torch

    blk = noise_matrix(seeds, positions, K=K, temps=temps, M=M)
    M = int(blk.shape[2])
    if topks is None:
        topks = [0] * M
    if topps is None:
        topps = [1.0] * M
    topps = [float(t) for t in topps]
    if len(topps) != M:
        topps = (topps + [1.0] * M)[:M]
    if isinstance(temps, (int, float)):
        temps = [float(temps)] * M
    else:
        temps = [float(t) for t in temps]
        if len(temps) != M:
            raise RuntimeError(f"{ENV}: {len(temps)} temperatures for {M} rows")
    # the per-row 1/T: a greedy row (T == 0) carries 1.0 (its sample is discarded by the select,
    # and 1/0 would be a NaN in the tempered values)
    inv_t = [1.0 / (t if t > 0.0 else 1.0) for t in temps]
    scale = (torch.tensor(inv_t, dtype=torch.float32).reshape(M, 1)
             .expand(M, 2 * int(K)).contiguous().reshape(1, 1, M, 2 * int(K)))
    return {
        "noise": blk[:, :, :, :2 * K].contiguous(),
        "flag": blk[:, :, :, 2 * K:].contiguous(),
        "rank": rank_mask(topks, K=K, M=M),
        "scale": scale,
        "topp": torch.tensor([max(t, TINY_P) for t in topps], dtype=torch.float32).reshape(1, 1, M, 1),
    }


_PARAM_MEMO = {}


def row_block_fast(seeds, positions, temps, topks, topps, K=DEFAULT_K, M=None, noise=True):
    """row_block, BIT-IDENTICAL, for the lean host step (QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT): the int64 splitmix
    in numpy (it wraps like torch's), the float part in the same torch ops, and the per-request parameter blocks
    (rank mask, 1/T, top_p, flag) memoised by their parameters (they change only with the batch).  The served
    row_block loops over the rows in Python (rank_mask) and rebuilds every block every step.  The memoised
    tensors are returned as they are: the caller never writes into them (it hands them to ttnn.from_torch).
    noise=False (QWEN36_MTP_SPEC_SAMPLE_SSP: the device makes it): the parameter blocks only, "noise" is None."""
    import numpy as np
    import torch

    K, M = int(K), int(M) if M is not None else len(seeds)
    seeds = [int(x) for x in seeds]
    positions = [int(x) for x in positions]
    if len(seeds) != M or len(positions) != M:
        raise RuntimeError(f"{ENV}: row_block_fast needs {M} seeds and positions; got {len(seeds)}/{len(positions)}")
    temps = [float(t) for t in temps]
    topks = [int(t) for t in topks] if topks is not None else [0] * M
    topps = [float(t) for t in topps] if topps is not None else [1.0] * M
    if not noise:
        return {"noise": None, **_param_memo(K, M, temps, topks, topps)}
    s = np.asarray(seeds, dtype=np.int64)
    p = np.asarray(positions, dtype=np.int64)
    with np.errstate(over="ignore"):
        base = (s * np.int64(_M3)) ^ (p * np.int64(_GOLD))
        h = base.reshape(-1, 1) + np.arange(2 * K, dtype=np.int64).reshape(1, -1) * np.int64(_M3)
        h = h ^ ((h >> 30) & 0x3FFFFFFFF)
        h = h * np.int64(_M1)
        h = h ^ ((h >> 27) & 0x1FFFFFFFFFF)
        h = h * np.int64(_M2)
        h = h ^ ((h >> 31) & 0x1FFFFFFFFFF)
    u = _u01(torch.from_numpy(h)).clamp_(1e-12, 1.0 - 1e-9)
    g = -torch.log(-torch.log(u))
    t = torch.tensor(temps, dtype=torch.float64).reshape(-1, 1)
    noise = (g * t).to(torch.float32).reshape(1, 1, M, 2 * K).contiguous()
    return {"noise": noise, **_param_memo(K, M, temps, topks, topps)}


def _param_memo(K, M, temps, topks, topps):
    """row_block_fast's memoised per-request parameter blocks (flag, rank, scale, topp)."""
    import torch

    key = (K, M, tuple(temps), tuple(topks), tuple(topps))
    prm = _PARAM_MEMO.get(key)
    if prm is None:
        if len(_PARAM_MEMO) > 256:
            _PARAM_MEMO.clear()
        rank = torch.arange(2 * K, dtype=torch.float32).reshape(1, -1)
        kk = torch.tensor([float(k) if k > 0 else float(2 * K) for k in topks], dtype=torch.float32).reshape(-1, 1)
        inv_t = [1.0 / (tv if tv > 0.0 else 1.0) for tv in temps]
        prm = {
            "flag": torch.tensor([1.0 if tv > 0.0 else 0.0 for tv in temps], dtype=torch.float32).reshape(1, 1, M, 1),
            "rank": (rank < kk).to(torch.float32).reshape(1, 1, M, 2 * K).contiguous(),
            "scale": (torch.tensor(inv_t, dtype=torch.float32).reshape(M, 1).expand(M, 2 * K).contiguous()
                      .reshape(1, 1, M, 2 * K)),
            "topp": torch.tensor([max(tp, TINY_P) for tp in topps], dtype=torch.float32).reshape(1, 1, M, 1),
        }
        _PARAM_MEMO[key] = prm
    return prm


def gumbel_noise(seed, M, K, temp=1.0):
    """The host's Gumbel noise [1, 1, M, 2K] fp32: -log(-log(u)) with u ~ U(0,1) from a seeded
    generator, scaled by `temp` (the device math is argmax_i (v_i + T*g_i)). Deterministic given the
    seed (the per-request seed path)."""
    import torch

    g = torch.Generator().manual_seed(int(seed))
    u = torch.rand((int(M), 2 * int(K)), generator=g, dtype=torch.float32)
    u = u.clamp_(1e-9, 1.0 - 1e-7)
    return ((-torch.log(-torch.log(u))) * float(temp)).reshape(1, 1, int(M), 2 * int(K))


# ---------------------------------------------------------------------------------------------- device path
def _topk_wide(logits_rm, K):
    """The per-shard top-K over the ROW_MAJOR logits [1, 1, M, per] bf16.

    Returns (values [1,1,M,K] bf16, indices [1,1,M,K] uint32), both the LOCAL (in-shard) indices.

    ttnn.experimental.topk_large_indices takes a ROW_MAJOR BFLOAT16 tensor and returns INDICES ONLY
    (a ROW_MAJOR UINT32 tensor, sorted descending); the values are gathered back out of the same
    row-major logits.  It is the ONLY path at the served width (124160/chip): ttnn.topk's uint16
    index path wraps above 65535, and its auto-route to this composite is not emulated by ttsim
    (the routed pipeline's LLK path emits an SFPCONFIG the simulator does not implement), so the
    wide op is called explicitly."""
    if WIDE_TOPK_ORACLE is not None:
        # the ttsim oracle: the op's own contract (a ROW_MAJOR UINT32 index tensor, sorted
        # descending) injected from the host.  See the WIDE_TOPK_ORACLE note above.
        idx = WIDE_TOPK_ORACLE(logits_rm, int(K))
    else:
        try:
            idx = ttnn.experimental.topk_large_indices(logits_rm, k=int(K))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{ENV}: the wide top-K (ttnn.experimental.topk_large_indices) failed on a "
                               f"ROW_MAJOR bf16 [1, 1, M, {int(logits_rm.shape[3])}] tensor with k={K}: {e}") from e
    idx = _rm(idx)
    if int(idx.shape[-1]) != int(K):
        raise RuntimeError(f"{ENV}: the wide top-K returned {int(idx.shape[-1])} indices, expected {K}")
    vals = ttnn.gather(logits_rm, dim=-1, index=idx)
    return _rm(vals), idx


def sample(logits, per, mesh, tt_ccl, topology, blk, K=DEFAULT_K, mf=None, deallocate=True, ret_debug=False):
    """One sampled token id per row, from the vocab-SHARDED logits [1, 1, M, per] bf16.

    logits: the mesh tensor, device d holds the vocab range [d*per, (d+1)*per).
    blk:    the host's per-row block (a dict of mesh tensors): noise [1,1,M,2K] (the SCALED Gumbel
            noise T*g, REPLICATED), rank [1,1,M,2K] (the top_k rank mask), scale [1,1,M,2K] (the
            per-row 1/T), topp [1,1,M,1] (the top_p threshold) and flag [1,1,M,1] (the select flag).
    Returns a REPLICATED [1, 1, M, 1] uint32 tensor of GLOBAL token ids.

    vLLM's ORDER (the s-sample3 review's blocker 2): temperature -> top_k -> softmax -> top_p.
      * the TEMPERATURE: the host's per-row 1/T scales the candidate values, so the softmax below
        runs on logits/T (vLLM divides the logits by T before anything else);
      * the TOP_K: the candidates whose rank is >= top_k are pushed to -3e38 BEFORE the softmax, so
        they carry no mass (vLLM masks the non-top_k logits to -inf before its softmax);
      * the SOFTMAX over what is left;
      * the TOP_P: the cumulative mass of the candidates ABOVE (the exclusive cumsum of the
        descending order) must be STRICTLY below the threshold (vLLM removes a token whose
        ascending cumsum reaches 1 - p, i.e. it keeps rank r iff Q_{r-1} < p).
    The noise is PRE-SCALED by the host, so the pick never divides:
    argmax_i (v_i / T + g_i) == argmax_i (v_i + T * g_i) for T > 0.
    """
    from models.tt_transformers.tt.ccl import tt_all_gather

    M = int(logits.shape[2])
    per = int(per)
    K = int(K)
    NC = 2 * K
    check_rows(M)
    mf = mustfail() if mf is None else bool(mf)
    s = _shape(logits)
    if len(s) != 4 or s[:3] != [1, 1, M] or s[3] != per or logits.dtype != ttnn.bfloat16:
        raise RuntimeError(f"{ENV}: logits {s} {logits.dtype}; expected [1, 1, {M}, {per}] bf16")
    if per % 32:
        raise RuntimeError(f"{ENV}: per={per} must be a tile multiple")
    for name, want in (("noise", NC), ("rank", NC), ("scale", NC), ("topp", 1), ("flag", 1)):
        t = blk.get(name)
        if t is None:
            raise RuntimeError(f"{ENV}: the host block has no '{name}' tensor")
        ns = _shape(t)
        if len(ns) != 4 or ns[:3] != [1, 1, M] or ns[3] != want:
            raise RuntimeError(f"{ENV}: block '{name}' {ns}; expected [1, 1, {M}, {want}]")

    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    vals, idx = _topk_wide(rm, K)  # [1,1,M,K] each, per shard (the local indices)
    v32 = ttnn.typecast(vals, ttnn.float32)
    i32 = ttnn.typecast(idx, ttnn.float32)
    ttnn.deallocate(vals)
    ttnn.deallocate(idx)
    gv = tt_all_gather(v32, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    gi = tt_all_gather(i32, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    ttnn.deallocate(v32)
    ttnn.deallocate(i32)
    if _shape(gv) != [1, 1, M, NC]:
        raise RuntimeError(f"{ENV}: gathered values {_shape(gv)}; expected [1, 1, {M}, {NC}]")
    # The GLOBAL descending order of the 2K candidates: a top-K over the merged block.  The input
    # is fp32, so the stock (non-composite) topk path is used and the indices are UINT32.  The
    # topk wants a TILE input (the fill_pad under it refuses ROW_MAJOR), so the merged block is
    # converted; the block is small (M x 2K), so the conversion is cheap.
    gv_t = gv if gv.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(gv, ttnn.TILE_LAYOUT)
    sv_t, si_t = ttnn.topk(gv_t, NC, 3)
    sv = _rm(sv_t)
    si = _rm(si_t)
    gi_rm = _rm(gi)
    # 1. the TEMPERATURE: the host's per-row 1/T (a greedy row carries 1.0), so the softmax below
    #    runs on logits/T (vLLM's step 1).
    svT = ttnn.multiply(sv, blk["scale"])
    # 2. the TOP_K: the candidates the rank mask drops go to -3e38, so they carry NO mass in the
    #    softmax (vLLM masks the non-top_k logits to -inf before its softmax).  ttnn's binary ops
    #    take the TENSOR first, so (1 - mask) is (mask - 1) times -1: the sign matters.
    valsT = ttnn.add(ttnn.multiply(svT, blk["rank"]),
                     ttnn.multiply(ttnn.subtract(blk["rank"], 1.0), -NEG))
    # 3. the SOFTMAX over the top_k survivors.
    probs = ttnn.softmax(valsT, dim=-1)
    cum = ttnn.cumsum(probs, dim=-1)
    before = _rm(ttnn.subtract(cum, probs))
    # 4. the TOP_P: vLLM keeps the candidate of rank r iff the mass ABOVE it is < p (STRICT).
    keep = _rm(ttnn.lt(before, blk["topp"]))
    mask = _rm(ttnn.multiply(keep, blk["rank"]))
    # the noise follows the GLOBAL order: the column j of the noise belongs to the candidate that
    # ended up at rank si[..., j].
    nsort = ttnn.gather(blk["noise"], dim=-1, index=si)
    noisy = ttnn.add(sv, nsort)
    # The ineligible candidates are pushed far below every candidate.  ttnn's binary ops take the
    # TENSOR first, so (1 - mask) is (mask - 1) times -1: the sign matters (a (mask - 1) * NEG
    # would give the ineligible candidates the LARGEST value and the argmax would pick one).
    masked = ttnn.add(ttnn.multiply(noisy, mask),
                       ttnn.multiply(ttnn.subtract(mask, 1.0), -NEG))
    dbg = {"sv": sv, "si": si, "mask": mask, "probs": probs, "before": before} if ret_debug else None
    r = ttnn.typecast(ttnn.argmax(masked, dim=-1, keepdim=True), ttnn.uint32)  # [1,1,M,1]
    slot = ttnn.gather(si, dim=-1, index=r)  # the candidate column of the pick
    lid = ttnn.gather(gi_rm, dim=-1, index=slot)  # its LOCAL id
    # the gathered columns are shard 0's K then shard 1's K: a rank >= K means shard 1 -> +per
    off = ttnn.multiply(ttnn.ge(ttnn.typecast(slot, ttnn.float32), float(K)), float(per))
    gid = ttnn.add(lid, off)
    ids = ttnn.typecast(gid, ttnn.uint32)
    if mf:
        ids = ttnn.add(ids, 1)  # QWEN36_DEVICE_SAMPLE_MUSTFAIL: the self-test must REFUSE it
    for t in (sv, si, gi_rm, gv_t, svT, valsT, probs, cum, before, keep, mask, nsort, noisy, masked, r, slot, lid, off, gid):
        if dbg is not None and t is dbg.get("sv"):
            continue
        if dbg is not None and t is dbg.get("si"):
            continue
        if dbg is not None and t is dbg.get("mask"):
            continue
        if dbg is not None and t is dbg.get("probs"):
            continue
        if dbg is not None and t is dbg.get("before"):
            continue
        ttnn.deallocate(t)
    ttnn.deallocate(rm)
    if deallocate:
        ttnn.deallocate(logits)
    if ret_debug:
        return ids, dbg
    return ids


def select_ids(argmax_ids, sample_ids, flag):
    """The per-row select: the greedy ids where flag == 0, the sampled ids where flag == 1.

    flag: [1, 1, M, 1] fp32 (the host's 0/1 select flag).  All three tensors have the SAME shape, so
    the arithmetic below needs no broadcast: out = argmax + flag * (sample - argmax) in fp32 (the
    token ids are < 2^24, so fp32 is exact) and back to uint32.
    """
    M = int(argmax_ids.shape[2])
    for name, t in (("argmax", argmax_ids), ("sample", sample_ids)):
        s = _shape(t)
        if s != [1, 1, M, 1]:
            raise RuntimeError(f"{ENV}: {name} ids {s}; expected [1, 1, {M}, 1]")
    if _shape(flag) != [1, 1, M, 1]:
        raise RuntimeError(f"{ENV}: select flag {_shape(flag)}; expected [1, 1, {M}, 1]")
    a = ttnn.typecast(argmax_ids, ttnn.float32)
    s = ttnn.typecast(sample_ids, ttnn.float32)
    f = ttnn.typecast(flag, ttnn.float32)
    out = ttnn.add(a, ttnn.multiply(f, ttnn.subtract(s, a)))
    res = ttnn.typecast(out, ttnn.uint32)
    for t in (a, s, f, out):
        ttnn.deallocate(t)
    return res


# ---------------------------------------------------------------------------------------------- host reference
def host_reference(logits_torch, blk, K=DEFAULT_K, ndev=2, topks=None):
    """vLLM's sampling ORDER, computed INDEPENDENTLY of the device formula (the self-test's
    reference; it must never be a mirror of the device, or it cannot see a device-side mistake).

    This is a port of vLLM 0.26's own sampler
    (vllm/v1/sample/ops/topk_topp_sampler.py, apply_top_k_top_p_pytorch) on the FULL vocabulary
    row, in vLLM's own formulation:
      1. temperature: logits / T
      2. top_k: the logits below the k-th largest become -inf (the ties at the k-th value are kept)
      3. softmax over the survivors (the -inf tokens carry no mass)
      4. top_p: the logits are sorted ASCENDING, probs_sum = cumsum(softmax(sorted)), and every
         token whose probs_sum <= 1 - p is removed (the largest logit is never removed).
    The device instead cuts on the DESCENDING exclusive cumsum, so the two are different
    formulations of the same rule and a disagreement is a real disagreement.

    logits_torch: [1, 1, M, ndev*per] bf16 (the full vocab, the shards concatenated); blk: the
    host's per-row block (the dict from row_block); topks: the per-row top_k (the rule needs the
    NUMBER, not the rank mask).  Returns the global token ids [M] (int64), the per-candidate
    eligibility [M, 2K] and the per-candidate probability [M, 2K] (both in the device's
    merged-then-sorted order)."""
    import torch

    M = int(logits_torch.shape[2])
    ndev = int(ndev)
    per = int(logits_torch.shape[3]) // ndev
    x = logits_torch[0, 0].float()  # [M, ndev*per]
    neg = torch.finfo(torch.float32).min
    # 1. the TEMPERATURE (the block's scale is the per-row 1/T; a greedy row carries 1.0)
    temp = 1.0 / blk["scale"][0, 0][:, 0].reshape(M)
    lg = x / temp.reshape(M, 1)
    if topks is None:
        topks = [0] * M
    for m in range(M):
        k = int(topks[m])
        if k > 0:
            # 2. the TOP_K: the logits below the k-th largest become -inf
            kth = torch.topk(lg[m], k).values[-1]
            lg[m] = torch.where(lg[m] < kth, torch.full_like(lg[m], neg), lg[m])
    # 3. + 4. the softmax over the survivors and the nucleus cut, in vLLM's ascending form
    for m in range(M):
        p = float(blk["topp"][0, 0, m, 0])
        s, order = torch.sort(lg[m], descending=False)
        ps = torch.softmax(s, dim=-1)
        cum = torch.cumsum(ps, dim=-1)
        rm = cum <= (1.0 - p)
        rm[-1] = False  # the largest logit is never removed
        s[rm] = neg
        lg[m] = s.scatter(0, order, s)
    elig = (lg > (neg / 2)).to(torch.float32)  # [M, V] 1.0 = the token survives the cuts
    # the per-token probability (the full-vocabulary softmax), for the self-test's comparison: the
    # tokens whose probability underflows to 0 are the fp32 tail, where vLLM's ASCENDING cumsum
    # and the device's DESCENDING cumsum saturate at different points (a documented deviation: the
    # tail carries no mass, so it cannot change the sampled token).
    prob = torch.softmax(lg, dim=-1)
    # the merged candidates and their GLOBAL descending order (the device's order)
    cands, wts = [], []
    for d in range(ndev):
        v, i = torch.topk(x[:, d * per:(d + 1) * per], K, dim=-1)  # [M, K]
        cands.append(i + d * per)
        wts.append(v)
    idx = torch.cat(cands, dim=-1)  # [M, 2K] the GLOBAL ids, shard 0's K then shard 1's K
    vals = torch.cat(wts, dim=-1)  # [M, 2K]
    noise = blk["noise"][0, 0]  # [M, 2K]
    order = torch.argsort(vals, dim=-1, descending=True, stable=True)
    svals = torch.gather(vals, 1, order)
    sidx = torch.gather(idx, 1, order)
    snoise = torch.gather(noise, 1, order)
    selig = torch.gather(elig, 1, sidx)  # [M, 2K] the eligibility in the device's order
    sprob = torch.gather(prob, 1, sidx)  # [M, 2K] the reference's probability per candidate
    # the Gumbel-max pick over the eligible candidates
    noisy = svals + snoise
    masked = torch.where(selig > 0.5, noisy, torch.full_like(noisy, NEG))
    r = torch.argmax(masked, dim=-1)  # [M]
    picked = sidx[torch.arange(M), r]
    return picked, selig, sprob


# ---------------------------------------------------------------------------------------------- self-test
def _planted_rows(M, ndev, per, K, g):
    """M rows of random bf16 logits with K DISTINCT planted values per shard (the top-K of each
    shard), so the candidate order of a row is unique and the device's top-k order is the host's.

    Why plant instead of making every value distinct: bf16 has 8 mantissa bits, so a bounded range
    holds only a few hundred distinct values -- far fewer than the 248320 of the served vocab (a
    geometric spread over that many values overflows fp32). The tie only matters for the CANDIDATES
    (the top-K of each shard), so planting those is enough: the rest of the row may tie freely, and
    a tie between the two shards is broken the same way on both sides (shard 0 first).

    The two shards' planted values are DISJOINT (set A = 100+4k, set B = 98+4k, k = K..1), because
    the device MERGES the two top-K lists into one global order and a cross-shard tie would leave
    that order ambiguous (the device's tie-break is unspecified).  A and B interleave, so the
    merged order is not simply "shard 0 then shard 1"."""
    import torch

    x = (torch.randn(1, 1, M, ndev * per, generator=g)).to(torch.bfloat16)
    for m in range(M):
        for d in range(int(ndev)):
            # the even rows: shard 0 = A, shard 1 = B; the odd rows: swapped
            a_is_d0 = (m % 2 == 0) == (d == 0)
            base = 100.0 if a_is_d0 else 98.0
            planted = (torch.arange(int(K), 0, -1, dtype=torch.float32) * 4.0 + base)
            pos = torch.randperm(int(per), generator=g)[:int(K)] + int(d) * int(per)
            x[0, 0, m, pos] = planted.to(torch.bfloat16)
    return x


def _flat_rows(M, ndev, per):
    """M rows of FLAT logits: the global top-20 spaced 0.1 apart and the rest of the vocabulary far
    below (-50).  The s-sample3 review's discriminating case.

    With Qwen's defaults (top_k=20, top_p=0.95, T=0.6) on 0.1-spaced logits, vLLM keeps 15 of the
    20 candidates; the old device cut kept 18-20 (it softmaxed the UN-tempered values over the
    whole candidate set, so the cumulative mass never reached 0.95 inside the top 20 and the top_k
    cut alone bound).  The planted rows of _planted_rows are too peaked (their values are 4 apart)
    for the nucleus to say anything about the ORDER: their nucleus is one token either way.

    The 20 tokens are split over the shards (10 each), so the merge of the two per-shard top-K
    lists is exercised, and their values are distinct, so the global order is unambiguous (the
    tail ties at -50 are all BELOW the top_k cut, so their order does not matter).
    """
    import torch

    V = int(ndev) * int(per)
    x = torch.full((1, 1, M, V), -50.0, dtype=torch.float32)
    vals = torch.arange(20, 0, -1, dtype=torch.float32) * -0.1  # -0.1 ... -2.0
    ids = torch.cat([torch.arange(10), torch.arange(10) + int(per)])
    for m in range(M):
        x[0, 0, m, ids] = vals
    return x.to(torch.bfloat16)


def _case(mesh, tt_ccl, topology, x, temps, topks, topps, K, ndev, M, per, seed, tag):
    """One self-test case: the device sampler against the INDEPENDENT host reference.

    Returns a dict of the checks (tag, tag+"_mustfail", tag+"_select", tag+"_cut", tag+"_debug").
    """
    import torch

    blk_h = row_block([seed + m for m in range(M)], [1000 + m for m in range(M)], temps,
                      topks=topks, topps=topps, K=K, M=M)
    set_wide_topk_host(x)  # the ttsim oracle's host copy (the card uses the real op)
    mapper = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=3)} if ndev > 1 else {}

    def dev():
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **mapper)
        rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if ndev > 1 else {}
        return t, {n: ttnn.from_torch(v.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep)
                   for n, v in blk_h.items()}

    t, db = dev()
    ids, dbg = sample(t, per, mesh, tt_ccl, topology, db, K=K, mf=mustfail(), ret_debug=True)
    d_sv = ttnn.to_torch(ttnn.get_device_tensors(dbg["sv"])[0]).reshape(M, 2 * K)
    d_si = ttnn.to_torch(ttnn.get_device_tensors(dbg["si"])[0]).reshape(M, 2 * K).to(torch.int64)
    d_mask = ttnn.to_torch(ttnn.get_device_tensors(dbg["mask"])[0]).reshape(M, 2 * K)
    chips = ttnn.get_device_tensors(ids) if ndev > 1 else [ids]
    # EVERY chip is checked: the sampled ids are replicated, so every chip must carry the
    # same global ids and they must match the host reference.
    got = []
    for c in chips:
        got.append([int(v) for v in ttnn.to_torch(c).reshape(-1)[:M].tolist()])
    ttnn.deallocate(ids)
    t, db = dev()
    ids = sample(t, per, mesh, tt_ccl, topology, db, K=K, mf=True)
    chips = ttnn.get_device_tensors(ids) if ndev > 1 else [ids]
    bad = [int(v) for v in ttnn.to_torch(chips[0]).reshape(-1)[:M].tolist()]
    ttnn.deallocate(ids)
    # the select: the greedy rows (flag == 0) must take the argmax ids, the sampled rows the sample
    t, db = dev()
    samp = sample(t, per, mesh, tt_ccl, topology, db, K=K, deallocate=False)
    greedy = ttnn.from_torch(torch.full((1, 1, M, 1), 7, dtype=torch.int32), dtype=ttnn.uint32,
                             layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             **({"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if ndev > 1 else {}))
    sel = select_ids(greedy, samp, db["flag"])
    chips = ttnn.get_device_tensors(sel) if ndev > 1 else [sel]
    sgot = [int(v) for v in ttnn.to_torch(chips[0]).reshape(-1)[:M].tolist()]
    for x_ in (samp, greedy, sel):
        ttnn.deallocate(x_)
    ttnn.deallocate(t)
    for x_ in db.values():
        ttnn.deallocate(x_)

    # the INDEPENDENT reference (vLLM's own order, on the full vocabulary)
    ref, selig, sprob = host_reference(x, blk_h, K, ndev, topks=topks)
    ref = [int(v) for v in ref.tolist()]
    # the device's eligibility against the reference's: the CUT check.  This is the check that
    # sees a wrong ORDER (the old device cut the UN-tempered softmax and kept 20 of 20 where the
    # reference keeps 15 of 20).
    #
    # The comparison is over the candidates that carry probability mass: in the fp32 tail (the
    # probability underflows to 0) vLLM's ASCENDING cumsum and the device's DESCENDING cumsum
    # saturate at different points, so the two disagree about tokens that cannot be sampled.  That
    # is a documented deviation (the tail has no mass), and it is counted as a diagnostic.
    live = sprob > 0
    diff = (d_mask - selig).abs().gt(1e-6)
    ncut = int((diff & live).sum())
    ntail = int((diff & (~live)).sum())
    kept_dev = [int(d_mask[m].sum().item()) for m in range(M)]
    kept_ref = [int(selig[m].sum().item()) for m in range(M)]
    kept_live = [int((selig[m] * live[m]).sum().item()) for m in range(M)]
    # the debug: the device's merged order against the host's, and the two eligibility sets
    h_vals, h_order = _host_merged(x, K, ndev, per)
    dbg_rows = []
    for m in range(M):
        dbg_rows.append({
            "row": m,
            "sv_match": bool(torch.allclose(d_sv[m], h_vals[m], atol=1e-6)),
            "si_match": bool(torch.equal(d_si[m], h_order[m])),
            "mask_match": bool(torch.allclose(d_mask[m], selig[m], atol=1e-6)),
            "kept_dev": kept_dev[m],
            "kept_ref": kept_ref[m],
            "kept_ref_live": kept_live[m],
            "dev_mask_head": [round(float(v), 3) for v in d_mask[m][:6].tolist()],
            "ref_mask_head": [round(float(v), 3) for v in selig[m][:6].tolist()],
        })
    res = {f"{tag}_debug": dbg_rows}
    n = sum(1 for ci, g in enumerate(got) for a, b in zip(g, ref) if a != b)
    nm = sum(1 for a, b in zip(bad, ref) if a != b)
    # the select's expectation: the greedy rows (T == 0) take 7, the sampled rows the reference
    want = [ref[m] if temps[m] > 0 else 7 for m in range(M)]
    ns_ = sum(1 for a, b in zip(sgot, want) if a != b)
    res[tag] = "exact" if n == 0 else f"DIFF {n}"
    if n:
        # the diagnostic: which rows differ, and what the device picked against the host's
        det = []
        for m in range(M):
            for ci, g in enumerate(got):
                if g[m] == ref[m]:
                    continue
                row = x[0, 0, m].float()
                det.append({"row": m, "chip": ci, "dev": int(g[m]), "host": int(ref[m]),
                            "dev_val": round(float(row[int(g[m])]), 4),
                            "host_val": round(float(row[int(ref[m])]), 4),
                            "dev_shard": int(g[m]) // per, "host_shard": int(ref[m]) // per,
                            "dev_rank": int((row > row[int(g[m])]).sum()),
                            "host_rank": int((row > row[int(ref[m])]).sum())})
        res[tag + "_detail"] = det[:8]
    res[tag + "_mustfail"] = nm
    res[tag + "_select"] = "exact" if ns_ == 0 else f"DIFF {ns_}"
    res[tag + "_cut"] = "exact" if ncut == 0 else f"DIFF {ncut}"
    res[tag + "_kept"] = {"dev": kept_dev, "ref": kept_ref, "ref_live": kept_live,
                          "fp32_tail_disagreements": ntail}
    return res


def selftest(mesh, tt_ccl, topology, cells, K=DEFAULT_K, temp=0.6, seed=4242):
    """The device sampler against the INDEPENDENT host reference (vLLM's own order), per (M, per):
    the ids must match bitwise, the ELIGIBILITY MASK must match (the cuts, in vLLM's order) and the
    MUSTFAIL branch must differ.  Also checks the per-row SELECT (the greedy ids win on a
    flag == 0 row) and the host noise's determinism.

    Two scenarios per cell:
      * the PLANTED rows (a peaked distribution, the top_k / top_p cuts exercised);
      * the FLAT rows (the review's 0.1-spaced case at Qwen's defaults top_k=20 / top_p=0.95 /
        T=0.6, where vLLM keeps 15 of the 20 candidates and the old device cut kept 20) at a
        narrow vocabulary (per=1024), so it is cheap on the card.

    Returns {tag: "exact"|"DIFF n", tag+"_mustfail": n, tag+"_select": ..., tag+"_cut": ...,
    "noise_determinism": ...}."""
    import torch

    ndev = _ndev(mesh)
    res = {}
    for M, per in sorted(set((int(m), int(p)) for m, p in cells)):
        g = torch.Generator().manual_seed(seed + 31 * M + per)
        x = _planted_rows(M, ndev, per, K, g)
        # the per-row block: the rows alternate a sampled row (T = temp) and a greedy row (T = 0);
        # the sampled rows carry a top_k / top_p so the cuts are exercised.
        temps = [temp if (m % 2 == 0) else 0.0 for m in range(M)]
        topks = [0 for m in range(M)]
        topps = [1.0 for m in range(M)]
        for m in range(M):
            if m % 2 == 0:
                topks[m] = 0 if (m // 2) % 2 == 0 else 8      # 0 = no truncation, 8 = a tight cut
                topps[m] = 1.0 if (m // 2) % 2 == 0 else 0.5   # 0.5 = a tight nucleus
        res.update(_case(mesh, tt_ccl, topology, x, temps, topks, topps, K, ndev, M, per,
                         seed + M, f"M{M}_P{per}"))
    # the FLAT scenario (the review's case): Qwen's defaults on 0.1-spaced logits.
    for M, per in sorted(set((int(m), int(p)) for m, p in cells)):
        fper = FLAT_PER
        xf = _flat_rows(M, ndev, fper)
        temps = [0.6 for _ in range(M)]
        topks = [20 for _ in range(M)]
        topps = [0.95 for _ in range(M)]
        r = _case(mesh, tt_ccl, topology, xf, temps, topks, topps, K, ndev, M, fper,
                  seed + 7 * M + 1, f"M{M}_flat")
        # the review's number: vLLM keeps 15 of the 20 candidates on 0.1-spaced logits with
        # top_k=20 / top_p=0.95 / T=0.6 (the old device cut kept 18-20: it softmaxed the
        # UN-tempered values over the whole candidate set).
        kept = r.get(f"M{M}_flat_kept", {})
        kd = kept.get("dev", [])
        kr = kept.get("ref", [])
        ok15 = bool(kd) and bool(kr) and all(k == 15 for k in kd) and all(k == 15 for k in kr)
        r[f"M{M}_flat_nucleus15"] = "exact" if ok15 else f"DIFF dev={kd} ref={kr}"
        res.update(r)
    # the host noise's determinism: the same (seed, position) gives the same block
    b1 = noise_matrix([11, 12], [5, 6], K=8, temps=[0.6, 0.6])
    b2 = noise_matrix([11, 12], [5, 6], K=8, temps=[0.6, 0.6])
    b3 = noise_matrix([11, 12], [5, 7], K=8, temps=[0.6, 0.6])  # row 1's position moved
    b4 = noise_matrix([13, 12], [5, 6], K=8, temps=[0.6, 0.6])  # row 0's seed moved
    d_same = int((b1 != b2).sum())
    d_diff = int((b1[0, 0, 1, :] != b3[0, 0, 1, :]).sum()) + int((b1[0, 0, 0, :] != b4[0, 0, 0, :]).sum())
    res["noise_determinism"] = "exact" if (d_same == 0 and d_diff > 0) else f"DIFF {d_same}/{d_diff}"
    return res


def _host_merged(x, K, ndev, per):
    """The host's merged candidate values [M, 2K] and their descending order (a FACT about the
    candidate set, not a formula: the device's merged order must be this one)."""
    import torch

    M = int(x.shape[2])
    xs = x[0, 0].float()
    vals = torch.cat([torch.topk(xs[:, d * per:(d + 1) * per], K, dim=-1)[0] for d in range(ndev)], dim=-1)
    order = torch.argsort(vals, dim=-1, descending=True, stable=True)
    return torch.gather(vals, 1, order), order


def selftest_ok(res):
    """Every check must be "exact" and every MUSTFAIL rung must differ.  The "_debug", "_detail"
    and "_kept" keys are diagnostics (lists / dicts), not checks."""
    checks = {k: v for k, v in res.items()
              if not k.endswith(("_mustfail", "_debug", "_detail", "_kept"))}
    return (all(v == "exact" for v in checks.values())
            and all(int(v) > 0 for k, v in res.items() if k.endswith("_mustfail"))
            and not any(k.endswith("_detail") for k in res))


def ensure_selftest(mesh, tt_ccl, topology, cells, K=DEFAULT_K, temp=None):
    """Once per process: fail closed (the MUSTFAIL must differ)."""
    if not enabled():
        return
    temp = profile_temperature() if temp is None else temp
    res = selftest(mesh, tt_ccl, topology, cells, K=K, temp=temp)
    if not selftest_ok(res):
        raise RuntimeError(
            f"{ENV} self-test REFUSED{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}")
    logger.info(f"{SELFTEST_BANNER}: cells {sorted(set((int(m), int(p)) for m, p in cells))} on {_ndev(mesh)} chip(s), "
                f"bitwise vs vLLM's own sampling order (temperature -> top_k -> softmax -> top_p) at "
                f"T={temp} K={K}; the wrong-input branch differs")
    if ENV not in _logged:
        _logged.add(ENV)
        logger.info(f"QWEN36_DEVICE_SAMPLE engaged: device-side sampled decode over the TP-sharded vocab "
                    f"(the wide top-K {K}/shard + one all-gather + a Gumbel-max pick over the host's T*g "
                    f"noise, with the top_k/top_p cuts on device in vLLM's order: temperature, then top_k, "
                    f"then the softmax, then top_p), T={temp}; "
                    f"the drafts stay greedy and the acceptance is a coincidence test, so the emitted "
                    f"distribution is exact (the sample-then-compare scheme)")
