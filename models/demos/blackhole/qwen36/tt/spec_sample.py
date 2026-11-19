# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_SPEC_SAMPLE=1: exact speculative sampling (Leviathan et al. / Chen et al. 2023) for the
MTP drafter at T > 0.

Why: the device sampler (dev_sample.py, QWEN36_DEVICE_SAMPLE) serves T > 0 with the COMPARE rule:
it samples the target at every verify row and accepts a GREEDY draft only when it equals that
sample. That is exact, but the acceptance is the coincidence rate P(s ~ p == argmax q), which is
low at T = 1.0. The EXACT rule accepts a SAMPLED draft x ~ q with probability min(1, p(x)/q(x)),
1 - TV(p, q) on average:

    for j = 0 .. n-1:            x_j ~ q_j (the chain sampled it), p_j = the verify's row j
        accept iff u_j * q_j(x_j) < p_j(x_j)           (u_j ~ U[0,1); no division)
        first rejection at m: emit x_0..x_{m-1}, then r ~ norm((p_m - q_m)^+)
    all n accepted: emit x_0..x_{n-1}, then a fresh sample from p_n (the verify's own device sample)

    P(emit y) = min(p(y), q(y)) + (1 - sum min(p, q)) * (p(y) - q(y))^+ / sum (p - q)^+ = p(y)

The verify's row j is fed draft j-1 (mtp_batch_host.plan: verify_tokens), so p_j and q_j are
conditioned on the same prefix: the rule's precondition holds.

THE DISTRIBUTIONS. p and q are both the vLLM order on a candidate block: temperature -> top_k ->
softmax -> top_p (keep rank r iff the mass above it is < top_p), renormalised over the support. The
block is the per-shard wide top-K (K per chip) merged over the two chips and sorted (2K = C2
candidates per row). With top_k <= K the block contains the support exactly (the plugin's device-
sampling gate refuses top_k > K); with top_k <= 0 and top_p < 1 the nucleus is truncated to 2K
candidates (the one documented deviation). Both p and q come from cut() on the device, and the
drafter's sample x is a Gumbel-max pick over the same masked block, so x ~ q exactly (up to fp32).

THE SOFTMAX. cut() does NOT use ttnn.softmax: on a [32,128] fp32 block it is off by up to 0.024 in
probability (ttsim, default compute config; 0.0025 with HiFi4 + fp32 accumulation), which was the
0.035 device-vs-host gap of the m-specsample3 review. The stable form here (subtract the row max,
exp, a TILE-layout sum, divide) is within 1e-7 ([source validation record]). Reductions run in
TILE layout: a ROW_MAJOR ttnn.sum is off by ~4e-4 relative.

THE JOIN. p is on the target's ids, q on the drafter's (carried from the previous step's chain, the
draft-vocab sub-ids mapped to full ids on the host). eq[i,j] = (pi_i == qi_j) gives q at each target
candidate for the residual; p(x) and q(x) are one-hot sums against x.

THE NOISE is host-made (splitmix keyed by the request's seed, the predicted position and a stream):
the chain's Gumbel noise (streams 20 + depth, pre-scaled by T), the accept uniform (stream 11) and the
residual's Exp(1) noise (stream 12; the residual draw is the exponential race argmin E_i / r_i, which
needs no log on the device). The target's own sample keeps dev_sample's verify noise (v_gum).

THE RULE IS PER REQUEST: vllm_xargs={"tt_spec_rule": "exact" | "compare"} in the request body
(platform.validate_request refuses any other value with a 400), else the arm's default
QWEN36_MTP_SPEC_SAMPLE_RULE. The plugin hands it down per row as sampling_params.spec_rule. A T = 0
row is greedy under both rules (the argmax ids, byte-identical to the flag off).

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import math
import os

import ttnn
from loguru import logger

ENV = "QWEN36_MTP_SPEC_SAMPLE"
SERVE_ENV = ENV + "_SERVE"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
K_ENV = ENV + "_K"
RULE_ENV = ENV + "_RULE"
FAST_ENV = ENV + "_HOSTNOISE"
FAST_MUSTFAIL_ENV = FAST_ENV + "_MUSTFAIL"
DEFAULT_K = 64
MAX_ROWS = 4096  # served verify rows are <= 32; the statistical gate batches more
NEG = -3.0e38  # "ineligible" in an argmax
CUT_NEG = -1.0e4  # the masked logit inside the softmax: exp(-1e4) == 0 in fp32 (ttsim: the SFPU exp of -3e38 is not 0)
RULE_EXACT = 1
RULE_COMPARE = 0
RULE_NAMES = {"exact": RULE_EXACT, "compare": RULE_COMPARE}
# host noise streams (independent splitmix streams per role)
STREAM_U = 11
STREAM_RES = 12
STREAM_CHAIN = 20  # + depth

SELFTEST_BANNER = "QWEN36_MTP_SPEC_SAMPLE self-test exact"
SERVE_BANNER = "QWEN36_MTP_SPEC_SAMPLE serving engaged"


def _flag(name):
    v = _tt_public_getenv(name, "0")
    if v not in ("0", "1"):
        raise RuntimeError(f"{name} must be 0 or 1; got {v!r}")
    return v == "1"


_MODE = _flag(ENV)
_SERVE = _flag(SERVE_ENV)
_MF = _flag(MUSTFAIL_ENV)
_FAST = _flag(FAST_ENV)
_FAST_MF = _flag(FAST_MUSTFAIL_ENV)

# ---------------------------------------------------------------------------------------------- the lean step (opus/sscost)
# Four levers that cut the sampled step's overhead WITHOUT changing the served distribution (docs: LEDGER
# an internal record).  Each is default off, logs its own banner and is self-tested at boot (fail closed).
#   HOSTSPLIT  the host builds only the blocks each write sends: the pre-verify call built the chain's noise and
#              the post-accept call rebuilt the verify's (both thrown away), so every step built every block twice.
#              The written values are the same objects' values (tier 1, bit-identical).
#   WSKIP      a sampler parameter block (1/T, the top_k mask, the top_p threshold, the flags) is not re-sent when it
#              equals what the device buffer already holds: they change only when the batch does.  Tier 1.
#   RMSHARE    the verify untilizes its logits ONCE and hands the row-major tensor to both the wide top-K and the
#              greedy argmax (argmax_fold.greedy_rm), which untilized the same logits a second time.  Tier 1.
#   DRAFTPICK  the drafter's non-exact rows (T = 0, or the compare rule) take the Gumbel-max pick with ZERO noise,
#              i.e. the block's argmax, instead of a second full-vocab argmax + a select.  The draft can differ
#              from the lowest-id argmax only on an exact tie at the row max; a draft never decides an emitted
#              token for those rows (greedy: the verify's argmax; compare: the verify's sample), so the served
#              output is unchanged and only acceptance can move on such a tie.  Exact rows are untouched.
HOSTSPLIT_ENV = ENV + "_HOSTSPLIT"
HOSTSPLIT_MUSTFAIL_ENV = HOSTSPLIT_ENV + "_MUSTFAIL"
WSKIP_ENV = ENV + "_WSKIP"
RMSHARE_ENV = ENV + "_RMSHARE"
DRAFTPICK_ENV = ENV + "_DRAFTPICK"
LEAN_MUSTFAIL_ENV = ENV + "_LEAN_MUSTFAIL"
_HOSTSPLIT = _flag(HOSTSPLIT_ENV)
_HOSTSPLIT_MF = _flag(HOSTSPLIT_MUSTFAIL_ENV)
_WSKIP = _flag(WSKIP_ENV)
_RMSHARE = _flag(RMSHARE_ENV)
_DRAFTPICK = _flag(DRAFTPICK_ENV)
_LEAN_MF = _flag(LEAN_MUSTFAIL_ENV)
# QWEN36_MTP_SPEC_SAMPLE_DUAL (docs/MTP-BATCH.md section 26): a greedy trace family (S6's ops, no sampler program)
# captured per bucket beside the sampled one; a step whose active rows are all T = 0 replays the greedy family.
# _MUSTFAIL: a step with exactly one sampled row runs the greedy family (that row emits argmax: a distribution gate
# must fail).  Read at import like every SPEC_SAMPLE flag (a reboot-class switch).
DUAL_ENV = ENV + "_DUAL"
DUAL_MUSTFAIL_ENV = DUAL_ENV + "_MUSTFAIL"
_DUAL = _flag(DUAL_ENV)
_DUAL_MF = _flag(DUAL_MUSTFAIL_ENV)
# the blocks WSKIP may skip: written by the host only, read by the traces only, and constant per request
WSKIP_NAMES = frozenset(("v_guf", "v_gkr", "v_gp", "v_gs", "d_sscale", "d_srank", "d_stopp", "d_sflag"))


def host_split():
    return _MODE and _SERVE and _HOSTSPLIT


def host_split_mustfail():
    return _HOSTSPLIT_MF


def wskip():
    return _MODE and _SERVE and _WSKIP


def rm_share():
    return _MODE and _SERVE and _RMSHARE


def draft_pick():
    return _MODE and _SERVE and _DRAFTPICK


def lean_mustfail():
    return _LEAN_MF


def dual():
    """QWEN36_MTP_SPEC_SAMPLE_DUAL is set (the boot refuses it without the served rule: mtp_batch.compile)."""
    return _DUAL


def dual_mustfail():
    return _DUAL_MF


def log_lean_banners():
    """One banner per engaged lever (literal strings: ttx lint scans the logger calls)."""
    if not serve() or "lean" in _logged:
        return
    _logged.add("lean")
    if _HOSTSPLIT:
        logger.info("QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT engaged: each host write builds only the blocks it sends (the "
                    "verify's before the verify, the chain's after the accept); bit-identical values")
    if _WSKIP:
        logger.info("QWEN36_MTP_SPEC_SAMPLE_WSKIP engaged: an unchanged sampler parameter block (1/T, top_k mask, "
                    "top_p, flags) is not re-sent; the device buffer already holds those bytes")
    if _RMSHARE:
        logger.info("QWEN36_MTP_SPEC_SAMPLE_RMSHARE engaged: the verify untilizes its logits once for the wide top-K "
                    "and the greedy argmax (argmax_fold.greedy_rm); bit-identical ids")
    if _DRAFTPICK:
        logger.info("QWEN36_MTP_SPEC_SAMPLE_DRAFTPICK engaged: the drafter's non-exact rows take the zero-noise pick "
                    "(the block argmax) instead of a second full-vocab argmax and a select; the emitted output of "
                    "those rows never depends on the draft")
_logged = set()


def enabled():
    return _MODE


def serve():
    """The served wiring (chain carry, device rule in the verify, host accept by the rule) is on."""
    return _MODE and _SERVE


def mustfail():
    return _MF


def k():
    """The candidate width per shard (the merged block is 2K)."""
    try:
        kk = int(_tt_public_getenv(K_ENV, str(DEFAULT_K)))
    except ValueError as e:
        raise RuntimeError(f"{K_ENV} must be an int; got {_tt_public_getenv(K_ENV)!r}") from e
    if not 16 <= kk <= 1024 or kk % 16:
        raise RuntimeError(f"{K_ENV} must be a multiple of 16 in 16..1024 (the wide top-K op); got {kk}")
    return kk


def parse_rule(value):
    """'exact' | 'compare' | 1 | 0 -> 1 | 0; None -> None. Anything else raises."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{ENV}: the rule must be 'exact' or 'compare'; got {value!r}")
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in RULE_NAMES:
        return RULE_NAMES[value.strip().lower()]
    raise RuntimeError(f"{ENV}: the rule must be 'exact' or 'compare'; got {value!r}")


def default_rule():
    """The arm's default rule (QWEN36_MTP_SPEC_SAMPLE_RULE, default 'exact')."""
    v = _tt_public_getenv(RULE_ENV, "exact")
    if v.strip().lower() not in RULE_NAMES:
        raise RuntimeError(f"{RULE_ENV} must be 'exact' or 'compare'; got {v!r}")
    return RULE_NAMES[v.strip().lower()]


def row_rules(sp, n):
    """The per-row rule flags of a decode step: sampling_params.spec_rule (the plugin's per-request
    resolution, TTSpecSamplingParams) else the arm's default for every row."""
    rr = getattr(sp, "spec_rule", None) if sp is not None else None
    if rr is None:
        return [default_rule()] * n
    vals = [int(v) for v in (rr.tolist() if hasattr(rr, "tolist") else rr)]
    out = []
    for i in range(n):
        out.append(parse_rule(vals[i]) if i < len(vals) else default_rule())
    return out


def log_serve_banner(K):
    if not serve() or "serve" in _logged:
        return
    _logged.add("serve")
    logger.info(
        f"{SERVE_BANNER}: exact speculative sampling in the served MTP path: the chain samples x ~ q (wide top-K "
        f"{int(K)}/shard, vLLM order, the request's own T/top_k/top_p), the verify tests u*q(x) < p(x) on device "
        f"and a rejection draws from norm((p-q)^+); per request vllm_xargs['tt_spec_rule'] ('exact'|'compare'), "
        f"else the arm default {RULE_ENV}={'exact' if default_rule() else 'compare'}; T=0 rows stay greedy"
    )


def log_fast_noise_banner():
    """The fast host-noise path's banner (once per process)."""
    if not _FAST or "fastnoise" in _logged:
        return
    _logged.add("fastnoise")
    logger.info(
        "QWEN36_MTP_SPEC_SAMPLE_HOSTNOISE engaged: the per-step host noise blocks are built with ONE hash per "
        "group of streams (numpy for the int64 splitmix64, torch for the float part) instead of one torch hash "
        "per stream; the values are BIT-IDENTICAL (the hash input per (row, stream, column) is unchanged), and "
        "the host cost of the step's noise drops from ~20 torch dispatches per stream to ~13 numpy dispatches "
        "per group (the device is idle while the host builds them)"
    )


def _shape(t):
    return [int(x) for x in t.shape]


def _rm(t):
    return t if t.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)


def _tile(t):
    return t if t.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def _ndev(dev):
    return int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1


def _check_rows(M):
    if not 1 <= int(M) <= MAX_ROWS:
        raise RuntimeError(f"{ENV}: M={M} outside 1..{MAX_ROWS}")


def _free(*ts):
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


# ---------------------------------------------------------------------------------------------- host noise
# splitmix64 multipliers as positive 63-bit ints (torch int64 arithmetic wraps like C)
_M1 = 0x3F58476D1CE4E5B9
_M2 = 0x149D049BB133111
_M3 = 0x2545F4914F6CDD1D
_GOLD = 0x1E3779B97F4A7C15


def _mix64(x):
    x = x ^ ((x >> 30) & 0x3FFFFFFFF)
    x = x * _M1
    x = x ^ ((x >> 27) & 0x1FFFFFFFFFF)
    x = x * _M2
    x = x ^ ((x >> 31) & 0x1FFFFFFFFFF)
    return x


def _u01(seeds, positions, C, stream):
    """U(0,1) float64 [M, C], keyed by (seed, position, stream, column)."""
    import torch

    s = torch.as_tensor([int(x) for x in seeds], dtype=torch.int64)
    p = torch.as_tensor([int(x) for x in positions], dtype=torch.int64)
    base = (s * _M3) ^ (p * _GOLD) ^ ((int(stream) * _M1) & 0x7FFFFFFFFFFFFFFF)
    cols = torch.arange(int(C), dtype=torch.int64).reshape(1, -1)
    h = _mix64(base.reshape(-1, 1) + cols * _M3)
    return (((h >> 11) & 0x1FFFFFFFFFFFFF).to(torch.float64) * (2.0 ** -53)).clamp_(1e-12, 1.0 - 1e-9)


def gumbel(seeds, positions, C, stream, temps=None):
    """Gumbel(0,1) noise [M, C] fp32, pre-scaled per row by temps (T*g: the device pick is argmax(v + T*g))."""
    import torch

    g = -torch.log(-torch.log(_u01(seeds, positions, C, stream)))
    if temps is not None:
        g = g * torch.as_tensor([float(t) for t in temps], dtype=torch.float64).reshape(-1, 1)
    return g.to(torch.float32)


def expo(seeds, positions, C, stream):
    """Exp(1) noise [M, C] fp32 (the residual's exponential race)."""
    import torch

    return (-torch.log(_u01(seeds, positions, C, stream))).to(torch.float32)


def uniform(seeds, positions, stream):
    """U(0,1) [M, 1] fp32 (the accept test)."""
    import torch

    return _u01(seeds, positions, 1, stream).to(torch.float32)


def param_rows(temps, topks, topps, C2):
    """Per-row cut parameters as device-shaped host blocks: scale [M,C2] (1/T; 1.0 for T == 0),
    rank [M,C2] (the top_k prefix mask), topp [M,1]."""
    import torch

    M = len(temps)
    scale = torch.tensor([1.0 / (float(t) if float(t) > 0.0 else 1.0) for t in temps], dtype=torch.float32)
    scale = scale.reshape(M, 1).expand(M, C2).contiguous()
    cols = torch.arange(C2, dtype=torch.float32).reshape(1, C2)
    kk = torch.tensor([float(k) if int(k) > 0 else float(C2) for k in topks], dtype=torch.float32).reshape(M, 1)
    rank = (cols < kk).to(torch.float32)
    topp = torch.tensor([max(float(p), 1e-30) for p in topps], dtype=torch.float32).reshape(M, 1)
    return {"scale": scale, "rank": rank, "topp": topp}


# ---------------------------------------------------------------------------------------------- host noise (fast)
# QWEN36_MTP_SPEC_SAMPLE_HOSTNOISE=1: the per-step noise blocks are built with ONE hash per GROUP of
# streams instead of one hash per stream, and the int64 part of the hash runs in numpy (the float part
# stays in torch).  The VALUES ARE BIT-IDENTICAL: the hash input per (row, stream, column) is unchanged
# (base(seed, pos, stream) + col * M3, exact int64 arithmetic), and the float conversion and the logs are
# the same torch ops on the same numbers.
#
# WHY (the measured cost, bench/probes/host_spec_noise_cost.py): the served path builds the noise on the
# HOST between the verify and the chain, where the DEVICE IS IDLE, so the cost is straight on the
# critical path of the step.  The tensors are tiny (a few KB), so the cost is the per-call DISPATCH
# overhead, not the data: the served path makes ~20 torch dispatches per stream and there are K + 2
# streams per step (the accept uniform, the residual's Exp(1) noise and one Gumbel block per chain
# depth).  Measured on the dev host at the served shapes: ~740 us per step at B4 (M=16, K=3, C2=128),
# ~800 us at B1, and more at B8 (the min of 300 reps; the medians are inflated by a host artifact).  The
# group path makes ~13 numpy dispatches per group (numpy has a much lower per-call overhead than torch on
# tiny arrays) and 2 groups per step: ~370-500 us.
#
# The numpy int64 arithmetic wraps exactly like the torch int64 arithmetic (C semantics), and the
# in-boot self-test compares the two builders BITWISE at every served row count, on both chips, with a
# wrong-input branch (a perturbed hash) that must differ.
_COL_OFFS_CACHE = {}


def fast_noise_enabled():
    """The fast host-noise path (default off)."""
    return _FAST


def _np_mix64(x):
    """splitmix64's finalizer on a numpy int64 array (the same wrap-around as the torch version)."""
    x = x ^ ((x >> 30) & 0x3FFFFFFFF)
    x = x * _M1
    x = x ^ ((x >> 27) & 0x1FFFFFFFFFF)
    x = x * _M2
    x = x ^ ((x >> 31) & 0x1FFFFFFFFFF)
    return x


def _col_offs(widths):
    """The per-column hash offsets of a group: concat(arange(w) * M3) per stream (cached per shape)."""
    import numpy as np

    key = tuple(int(w) for w in widths)
    off = _COL_OFFS_CACHE.get(key)
    if off is None:
        off = np.concatenate([np.arange(int(w), dtype=np.int64) * _M3 for w in widths])
        _COL_OFFS_CACHE[key] = off
    return off


def noise_group(items, mustfail=False):
    """The group's noise blocks, built with ONE hash.  items: a list of dicts with
    "seeds" (per row), "positions" (per row), "stream" (int), "width" (int), "kind"
    ("u01" | "expo" | "gumbel") and, for "gumbel", "temps" (per row).  All items have the same row count.

    Returns a list of torch float32 tensors, one per item, of shape [rows, width].  The values are
    BIT-IDENTICAL to the per-stream path (uniform / expo / gumbel above): the hash input per
    (row, stream, column) is base(seed, pos, stream) + col * M3 in both."""
    import numpy as np
    import torch

    if not items:
        return []
    rows = len(items[0]["seeds"])
    for it in items:
        if len(it["seeds"]) != rows or len(it["positions"]) != rows:
            raise RuntimeError(f"{FAST_ENV}: every item of a group must have the same row count")
    widths = [int(it["width"]) for it in items]
    bases = []
    for it in items:
        s = np.asarray([int(x) for x in it["seeds"]], dtype=np.int64)
        p = np.asarray([int(x) for x in it["positions"]], dtype=np.int64)
        b = (s * _M3) ^ (p * _GOLD) ^ np.int64((int(it["stream"]) * _M1) & 0x7FFFFFFFFFFFFFFF)
        bases.append(b)
    comb = np.concatenate([np.repeat(b[:, None], w, axis=1) for b, w in zip(bases, widths)], axis=1)
    if mustfail:
        offs = _col_offs(widths) ^ 1
    else:
        offs = _col_offs(widths)
    h = _np_mix64(comb + offs)
    ht = torch.from_numpy(h)
    u = ((ht >> 11) & 0x1FFFFFFFFFFFFF).to(torch.float64) * (2.0 ** -53)
    u = u.clamp_(1e-12, 1.0 - 1e-9)
    out = []
    off = 0
    for it, w in zip(items, widths):
        seg = u[:, off:off + w]
        off += w
        kind = it["kind"]
        if kind == "u01":
            v = seg
        elif kind == "expo":
            v = -torch.log(seg)
        elif kind == "gumbel":
            g = -torch.log(-torch.log(seg))
            temps = it.get("temps")
            if temps is not None:
                g = g * torch.as_tensor([float(t) for t in temps], dtype=torch.float64).reshape(-1, 1)
            v = g
        else:
            raise RuntimeError(f"{FAST_ENV}: unknown noise kind {kind!r}")
        out.append(v.to(torch.float32).contiguous())
    return out


def param_rows_fast(temps, topks, topps, C2):
    """param_rows with numpy (the same values): scale [M,C2], rank [M,C2], topp [M,1] float32."""
    import numpy as np
    import torch

    M = len(temps)
    inv_t = np.asarray([1.0 / (float(t) if float(t) > 0.0 else 1.0) for t in temps], dtype=np.float32)
    scale = np.repeat(inv_t.reshape(M, 1), int(C2), axis=1)
    cols = np.arange(int(C2), dtype=np.float32).reshape(1, int(C2))
    kk = np.asarray([float(k) if int(k) > 0 else float(C2) for k in topks], dtype=np.float32).reshape(M, 1)
    rank = (cols < kk).astype(np.float32)
    topp = np.asarray([max(float(p), 1e-30) for p in topps], dtype=np.float32).reshape(M, 1)
    return {"scale": torch.from_numpy(scale), "rank": torch.from_numpy(rank), "topp": torch.from_numpy(topp)}


# ---------------------------------------------------------------------------------------------- device
def candidate_block(logits, per, mesh, tt_ccl, topology, K):
    """The merged candidate block over the TP-sharded vocab, in the global descending order.

    logits: [1,1,M,per] bf16 per chip (chip d holds ids [d*per, (d+1)*per)), TILE or ROW_MAJOR; NOT freed.
    Returns (sv [1,1,M,2K] fp32 sorted values, si [1,1,M,2K] uint32 the merged column of each sorted
    entry, gids [1,1,M,2K] fp32 global ids), ROW_MAJOR."""
    from models.demos.blackhole.qwen36.tt import dev_sample
    from models.tt_transformers.tt.ccl import tt_all_gather

    M, per, K = int(logits.shape[2]), int(per), int(K)
    NC = 2 * K
    _check_rows(M)
    s = _shape(logits)
    if len(s) != 4 or s[:3] != [1, 1, M] or s[3] != per or logits.dtype != ttnn.bfloat16:
        raise RuntimeError(f"{ENV}: logits {s} {logits.dtype}; expected [1, 1, {M}, {per}] bf16")
    # QWEN36_MTP_SPEC_SAMPLE_RMSHARE: the caller may hand over logits that are ALREADY row-major (the verify
    # untilizes once and feeds the same tensor to the greedy argmax); then nothing is converted and nothing of
    # the caller's is freed.  The layout is tested, not the handle: ttnn.to_layout on a row-major tensor may
    # return a new handle to the SAME buffer, and freeing it would free the caller's logits.
    own = logits.layout != ttnn.ROW_MAJOR_LAYOUT
    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT) if own else logits
    vals, idx = dev_sample._topk_wide(rm, K)
    v32 = ttnn.typecast(vals, ttnn.float32)
    i32 = ttnn.typecast(idx, ttnn.float32)
    _free(vals, idx)
    if own:
        _free(rm)
    gv = tt_all_gather(v32, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    gi = tt_all_gather(i32, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    _free(v32, i32)
    if _shape(gv) != [1, 1, M, NC]:
        raise RuntimeError(f"{ENV}: gathered values {_shape(gv)}; expected [1, 1, {M}, {NC}]")
    gv_t = _tile(gv)
    sv_t, si_t = ttnn.topk(gv_t, NC, 3)
    sv, si, gi_rm = _rm(sv_t), _rm(si_t), _rm(gi)
    lid = ttnn.gather(gi_rm, dim=-1, index=si)
    off = ttnn.multiply(ttnn.ge(ttnn.typecast(si, ttnn.float32), float(K)), float(per))  # merged col >= K: chip 1
    gids = ttnn.add(lid, off)
    for t in {id(x): x for x in (gv, gv_t, sv_t, si_t, gi, gi_rm, lid, off)}.values():
        if t is not sv and t is not si:
            _free(t)
    return sv, si, gids


def cut(sv, scale, rank, topp, M, mesh):
    """The vLLM order on a sorted block: temperature -> top_k -> softmax -> top_p, renormalised.

    sv: [1,1,M,C2] fp32 RM descending; scale/rank [1,1,M,C2] fp32; topp [1,1,M,1] fp32.
    Returns (probs, mask), [1,1,M,C2] fp32 RM; probs is 0 off the support. Column 0 is the row max
    (descending values, positive 1/T, rank[0] == 1), so the stable softmax subtracts it.

    top_k keeps TIES at the k-th value, as vLLM does (apply_top_k_top_p keeps the logits >= the k-th
    largest): rank marks the first k sorted columns, the k-th value is the min over them, and the kept
    set is every column >= it. bf16 logits tie often at the boundary (the step gate saw 6-way ties at
    rank 20 on random heads), so an exactly-k cut is a different distribution, not a rounding."""
    vt = _tile(ttnn.multiply(sv, scale))
    rank_t, topp_t = _tile(rank), _tile(topp)
    inv = ttnn.subtract(rank_t, 1.0)
    big = ttnn.add(ttnn.multiply(vt, rank_t), ttnn.multiply(inv, CUT_NEG))  # outside the prefix: +1e4
    kth = ttnn.min(big, dim=-1, keepdim=True)
    rk = ttnn.typecast(ttnn.ge(vt, kth), ttnn.float32)  # ties at the k-th value kept
    vk = ttnn.add(ttnn.multiply(vt, rk), ttnn.multiply(ttnn.subtract(rk, 1.0), -CUT_NEG))
    m0 = ttnn.max(vk, dim=-1, keepdim=True)
    e = ttnn.exp(ttnn.subtract(vk, m0))
    pr = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True))
    cum = ttnn.cumsum(pr, dim=-1)
    before = ttnn.subtract(cum, pr)
    keep = ttnn.typecast(ttnn.lt(before, topp_t), ttnn.float32)
    mask_t = ttnn.multiply(keep, rk)
    pm = ttnn.multiply(pr, mask_t)
    probs_t = ttnn.divide(pm, ttnn.sum(pm, dim=-1, keepdim=True))
    probs, mask = _rm(probs_t), _rm(mask_t)
    keepers = {id(probs), id(mask), id(sv), id(scale), id(rank), id(topp)}
    for t in (vt, rank_t, topp_t, inv, big, kth, rk, vk, m0, e, pr, cum, before, keep, mask_t, pm, probs_t):
        if id(t) not in keepers:
            keepers.add(id(t))
            _free(t)
    return probs, mask


def pick(sv, si, gids, mask, noise):
    """The Gumbel-max sample over the support: argmax(sv + T*g) with the host's pre-scaled noise in the
    MERGED column order (gathered to the sorted order by si). Returns [1,1,M,1] uint32 global ids."""
    nsort = ttnn.gather(noise, dim=-1, index=si)
    noisy = ttnn.add(sv, nsort)
    masked = ttnn.add(ttnn.multiply(noisy, mask), ttnn.multiply(ttnn.subtract(mask, 1.0), -NEG))
    r = ttnn.typecast(ttnn.argmax(masked, dim=-1, keepdim=True), ttnn.uint32)
    out = ttnn.typecast(ttnn.gather(gids, dim=-1, index=r), ttnn.uint32)
    _free(nsort, noisy, masked, r)
    return out


def candidates(logits, per, mesh, tt_ccl, topology, blk, K):
    """candidate_block + cut + pick. blk (role-keyed, dev_sample's verify roles): noise [1,1,M,2K]
    (pre-scaled T*g, merged order), scale [1,1,M,2K], rank [1,1,M,2K], topp [1,1,M,1].
    The logits are NOT freed. Returns {"probs", "ids", "pick"} (the caller frees them)."""
    M = int(logits.shape[2])
    NC = 2 * int(K)
    for name, want in (("noise", NC), ("rank", NC), ("scale", NC), ("topp", 1)):
        t = blk.get(name)
        if t is None:
            raise RuntimeError(f"{ENV}: the block has no '{name}' tensor (keys {sorted(blk)})")
        if _shape(t) != [1, 1, M, want]:
            raise RuntimeError(f"{ENV}: block '{name}' {_shape(t)}; expected [1, 1, {M}, {want}]")
    sv, si, gids = candidate_block(logits, per, mesh, tt_ccl, topology, K)
    probs, mask = cut(sv, blk["scale"], blk["rank"], blk["topp"], M, mesh)
    pk = pick(sv, si, gids, mask, blk["noise"])
    _free(sv, si, mask)
    return {"probs": probs, "ids": gids, "pick": pk}


def _tsum(x, dim):
    """A TILE-layout sum (a ROW_MAJOR ttnn.sum is off by ~4e-4 relative in ttsim)."""
    xt = _tile(x)
    s = ttnn.sum(xt, dim=dim, keepdim=True)
    out = _rm(s)
    if xt is not x:
        _free(xt)
    if s is not out:
        _free(s)
    return out


def _bcast_col(x, M, C2):
    """[1,1,M,1] -> [1,1,M,C2] (the comparison ops do not broadcast a column in ROW_MAJOR)."""
    return ttnn.reshape(ttnn.expand(ttnn.reshape(x, (1, M, 1, 1)), (1, M, 1, C2)), (1, 1, M, C2))


MF_MODES = ("accept", "resid_p")


def rule(p, pi, q, qi, x, u, nres, mf=None, ret_debug=False):
    """The exact rule for every row, on the device.

    p/pi:  [1,1,M,C2] fp32 the target's post-cut probs (0 off its support) and global ids
    q/qi:  [1,1,M,C2] fp32 the drafter's carried post-cut probs and FULL ids (a point mass for a
           deterministic draft)
    x:     [1,1,M,1] fp32 the draft id; u: [1,1,M,1] fp32 U[0,1); nres: [1,1,M,C2] fp32 Exp(1)
    mf:    None (env MUSTFAIL -> "accept"), False, "accept" (flip every accept: the boot self-test's
           wrong branch) or "resid_p" (the residual draws from p instead of (p-q)^+: a distribution
           bug the chi-square gate must catch).
    Returns {"accept" [1,1,M,1] fp32 0/1, "resid" [1,1,M,1] uint32}. Inputs are not freed."""
    M, C2 = int(p.shape[2]), int(p.shape[3])
    _check_rows(M)
    if mf is None:
        mf = "accept" if mustfail() else False
    if mf and mf not in MF_MODES:
        raise RuntimeError(f"{ENV}: unknown must-fail mode {mf!r}")
    # q at each target candidate: eq[i, j] = (pi_i == qi_j); ids are unique per list, so one term survives
    ae = ttnn.expand(ttnn.reshape(pi, (1, M, C2, 1)), (1, M, C2, C2))
    be = ttnn.expand(ttnn.reshape(qi, (1, M, 1, C2)), (1, M, C2, C2))
    eq = ttnn.typecast(ttnn.eq(ae, be), ttnn.float32)
    q4 = ttnn.expand(ttnn.reshape(q, (1, M, 1, C2)), (1, M, C2, C2))
    eqq = ttnn.multiply(eq, q4)
    q_at_p = ttnn.reshape(_tsum(eqq, 3), (1, 1, M, C2))
    xs = _bcast_col(x, M, C2)
    isx_p = ttnn.typecast(ttnn.eq(pi, xs), ttnn.float32)
    isx_q = ttnn.typecast(ttnn.eq(qi, xs), ttnn.float32)
    px_m, qx_m = ttnn.multiply(isx_p, p), ttnn.multiply(isx_q, q)
    p_at_x = _tsum(px_m, 3)
    q_at_x = _tsum(qx_m, 3)
    uq = ttnn.multiply(u, q_at_x)
    accept = ttnn.typecast(ttnn.lt(uq, p_at_x), ttnn.float32)
    if mf == "accept":
        flipped = ttnn.multiply(ttnn.subtract(accept, 1.0), -1.0)
        _free(accept)
        accept = flipped
    # the residual (p - q)^+ over the target's support, drawn by the exponential race argmin E_i / r_i
    rn = ttnn.subtract(p, q_at_p) if mf != "resid_p" else ttnn.clone(p)
    pos = ttnn.typecast(ttnn.gt(rn, 0.0), ttnn.float32)
    r = ttnn.multiply(rn, pos)
    ratio = ttnn.divide(nres, ttnn.add(r, 1e-30))
    score = ttnn.add(ttnn.multiply(ratio, ttnn.multiply(pos, -1.0)), ttnn.multiply(ttnn.subtract(pos, 1.0), -NEG))
    j = ttnn.typecast(ttnn.argmax(score, dim=-1, keepdim=True), ttnn.uint32)
    resid = ttnn.typecast(ttnn.gather(pi, dim=-1, index=j), ttnn.uint32)
    out = {"accept": accept, "resid": resid}
    if ret_debug:
        out.update(p_at_x=p_at_x, q_at_x=q_at_x, q_at_p=q_at_p)
    else:
        _free(p_at_x, q_at_x, q_at_p)
    _free(eq, eqq, isx_p, isx_q, px_m, qx_m, uq, rn, pos, r, ratio, score, j)
    return out


# ---------------------------------------------------------------------------------------------- host reference
def host_vllm_probs(logits, temp, top_k, top_p):
    """The target distribution by vLLM's own semantics on a FULL row (fp64): logits / T, top_k (keep
    the logits >= the k-th largest), softmax, top_p (sort ascending, drop the tokens whose cumulative
    mass is <= 1 - p, never the last), renormalised. Returns a dense [V] fp64 tensor."""
    import torch

    v = logits.double() / float(temp)
    if top_k and int(top_k) > 0 and int(top_k) < v.numel():
        kth = torch.topk(v, int(top_k)).values[-1]
        v = torch.where(v < kth, torch.full_like(v, -math.inf), v)
    pr = torch.softmax(v, dim=-1)
    if top_p is not None and float(top_p) < 1.0:
        sp, si = torch.sort(pr, descending=False)
        cs = torch.cumsum(sp, dim=-1)
        drop = cs <= (1.0 - float(top_p))
        drop[-1] = False
        keep = torch.ones_like(pr, dtype=torch.bool)
        keep[si[drop]] = False
        pr = torch.where(keep, pr, torch.zeros_like(pr))
    return pr / pr.sum()


def host_rule(p, pi, q, qi, x, u, nres, mf=False):
    """The device rule's arithmetic in fp32 on the host ([M, C2] / [M] torch tensors). Returns
    (accept [M] float, resid [M] int64)."""
    import torch

    p, q, nres = p.float(), q.float(), nres.float()
    pi, qi, x, u = pi.float(), qi.float(), x.float().reshape(-1), u.float().reshape(-1)
    eq = (pi.unsqueeze(2) == qi.unsqueeze(1)).float()
    q_at_p = (eq * q.unsqueeze(1)).sum(dim=2)
    p_at_x = ((pi == x.unsqueeze(1)).float() * p).sum(dim=1)
    q_at_x = ((qi == x.unsqueeze(1)).float() * q).sum(dim=1)
    accept = ((u * q_at_x) < p_at_x).float()
    if mf == "accept":
        accept = 1.0 - accept
    rn = p - q_at_p if mf != "resid_p" else p.clone()
    pos = (rn > 0).float()
    r = rn * pos
    score = torch.where(pos > 0, -(nres / (r + 1e-30)), torch.full_like(r, NEG))
    j = torch.argmax(score, dim=1)
    resid = pi[torch.arange(pi.shape[0]), j].long()
    return accept, resid


def host_cut(sv, scale, rank, topp):
    """cut() in fp64 on the host ([M, C2] blocks, topp [M, 1]). Returns (probs, mask)."""
    import torch

    vt = sv.double() * scale.double()
    kth = torch.where(rank > 0, vt, torch.full_like(vt, math.inf)).min(dim=-1, keepdim=True).values
    rk = vt >= kth  # ties at the k-th value kept (vLLM)
    vk = torch.where(rk, vt, torch.full_like(vt, -math.inf))
    pr = torch.softmax(vk, dim=-1)
    before = torch.cumsum(pr, dim=-1) - pr
    mask = ((before < topp.double()) & rk).double()
    pm = pr * mask
    return pm / pm.sum(dim=-1, keepdim=True), mask


# ---------------------------------------------------------------------------------------------- chi-square
def _gser(k, x):
    if x <= 0.0:
        return 0.0
    ap, total = k, 1.0 / k
    dl = total
    for _ in range(1000):
        ap += 1.0
        dl *= x / ap
        total += dl
        if abs(dl) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + k * math.log(x) - math.lgamma(k))


def _gcf(k, x):
    tiny = 1e-300
    b = x + 1.0 - k
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - k)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return math.exp(-x + k * math.log(x) - math.lgamma(k)) * h


def gamma_upper(k, x):
    k, x = float(k), float(x)
    if x <= 0.0:
        return 1.0
    if x < k + 1.0:
        return max(0.0, min(1.0, 1.0 - _gser(k, x)))
    return max(0.0, min(1.0, _gcf(k, x)))


def chi_square(counts, probs, fail_count=0):
    """Pearson chi-square of observed counts vs expected probs, cells with expected < 5 pooled.
    fail_count: samples outside the reference support (expected 0): any makes the p-value 0.
    Returns (chi2, dof, pvalue)."""
    n = float(sum(counts) + fail_count)
    exp = [n * float(p) for p in probs]
    keep = [i for i, e in enumerate(exp) if e >= 5.0]
    rest = [i for i in range(len(counts)) if i not in set(keep)]
    cells = [(counts[i], exp[i]) for i in keep]
    po, pe = sum(counts[i] for i in rest), sum(exp[i] for i in rest)
    if pe > 0 or po > 0:
        cells.append((po, pe))
    chi2 = sum((o - e) ** 2 / e for o, e in cells if e > 0)
    if any(e == 0 and o > 0 for o, e in cells) or fail_count:
        return float("inf"), max(len(cells) - 1, 1), 0.0
    dof = max(len(cells) - 1, 1)
    return chi2, dof, gamma_upper(dof / 2.0, chi2 / 2.0)


# ---------------------------------------------------------------------------------------------- boot self-test
def _synthetic(M, C2, seed=4242, temp=1.0, topk=20, topp=0.95):
    """Host blocks for the self-test: the target's and the drafter's post-cut probs over partially
    overlapping id lists, x ~ q, the accept uniform and the residual noise."""
    import torch

    g = torch.Generator().manual_seed(seed)
    sv = torch.sort(torch.randn(M, C2, generator=g, dtype=torch.float64) * 2.0, dim=-1, descending=True).values
    svq = torch.sort(torch.randn(M, C2, generator=g, dtype=torch.float64) * 2.0, dim=-1, descending=True).values
    prm = param_rows([temp] * M, [topk] * M, [topp] * M, C2)
    p, _ = host_cut(sv, prm["scale"], prm["rank"], prm["topp"])
    q, _ = host_cut(svq, prm["scale"], prm["rank"], prm["topp"])
    pi = torch.stack([torch.randperm(4 * C2, generator=g)[:C2] for _ in range(M)]).float()
    qi = torch.cat([pi[:, : C2 // 4], torch.stack([torch.randperm(4 * C2, generator=g)[:C2 - C2 // 4] + 8 * C2
                                                    for _ in range(M)]).float()], dim=1)
    qi = qi[:, torch.randperm(C2, generator=g)]  # the shared ids at shuffled ranks
    xi = torch.multinomial(q.float(), 1, generator=g).reshape(M)
    x = qi[torch.arange(M), xi].reshape(M, 1)
    u = torch.rand(M, 1, generator=g)
    nres = expo(list(range(M)), list(range(M)), C2, STREAM_RES)
    return p.float(), pi, q.float(), qi, x, u, nres


def selftest(mesh, cells, K):
    """The device rule vs host_rule, bitwise on accept and resid, on every chip, per row count; the
    must-fail branch (accept flipped) must differ. Returns {tag: "exact" | "DIFF n", tag_mustfail: n}."""
    import torch

    C2 = 2 * int(K)
    ndev = _ndev(mesh)
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if ndev > 1 else {}

    def dev(t):
        return ttnn.from_torch(t.contiguous().reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep)

    def back(t):
        chips = ttnn.get_device_tensors(t) if ndev > 1 else [t]
        return [ttnn.to_torch(c).float().reshape(-1) for c in chips]

    res = {}
    for M in sorted({int(m) for m in cells}):
        p, pi, q, qi, x, u, nres = _synthetic(M, C2)
        d = [dev(t) for t in (p, pi, q, qi, x, u, nres)]
        ref_a, ref_r = host_rule(p, pi, q, qi, x, u, nres)
        out = rule(*d, mf=False)
        n = sum(int((a != ref_a).sum()) for a in back(out["accept"])) + \
            sum(int((r.long() != ref_r).sum()) for r in back(out["resid"]))
        res[f"M{M}"] = "exact" if n == 0 else f"DIFF {n}"
        omf = rule(*d, mf="accept")
        res[f"M{M}_mustfail"] = sum(int((a != ref_a).sum()) for a in back(omf["accept"]))
        res[f"M{M}_accept_rate"] = round(float(ref_a.mean()), 3)
        _free(*d, out["accept"], out["resid"], omf["accept"], omf["resid"])
    return res


def selftest_ok(res):
    return (all(v == "exact" for k_, v in res.items() if k_.startswith("M") and "_" not in k_)
            and all(int(v) > 0 for k_, v in res.items() if k_.endswith("_mustfail")))


def noise_selftest(mesh, K):
    """The fast host-noise builder vs the served (per-stream torch) builder, BITWISE, at the served row
    counts, and the block on EVERY chip.  The wrong-input branch (a perturbed hash) must differ.

    Returns {f"M{M}": "exact" | "DIFF n", f"M{M}_mustfail": n, f"M{M}_chips": n, f"M{M}_chip_diffs": n}."""
    import torch

    C2 = 2 * int(K)
    ndev = _ndev(mesh)
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if ndev > 1 else {}
    res = {}
    for M in (4, 8, 16, 32):
        bw = max(1, min(M, 8))
        seeds_v = [1000 + i for i in range(M)]
        pos_v = list(range(M))
        seeds_c = [2000 + i for i in range(bw)]
        pos_c = [[k for _ in range(bw)] for k in range(3)]
        items_v = [
            {"seeds": seeds_v, "positions": pos_v, "stream": STREAM_U, "width": 1, "kind": "u01"},
            {"seeds": seeds_v, "positions": pos_v, "stream": STREAM_RES, "width": C2, "kind": "expo"},
        ]
        items_c = [
            {"seeds": seeds_c, "positions": pos_c[k], "stream": STREAM_CHAIN + k, "width": C2, "kind": "gumbel",
             "temps": [0.6] * bw} for k in range(3)
        ]
        fast_v = noise_group(items_v, mustfail=_FAST_MF)
        fast_c = noise_group(items_c, mustfail=_FAST_MF)
        ref_v = [uniform(seeds_v, pos_v, STREAM_U), expo(seeds_v, pos_v, C2, STREAM_RES)]
        ref_c = [gumbel(seeds_c, pos_c[k], C2, STREAM_CHAIN + k, temps=[0.6] * bw) for k in range(3)]
        n = sum(int((a != b).sum()) for a, b in zip(fast_v, ref_v))
        n += sum(int((a != b).sum()) for a, b in zip(fast_c, ref_c))
        res[f"M{M}"] = "exact" if n == 0 else f"DIFF {n}"
        # the wrong-input branch: the perturbed hash must differ from the served builder
        mf = noise_group(items_v, mustfail=True)
        res[f"M{M}_mustfail"] = sum(int((a != b).sum()) for a, b in zip(mf, ref_v))
        # the block on EVERY chip: the fast builder's block, uploaded and read back
        blk = fast_v[1].reshape(1, 1, M, C2).contiguous()
        d = ttnn.from_torch(blk, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, **rep)
        chips = ttnn.get_device_tensors(d) if ndev > 1 else [d]
        want = fast_v[1].reshape(-1)
        bad = 0
        for c in chips:
            got = ttnn.to_torch(c).float().reshape(-1)
            bad += int((got != want).sum())
        res[f"M{M}_chips"] = ndev
        res[f"M{M}_chip_diffs"] = bad
        ttnn.deallocate(d)
    return res


def noise_selftest_ok(res):
    """Every row count must be bitwise equal, the wrong-input branch must differ, and every chip must
    carry the same block."""
    exact = [v for k_, v in res.items() if k_.startswith("M") and "_" not in k_]
    mf = [int(v) for k_, v in res.items() if k_.endswith("_mustfail")]
    chips = [int(v) for k_, v in res.items() if k_.endswith("_chip_diffs")]
    return bool(exact) and all(v == "exact" for v in exact) and all(v > 0 for v in mf) and all(v == 0 for v in chips)


def lean_selftest(mesh, tt_ccl, topology, verify_cells, draft_cells, K, mf=None):
    """RMSHARE: at every served verify (M, per), the greedy ids from the row-major logits (argmax_fold.greedy_rm)
    and the candidate block from them must equal the served TILE path's BITWISE on every chip.  DRAFTPICK: at
    every served chain (bw, per), the zero-noise pick must land on a row maximum of the logits for every row.
    The wrong-input branch (mf: greedy_rm's +1 must-fail, a pick with full Gumbel noise) must differ.
    Returns {tag: "exact" | "DIFF n", tag + "_mustfail": n}."""
    import torch

    from models.demos.blackhole.qwen36.tt import argmax_fold

    mf = lean_mustfail() if mf is None else bool(mf)
    ndev = _ndev(mesh)
    shard = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=3)} if ndev > 1 else {}
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if ndev > 1 else {}
    g = torch.Generator().manual_seed(20261021)
    res = {}

    def up(t, layout=ttnn.ROW_MAJOR_LAYOUT, dt=ttnn.float32, m=rep):
        return ttnn.from_torch(t.contiguous(), dtype=dt, layout=layout, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, **m)

    def chips(t):
        return [ttnn.to_torch(c).float().reshape(-1) for c in (ttnn.get_device_tensors(t) if ndev > 1 else [t])]

    def logits_host(M, per):
        x = torch.randn(1, 1, M, ndev * per, generator=g) * 2.0
        # an exact tie at the row max in the even rows (the tie rule must hold through the share). Never at M = 1:
        # the DRAFTPICK must-fail counts picks OFF the row max, and with the single row tied the noisy pick lands on
        # one of the two tied maxima, so the must-fail could never fail and every DRAFTPICK boot was refused (card
        # runs 2026-10-21: pick_M1_P32768_mustfail = 0).
        for m in range(M):
            if m % 2 == 0 and M > 1:
                a, b = torch.randperm(ndev * per, generator=g)[:2].tolist()
                x[0, 0, m, b] = x[0, 0, m, a] = 12.0
        return x.to(torch.bfloat16)

    if _RMSHARE:
        for M, per in sorted({(int(a), int(b)) for a, b in verify_cells}):
            x = logits_host(M, per)
            tile = up(x, ttnn.TILE_LAYOUT, ttnn.bfloat16, shard)
            ref = argmax_fold.greedy(tile, per, mesh, tt_ccl, topology, mf=False)  # frees tile
            tile = up(x, ttnn.TILE_LAYOUT, ttnn.bfloat16, shard)
            rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT)
            got = argmax_fold.greedy_rm(rm, per, mesh, tt_ccl, topology, mf=mf)
            bad = argmax_fold.greedy_rm(rm, per, mesh, tt_ccl, topology, mf=True)
            ct = candidate_block(tile, per, mesh, tt_ccl, topology, K)
            cr = candidate_block(rm, per, mesh, tt_ccl, topology, K)
            n = sum(int((a[:M] != b[:M]).sum()) for a, b in zip(chips(ref), chips(got)))
            n += sum(int((a != b).sum()) for ta, tb in zip(ct, cr) for a, b in zip(chips(ta), chips(tb)))
            nm = sum(int((a[:M] != b[:M]).sum()) for a, b in zip(chips(ref), chips(bad)))
            res[f"rm_M{M}_P{per}"] = "exact" if n == 0 else f"DIFF {n}"
            res[f"rm_M{M}_P{per}_mustfail"] = nm
            _free(ref, got, bad, rm, tile, *ct, *cr)
    if _DRAFTPICK:
        for M, per in sorted({(int(a), int(b)) for a, b in draft_cells}):
            x = logits_host(M, per)
            prm = param_rows([1.0] * M, [20] * M, [0.95] * M, 2 * int(K))
            zero = torch.zeros(1, 1, M, 2 * int(K))
            noise = gumbel(list(range(M)), list(range(M)), 2 * int(K), STREAM_CHAIN, temps=[4.0] * M)
            blk = {"scale": up(prm["scale"].reshape(1, 1, M, -1)), "rank": up(prm["rank"].reshape(1, 1, M, -1)),
                   "topp": up(prm["topp"].reshape(1, 1, M, 1))}
            hmax = x[0, 0].float().max(dim=-1).values
            # the wrong branch runs over several noise draws when there are few untied rows (M < 4): one draw on one
            # row lands on the row max by chance (~1 in the support size), which would refuse a correct boot
            rounds = [noise] + [gumbel(list(range(M)), [r * 4096 + m for m in range(M)], 2 * int(K), STREAM_CHAIN,
                                       temps=[4.0] * M) for r in range(1, max(1, 4 // M))]
            runs = [("pick", noise if mf else zero)] + [("mf", nz) for nz in rounds]
            out = {"pick": 0, "mf": 0}
            for tag, nz in runs:
                tile = up(x, ttnn.TILE_LAYOUT, ttnn.bfloat16, shard)
                nzt = up(nz.reshape(1, 1, M, -1))
                c = candidates(tile, per, mesh, tt_ccl, topology, dict(blk, noise=nzt), K)
                ids = [v[:M].long() for v in chips(c["pick"])]
                out[tag] += sum(int((x[0, 0].float()[torch.arange(M), i] != hmax).sum()) for i in ids)
                _free(tile, nzt, c["pick"], c["probs"], c["ids"])
            res[f"pick_M{M}_P{per}"] = "exact" if out["pick"] == 0 else f"DIFF {out['pick']}"
            res[f"pick_M{M}_P{per}_mustfail"] = out["mf"]
            _free(*blk.values())
    return res


def lean_selftest_ok(res):
    checks = {k_: v for k_, v in res.items() if not k_.endswith("_mustfail")}
    return (all(v == "exact" for v in checks.values())
            and all(int(v) > 0 for k_, v in res.items() if k_.endswith("_mustfail")))


def ensure_lean_selftest(mesh, tt_ccl, topology, verify_cells, draft_cells, K):
    """Once per process, before any capture: fail closed.  LEAN_MUSTFAIL=1 serves the wrong branches, so the
    check must refuse the boot."""
    if not (rm_share() or draft_pick()) or "lean_selftest" in _logged:
        return
    res = lean_selftest(mesh, tt_ccl, topology, verify_cells, draft_cells, K)
    if not lean_selftest_ok(res):
        raise RuntimeError(f"QWEN36_MTP_SPEC_SAMPLE lean self-test REFUSED: {res}")
    _logged.add("lean_selftest")
    logger.info(f"QWEN36_MTP_SPEC_SAMPLE lean self-test exact: RMSHARE ids + candidate block bitwise vs the TILE path, "
                f"DRAFTPICK zero-noise pick at the row max, on {_ndev(mesh)} chip(s); the wrong branches differ ({res})")


def ensure_selftest(mesh, cells, K):
    """Once per process, before any capture: fail closed (with MUSTFAIL=1 the rule's accepts are
    flipped, so the check must refuse the boot)."""
    if not enabled() or "selftest" in _logged:
        return
    res = selftest(mesh, cells, K)
    if mustfail():
        raise RuntimeError(f"{MUSTFAIL_ENV}=1: refusing the boot after the self-test ({res}): the served rule "
                           f"flips every accept under this flag")
    if not selftest_ok(res):
        raise RuntimeError(f"{ENV} self-test REFUSED: {res}")
    if _FAST:
        # QWEN36_MTP_SPEC_SAMPLE_HOSTNOISE: the fast host-noise builder must be BIT-IDENTICAL to the
        # served one at every served row count, on every chip, and the perturbed-hash branch must
        # differ (with FAST_MUSTFAIL=1 the builder IS perturbed, so this check must refuse).
        nres = noise_selftest(mesh, K)
        if not noise_selftest_ok(nres):
            raise RuntimeError(f"{FAST_ENV} self-test REFUSED: {nres}")
        logger.info(f"QWEN36_MTP_SPEC_SAMPLE_HOSTNOISE self-test exact: the grouped host-noise builder is bitwise equal to the "
                    f"served per-stream builder at rows 4/8/16/32 on {_ndev(mesh)} chip(s), C2={2 * int(K)}; "
                    f"the perturbed-hash branch differs ({nres})")
    _logged.add("selftest")
    logger.info(f"{SELFTEST_BANNER}: the device rule (accept + residual) bitwise vs the host's fp32 arithmetic at "
                f"rows {sorted({int(m) for m in cells})} on {_ndev(mesh)} chip(s), C2={2 * int(K)}; the flipped-accept "
                f"branch differs ({res})")
