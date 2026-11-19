# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_BATCH host bookkeeping: pure Python, no ttnn, no torch.

Batched MTP speculative decode (docs/MTP-BATCH.md) splits into a device half (tt/mtp_batch.py: the
verify and commit+draft traces) and this host half, which owns every per-row decision:

  plan()    per-row draft validity, the j-major verify rows (u = j*Bw + b), positions, caps
  accept()  greedy exact acceptance per row, the emitted tokens, and the commit/draft inputs
  finish()  store the new drafts per slot

Correctness never depends on this state: verify re-derives every emitted token from the base model
(emitted = verify argmax ids[b, 0..m_b]), so a wrong draft, a stale slot or a lost request only
costs acceptance. The one invariant that DOES matter is the position cap in plan(): no verify row
may address a KV position past the page table the runner handed us.

Row layout: every device tensor of a bucket of width Bw and T = K+1 verify rows per user is
j-major, row u = j*Bw + b (user b, verify depth j). Verify depth j carries token t0 (j=0) or
draft d_j (j>=1) at position p_b + j.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import base64
import os
import struct
from dataclasses import dataclass, field

try:  # this module is otherwise dependency-free (tools/ttx/tests load it in a python without loguru)
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

ENV_ON = "QWEN36_MTP_BATCH"
ENV_K = "QWEN36_MTP_BATCH_K"
ENV_BUCKETS = "QWEN36_MTP_BATCH_BUCKETS"
# Debug levers (docs/MTP-BATCH.md section 8a). FORCE_REJECT verifies the drafts at full width
# M = Bw*T but accepts none of them, so every row emits exactly the plain-decode token each step:
# output must then equal plain decode bit for bit unless the wider verify changes numerics.
# TRACE logs every step of the matching rows (request-id substring, "*" = all, "slot:N").
ENV_FORCE_REJECT = "QWEN36_MTP_BATCH_FORCE_REJECT"
ENV_TRACE = "QWEN36_MTP_BATCH_TRACE"
# PLAIN_ONLY builds and captures everything (same boot, same memory, sync + non-resident) but never
# speculates: the control that separates "the MTP arm's serving mode" from "speculation".
ENV_PLAIN_ONLY = "QWEN36_MTP_BATCH_PLAIN_ONLY"
# MUSTFAIL=commit_plus1 commits the GDN recurrent + conv state one verify depth PAST the accepted
# length (min(m_b+1, K)) while emitting the same tokens: the state then holds a rejected draft, so
# an equivalence gate against plain decode MUST fail. A gate that passes with it set is blind.
ENV_MUSTFAIL = "QWEN36_MTP_BATCH_MUSTFAIL"
MUSTFAIL_MODES = ("commit_plus1",)
# TIMING=1: host timers around each phase of a speculative step, with ttnn.synchronize_device
# between phases (the device side splits the commit/draft trace in two). A diagnostic, never a speed
# number: the syncs serialise host and device. PhaseTimer below does the pure-Python bookkeeping.
ENV_TIMING = "QWEN36_MTP_BATCH_TIMING"
# 16, not 64: a window closes every 16 steps AND when the batch drains (flush=idle), so a window never spans two
# serving bursts and the tail of the last burst loses < 16 steps. ttx (ttxlib/mtplog.py) parses these lines per cell.
TIMING_EVERY = 16
TIMING_COUNTS = ("rows", "emitted", "accepted", "drafted")
TIMING_PHASES = ("prep", "verify", "rb_verify", "accept", "commit", "draft", "rb_draft", "total")
TILE_ROWS = int(_tt_public_getenv("QWEN36_MTP_VERIFY_MAX_ROWS", "32"))  # decode matmul/norm/CCL verify row cap (default 32)
# Verify depth T = K+1 <= 16 (verify-depth-gt4): the deepest T any verify-path program covers (state_once.T_MAX, the
# kv_fold MAX_T, TT_SDPA_DEC_SPEC_FOLD's T <= 16 with attention/tp.py's candidate chunking). Rows stay Bw*T <= 32, so
# K=15 runs at Bw <= 2 only and K=7 at Bw <= 4. Was 7 (T <= 8) before the kv_fold cap was lifted.
MAX_K = 15
# Long-context acceptance stats (LC0). Always on (pure host counters, a few list increments per row-step):
#   depth_try[k] / depth_hit[k]: rows whose draft k+1 was compared (m >= k and n_draft > k) / accepted (m > k), so
#     depth_hit[k] / depth_try[k] is the CONDITIONAL acceptance at draft depth k+1;
#   mhist[m]: rows that accepted exactly m drafts (all rows, any n_draft);
#   pos_rows / pos_drafted / pos_accepted[i]: the same row counters bucketed by the row's base position p
#     (absolute context position) in POS_BIN-token bins.
# QWEN36_MTP_BATCH_CAPTURE=1 additionally keeps, per slot, every committed token id and every step's (m, n_draft)
# and logs ONE line per finished request (pop_capture, called on release): the raw material for offline window /
# lookup / KV-dtype replays. ttx (ttxlib/mtplog.py) writes those lines to a .mtp-capture.jsonl next to the record.
POS_BIN = 16384
ENV_CAPTURE = "QWEN36_MTP_BATCH_CAPTURE"
CAPTURE_MAX_TOKENS = 65536  # per request; a longer one is logged truncated (capture_truncated=1)


def enabled():
    return _tt_public_getenv(ENV_ON, "0") == "1"


def k_cap():
    k = int(_tt_public_getenv(ENV_K, "3"))
    if not 1 <= k <= MAX_K:
        raise ValueError(f"{ENV_K}={k}: must be in [1, {MAX_K}]")
    return k


def capture_on():
    return _tt_public_getenv(ENV_CAPTURE, "0") == "1"


def _ints(xs):
    return "/".join(str(int(x)) for x in xs)


def force_reject():
    return _tt_public_getenv(ENV_FORCE_REJECT, "0") == "1"


def plain_only():
    return _tt_public_getenv(ENV_PLAIN_ONLY, "0") == "1"


def mustfail():
    """The must-fail mode (None when unset). An unknown value refuses: a typo must not silently
    run the passing arm under a must-fail label."""
    v = _tt_public_getenv(ENV_MUSTFAIL, "")
    if not v:
        return None
    if v not in MUSTFAIL_MODES:
        raise ValueError(f"{ENV_MUSTFAIL}={v!r}: one of {MUSTFAIL_MODES}")
    return v


def timing():
    return _tt_public_getenv(ENV_TIMING, "0") == "1"


# OVERLAP=1 (docs/MTP-BATCH.md section 12): the commit/draft trace is enqueued without blocking and its draft
# ids come back through an event; the host finishes the step (stores the drafts per slot) at the start of
# whatever touches the slot state next (the next decode step, a prefill, a release, a remap). The vLLM host
# work between steps (update_from_output, schedule, input build) then runs while the device commits and
# drafts. OVERLAP_TIMING=1 logs host wall clocks per phase WITHOUT device syncs (the sync phase split,
# TIMING=1, would serialise exactly what this overlaps, so the two refuse each other).
ENV_OVERLAP = "QWEN36_MTP_OVERLAP"
ENV_OVERLAP_TIMING = "QWEN36_MTP_OVERLAP_TIMING"
OVERLAP_PHASES = ("host", "wait_c", "prep", "verify", "accept", "total")
# OVERLAP_TIMING sub-phases (diagnostic, host clocks): the drain (d_ev: the draft-event wait, d_read: the draft/probs
# readback + chain parse, d_chk: the DEVACCEPT m-page check) and the prep (p_plan: host.plan, p_in: the input dicts,
# p_wr: the H2D input write enqueue, p_cc: the deferred-commit event wait before V).
OVERLAP_SUB_PHASES = ("d_ev", "d_read", "d_chk", "p_plan", "p_in", "p_wr", "p_cc")
# EARLY_RESEED=1 (needs OVERLAP): the MTP-KV reseed over the T verify rows depends on the plan only (positions p+j, origin),
# never on the accepted length, so it is captured as its own trace and enqueued right after verify; the device runs it while
# the host reads the verify ids and accepts. The commit + draft trace follows the accept as before.
ENV_EARLY_RESEED = "QWEN36_MTP_EARLY_RESEED"
# QWEN36_MTP_ASYNC (plugin, vllm_tt_plugin/mtp_async.py): read here only to tighten the KV capacity to the
# pages vLLM actually allocated (alloc_capacity), since under async scheduling the allocation is sized from a
# count that lags by up to K.
ENV_ASYNC = "QWEN36_MTP_ASYNC"


def overlap():
    return _tt_public_getenv(ENV_OVERLAP, "0") == "1"


def overlap_timing():
    v = _tt_public_getenv(ENV_OVERLAP_TIMING, "0") == "1"
    if v and not overlap():
        raise ValueError(f"{ENV_OVERLAP_TIMING}=1 needs {ENV_OVERLAP}=1")
    if v and timing():
        raise ValueError(f"{ENV_OVERLAP_TIMING}=1 and {ENV_TIMING}=1 refuse each other (the sync split serialises the overlap)")
    return v


def early_reseed():
    v = _tt_public_getenv(ENV_EARLY_RESEED, "0") == "1"
    if v and not overlap():
        raise ValueError(f"{ENV_EARLY_RESEED}=1 needs {ENV_OVERLAP}=1 (its verify-id read is the overlap's event read)")
    if v and timing():
        raise ValueError(f"{ENV_EARLY_RESEED}=1 and {ENV_TIMING}=1 refuse each other (TIMING captures its own C split)")
    return v


# SO1_CSPLIT=1 (Option A, docs/MTP-BATCH.md section 16f; needs OVERLAP + QWEN36_MTP_GDN_STATE_ONCE, refuses
# TIMING): the commit/draft C trace (C2 under EARLY_RESEED) is captured as cc (GDN commit) + cd (drafts, plus the
# reseed when it is not already in R), cd is
# enqueued first so the draft event (wait_c) no longer includes the so1 commit recompute, and the commit event is
# waited right before the next verify. Opt-in: unset, OVERLAP+STATE_ONCE keeps the one C trace (byte-identical).
ENV_SO1_CSPLIT = "QWEN36_MTP_SO1_CSPLIT"


def so1_csplit():
    v = _tt_public_getenv(ENV_SO1_CSPLIT, "0") == "1"
    if v and not overlap():
        raise ValueError(f"{ENV_SO1_CSPLIT}=1 needs {ENV_OVERLAP}=1 (it splits the overlap's C trace)")
    if v and timing():
        raise ValueError(f"{ENV_SO1_CSPLIT}=1 and {ENV_TIMING}=1 refuse each other (TIMING captures its own C split)")
    return v


def async_sched():
    return _tt_public_getenv(ENV_ASYNC, "0") == "1"


def alloc_capacity(page_table_row, block_size):
    """Positions backed by pages vLLM allocated to this row: the leading non-null entries times block_size.
    vLLM's null block is id 0 and a row's unallocated tail is 0-filled, so a write past this bound would land
    in the null block and the position's KV would be lost. Sized to the scheduler's lookahead (2K async)."""
    n = 0
    for v in page_table_row:
        if int(v) == 0:
            break
        n += 1
    return n * int(block_size)


# QWEN36_MTP_CAP_TORCH: the same bound from a vectorised scan of the page-table row, without the host
# list conversion and the python loop.  The served path converts the whole 12296-column row with
# .tolist() and walks it in python on EVERY speculative step, inside the "prep" phase where the device
# is IDLE (the V trace cannot be enqueued before it) -- so the cost is on the decode cycle's critical
# path.  Measured (host wall clock, the sim runtime): the served scan 0.119-0.149 ms/row, the
# vectorised scan 0.021 ms/row, the same integer on every shape (empty, no nulls, leading nulls, a
# hole, random rows).
ENV_CAP_TORCH = "QWEN36_MTP_CAP_TORCH"
ENV_CAP_TORCH_MUSTFAIL = ENV_CAP_TORCH + "_MUSTFAIL"
SELFTEST_CAP_TORCH_BANNER = "QWEN36_MTP_CAP_TORCH self-test exact"


def cap_torch_enabled():
    v = _tt_public_getenv(ENV_CAP_TORCH, "0").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{ENV_CAP_TORCH}={v!r}: expected unset, 0 or 1")
    return v == "1"


def cap_torch_mustfail():
    v = _tt_public_getenv(ENV_CAP_TORCH_MUSTFAIL, "0").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{ENV_CAP_TORCH_MUSTFAIL}={v!r}: expected unset, 0 or 1")
    return v == "1"


def alloc_capacity_t(page_table_row, block_size):
    """The leading non-null entries of a page-table row, times the block size, from a vectorised scan
    of the row: the index of the first null entry (numpy.argmax returns the FIRST maximal, and the
    mask's True marks the null entries).  Same integer as alloc_capacity on every shape."""
    import numpy as np

    a = page_table_row.numpy()
    nulls = a == 0
    n = int(a.size) if not bool(nulls.any()) else int(np.argmax(nulls))
    if cap_torch_mustfail():
        # MUSTFAIL: the count of ALL non-null entries, not the leading prefix.  A row with a hole
        # then differs from the served scan, which the self-test must catch (it refuses the boot).
        n = int((a != 0).sum())
    return n * int(block_size)


def selftest_cap_torch(nblk=12296, seed=7, block_size=64):
    """Fail closed: the vectorised capacity must equal the served list scan on every row shape, and
    the row with a HOLE (a null entry in the middle) is the wrong-input branch: the MUSTFAIL variant
    (the count of all non-null entries) differs on it, so a wrong candidate is caught."""
    import torch

    i64 = torch.int64
    rows = [
        torch.zeros(0, dtype=i64),  # empty
        torch.ones(16, dtype=i64),  # no nulls at all
        torch.cat([torch.zeros(4, dtype=i64), torch.ones(4, dtype=i64)]),  # leading nulls
        # a HOLE: non-null entries before AND after it.  The served scan stops at the hole (2
        # entries); the MUSTFAIL variant counts all 6 -> they differ, so a wrong candidate is caught.
        torch.cat([torch.ones(2, dtype=i64), torch.zeros(2, dtype=i64), torch.ones(4, dtype=i64)]),
    ]
    g = torch.Generator().manual_seed(seed)
    for _ in range(8):
        rows.append(torch.randint(0, 5, (nblk,), generator=g, dtype=i64))
    bad = []
    for i, r in enumerate(rows):
        ref = alloc_capacity(r.tolist(), block_size)
        got = alloc_capacity_t(r, block_size)
        if int(got) != int(ref):
            bad.append({"row": i, "n": int(r.numel()), "nulls": int((r == 0).sum()), "served": int(ref),
                        "vectorised": int(got)})
    if bad:
        if cap_torch_mustfail():
            raise RuntimeError(
                f"{ENV_CAP_TORCH_MUSTFAIL} engaged: the vectorised capacity differs from the served scan on "
                f"{len(bad)} row(s) (the MUSTFAIL count of all non-null entries): {bad}")
        raise RuntimeError(
            f"{ENV_CAP_TORCH} self-test REFUSED the boot: the vectorised capacity differs from the served scan: {bad}")
    logger.info(f"{SELFTEST_CAP_TORCH_BANNER}: {len(rows)} rows (empty, no nulls, leading nulls, a hole, "
                f"{nblk}-column random rows) give the served scan's integer")
    return {"rows": len(rows), "exact": True}


ENV_STATE_SLOT = "QWEN36_MTP_STATE_SLOT"
SLOT_SKIP = 0xFFFF  # slot-index entry that leaves the row's GDN state untouched (gdn/state_slot.py SKIP)
SLOT_LEN = max(32, TILE_ROWS)  # one uint32 page, one entry per decode row


def state_slot():
    return _tt_public_getenv(ENV_STATE_SLOT, "0") == "1"


def slot_index(gdn_onehot, bw, T):
    """The QWEN36_MTP_STATE_SLOT commit page from the where-chain's one-hot rows: entry b (< bw) is
    the single j with gdn_onehot[j][b] == 1 (the verify depth whose state row b commits), entries >= bw
    are SLOT_SKIP (the persistent state rows outside the bucket are never touched, as the where path
    leaves them). Both commit paths are built from the SAME one-hot, so a must-fail depth change
    (commit_plus1) moves both. Refuses a row that is not exactly one-hot."""
    assert len(gdn_onehot) == T and bw <= SLOT_LEN, (len(gdn_onehot), T, bw)
    out = [SLOT_SKIP] * SLOT_LEN
    for b in range(bw):
        hot = [j for j in range(T) if gdn_onehot[j][b] == 1.0]
        if len(hot) != 1 or any(gdn_onehot[j][b] not in (0.0, 1.0) for j in range(T)):
            raise ValueError(f"row {b}: commit select is not one-hot ({[gdn_onehot[j][b] for j in range(T)]})")
        out[b] = hot[0]
    return out



# ---------------------------------------------------------------------------------------------- dual trace family
# QWEN36_MTP_SPEC_SAMPLE_DUAL=1 (docs/MTP-BATCH.md section 26; needs QWEN36_MTP_SPEC_SAMPLE_SERVE=1): every speculating
# bucket has TWO captured trace families, the sampled one (the served exact rule, the device sampler: -s6-ss's ops) and
# the greedy one (S6's ops: no sampler program anywhere). Each step picks one from its rows' sampling params: a batch
# whose ACTIVE rows are all T = 0 runs the greedy family, any active T > 0 row (either rule) runs the sampled family.
# Both families read and write the same persistent state (GDN state, KV, the MTP head's KV, the host slot state); the
# noise is a stateless hash of (seed, position, stream). The one thing a family switch changes is the drafter's
# candidate block q, which only the sampled chain produces: after a greedy step a row's drafts have no q. Only an
# exact-rule row needs q (the rule's accept test); every other row's rule input is ignored, so plan(need_q=...) gives
# it a zero block, and an exact row without q drops its drafts (n_draft = 0: the step emits the target's own sample,
# which is exact). A T > 0 row is never in a greedy step, so that drop is a guard, counted, not a served path.
ENV_DUAL = "QWEN36_MTP_SPEC_SAMPLE_DUAL"
ENV_DUAL_MUSTFAIL = "QWEN36_MTP_SPEC_SAMPLE_DUAL_MUSTFAIL"
DUAL_SAMPLED, DUAL_GREEDY = "s", "g"


def _col(sp, name, n, default):
    v = getattr(sp, name, None) if sp is not None else None
    v = list(v.tolist() if hasattr(v, "tolist") else (v or []))
    return [v[b] if b < len(v) and v[b] is not None else default for b in range(n)]


def dual_family(sp, positions, bw, rules=None, mustfail=False, have_greedy=True):
    """The trace family of one step of width bw: (family, need_q, n_sampled, reason).

    A row is active iff positions[b] >= 0 (the runner's padded rows are -1). n_sampled counts the active rows with
    temperature > 0 (the device sampler's own test: _Bucket._spec_params' greedy = T <= 0). need_q[b] is 1 where the
    step's rule needs the drafter's q for row b: active, T > 0 and the exact rule (rules[b] == 1; None = every T > 0
    row exact). mustfail (QWEN36_MTP_SPEC_SAMPLE_DUAL_MUSTFAIL): a step with exactly ONE sampled row runs the greedy
    family (that row then emits argmax tokens: a distribution gate MUST fail). have_greedy False (no greedy family at
    this bucket, the TRACE budget): the sampled family, reason "no greedy trace"."""
    temps = [float(t) for t in _col(sp, "temperature", bw, 0.0)]
    act = [int(positions[b]) >= 0 for b in range(bw)]
    samp = [1 if (act[b] and temps[b] > 0.0) else 0 for b in range(bw)]
    rr = list(rules) if rules is not None else [1] * bw
    need_q = [1 if (samp[b] and int(rr[b]) == 1) else 0 for b in range(bw)]
    n = sum(samp)
    fam = DUAL_GREEDY if n == 0 else DUAL_SAMPLED
    reason = None
    if mustfail and n == 1:
        fam = DUAL_GREEDY
        reason = "mustfail"
    if fam == DUAL_GREEDY and not have_greedy:
        fam, reason = DUAL_SAMPLED, "no greedy trace"
    return fam, need_q, n, reason

# ---------------------------------------------------------------------------------------------- device accept
# QWEN36_MTP_DEVACCEPT=1 (docs/MTP-BATCH.md section 25; needs STATE_SLOT + OVERLAP, refuses TIMING): the accepted lengths
# m_b and every input of the commit/draft traces that the host accept used to write after reading the verify back --
# the slot page gslot, the draft anchor rows, t0n, the K draft KV / RoPE positions and (QWEN36_MTP_SPEC_SAMPLE_SERVE)
# the K chain-noise blocks d_sgum{k} -- are computed ON DEVICE by one JIT program (tt/mtp_accept.py,
# kernels/mtp_accept/accept.cpp) captured as the first program of the draft trace (cd under SO1_CSPLIT, C2 / C
# otherwise). Its inputs are the verify's own outputs (the argmax/sample ids; under the served exact rule also the
# rule's accept flags and residual ids) plus plan-only blocks the host writes BEFORE the verify: one plan page
# (da_plan_page) and, under the served rule, the chain noise for EVERY possible m (da_noise_rows: (K+1)*bw rows per
# depth; the kernel copies row m_b*bw + b). V, R and the commit/draft traces are then enqueued back to back. The host
# still runs accept() on the read-back verify outputs for the emitted tokens and the slot state, off the device's
# critical path, and checks the device's m / t0n page against it at the next drain (fail closed).
#
# This is c152's QWEN36_MTP_DEVICE_ACCEPT (branch mtp-device-accept, greedy only, inert on the 10-01 cm tree) ported to
# the S6 tree (SO1_CSPLIT: the program leads cd) and extended to exact-sampled rows. da_reference() is the kernel in
# Python, word for word; da_expected() is what the host-accept path writes today. The tests, the boot self-test and the
# ttsim gate hold the three equal.
ENV_DEVACCEPT = "QWEN36_MTP_DEVACCEPT"
ENV_DEVACCEPT_MUSTFAIL = "QWEN36_MTP_DEVACCEPT_MUSTFAIL"
DA_PAGE_LEN = 256  # int32 words of the plan page (one 1 KiB ROW_MAJOR page)
DA_FLAGS = 0  # word 0: bit 0 commit_plus1 (QWEN36_MTP_BATCH_MUSTFAIL), bit 1 the device-accept must-fail (row 0)
DA_N, DA_ACTIVE, DA_KVBASE, DA_ROPEBASE, DA_LIM, DA_EXACT = 32, 64, 96, 128, 160, 192  # per-row blocks of 32 words
DA_FLAG_COMMIT_PLUS1, DA_FLAG_MUSTFAIL = 1, 2
DA_LIM_OPEN = 1 << 30  # lim for warmup rows: no draft position is capped
DA_M_LEN = 64  # the m page: words 0..31 m_b, words 32..63 t0n_b (what the host checks at the drain)
DA_F32_HALF, DA_F32_INF = 0x3F000000, 0x7F800000  # fp32 bits of 0.5 and +inf


def devaccept():
    return _tt_public_getenv(ENV_DEVACCEPT, "0") == "1"


def devaccept_mustfail():
    v = _tt_public_getenv(ENV_DEVACCEPT_MUSTFAIL, "0") == "1"
    if v and not devaccept():
        raise ValueError(f"{ENV_DEVACCEPT_MUSTFAIL}=1 needs {ENV_DEVACCEPT}=1")
    return v


def devaccept_check(slot, overlap_on, timing_on, lookup_mustfail=None):
    """The boot refusals of QWEN36_MTP_DEVACCEPT (fail closed). Returns the list of reasons (empty = ok)."""
    bad = []
    if not slot:
        bad.append(f"{ENV_STATE_SLOT}=1 (the device writes the slot-index commit page; the where-chain's one-hot masks "
                   f"have no device producer)")
    if not overlap_on:
        bad.append(f"{ENV_OVERLAP}=1 (V and C are enqueued back to back; the m page is read with the drafts)")
    if timing_on:
        bad.append(f"{ENV_TIMING} off (the synced phase split writes the C inputs from the host accept)")
    if lookup_mustfail == "trust":
        bad.append(f"{ENV_LOOKUP_MUSTFAIL} != trust (the device has no 'commit the lookup drafts blind' branch)")
    return bad


# ---------------------------------------------------------------------------------------- QWEN36_MTP_DEVFEED
# QWEN36_MTP_DEVFEED=1 (docs/MTP-BATCH.md section 31): the sampled family's draft trace ends with one program
# (mtp_accept.feed) that writes the next verify's draft-dependent inputs (v_tok rows j >= 1 with row 0 = the device
# t0n, v_sqp, v_sqi, v_sx) from the chain's outputs, so a FAST step (feed_predict) writes only the draft-independent
# inputs and enqueues its verify before the host reads the previous step's drafts back. Every other step is the S10
# step unchanged (the host writes every input after the drain).
ENV_DEVFEED = "QWEN36_MTP_DEVFEED"
ENV_DEVFEED_MUSTFAIL = "QWEN36_MTP_DEVFEED_MUSTFAIL"
ENV_DEVFEED_CHECK = "QWEN36_MTP_DEVFEED_CHECK"
# QWEN36_MTP_DEVFEED_PAD (docs/MTP-BATCH.md section 32): a step with PADDED rows (position -1: no request in the row, at
# a cell's ramp-up and tail) may be a fast step too. A padded row's verify outputs are never read (accept() skips it, the
# device accept gets n=0 / active=0, the commit writes nothing for it, its KV positions are -1), so the device feed's
# values in its rows are don't-care: the plan equality and the step self-test compare the ACTIVE rows only.
ENV_DEVFEED_PAD = "QWEN36_MTP_DEVFEED_PAD"
ENV_DEVFEED_PAD_MUSTFAIL = "QWEN36_MTP_DEVFEED_PAD_MUSTFAIL"
FEED_NAMES = ("v_tok", "v_sqp", "v_sqi", "v_sx")  # the inputs the feed program writes
FEED_OV_PHASES = ("f_pred", "f_in", "f_wr", "f_enq", "f_drain", "f_plan", "f_fast")  # OVERLAP_TIMING fast-step split


def devfeed():
    v = _tt_public_getenv(ENV_DEVFEED, "0").strip()
    if v not in ("0", "1", ""):
        raise ValueError(f"{ENV_DEVFEED}={v!r}: expected 0 or 1")
    return v == "1"


def devfeed_mustfail():
    v = _tt_public_getenv(ENV_DEVFEED_MUSTFAIL, "0").strip() == "1"
    if v and not devfeed():
        raise ValueError(f"{ENV_DEVFEED_MUSTFAIL}=1 needs {ENV_DEVFEED}=1")
    return v


def devfeed_pad():
    v = _tt_public_getenv(ENV_DEVFEED_PAD, "0").strip()
    if v not in ("0", "1", ""):
        raise ValueError(f"{ENV_DEVFEED_PAD}={v!r}: expected 0 or 1")
    if v == "1" and not devfeed():
        raise ValueError(f"{ENV_DEVFEED_PAD}=1 needs {ENV_DEVFEED}=1")
    return v == "1"


def devfeed_pad_mustfail():
    """The padded-row step self-test compares EVERY row (no active-row mask): it MUST refuse at the first checked padded
    fast step, because the feed writes the chain's values into padded rows where the host build writes zeros / t0."""
    v = _tt_public_getenv(ENV_DEVFEED_PAD_MUSTFAIL, "0").strip() == "1"
    if v and not devfeed_pad():
        raise ValueError(f"{ENV_DEVFEED_PAD_MUSTFAIL}=1 needs {ENV_DEVFEED_PAD}=1")
    return v


def feed_active_rows(plan, M, K):
    """The fed-buffer rows the step self-test compares under QWEN36_MTP_DEVFEED_PAD: v_tok rows j*bw + b (j <= K) and the
    q / x rows j*bw + b (j < K) of ACTIVE rows b, plus the zero tail rows >= K*bw (written for every row by the feed).
    Returns (tok_rows, q_rows)."""
    bw = plan.bw
    act = [b for b in range(bw) if plan.active[b]]
    tok = sorted(j * bw + b for j in range(K + 1) for b in act)
    q = sorted([j * bw + b for j in range(K) for b in act] + list(range(K * bw, M)))
    return tok, q


def devfeed_check_steps():
    """Fast steps whose fed inputs are read back and compared with the host build (in-boot step self-test)."""
    try:
        return max(0, int(_tt_public_getenv(ENV_DEVFEED_CHECK, "64")))
    except ValueError as e:
        raise ValueError(f"{ENV_DEVFEED_CHECK} must be an int") from e


def map_q_ids(qib, ids):
    """The chain's q-id blocks (draft-vocab SUB-ids as fp32) -> FULL ids as fp32, as served (mtp_batch._read_chain);
    ids: the draft vocab's id table, a torch int64 tensor."""
    return [ids[t.long().clamp_(0, ids.numel() - 1)].float() for t in qib]


def spec_v_blocks(plan, bw, K, M, C2, exact):
    """QWEN36_MTP_SPEC_SAMPLE: the verify's draft-dependent rule blocks of one step (rows j*bw + b): v_sqp / v_sqi the
    carried q block of draft j (zeros on rows without a draft), v_sx the draft id, v_sxf 1.0 where the row is EXACT and
    j < n_draft. The one builder of both the served host write (mtp_batch._spec_block) and the device feed's reference."""
    import torch

    qp = torch.zeros((M, C2), dtype=torch.float32)
    qi = torch.zeros((M, C2), dtype=torch.float32)
    xs = torch.zeros((M, 1), dtype=torch.float32)
    xf = torch.zeros((M, 1), dtype=torch.float32)
    if plan is not None and plan.qprobs is not None:
        qp[: K * bw] = plan.qprobs
        qi[: K * bw] = plan.qids
        for j in range(K):
            for b in range(bw):
                xs[j * bw + b, 0] = float(plan.drafts[b][j]) if j < len(plan.drafts[b]) else 0.0
                xf[j * bw + b, 0] = 1.0 if (plan.active[b] and exact[b] and j < plan.n_draft[b]) else 0.0
    return qp, qi, xs, xf


FEED_PLAN_FIELDS = ("bw", "K", "T", "active", "p", "t0", "n_draft", "origin", "verify_pos", "verify_rope_pos",
                    "kv_capacity", "source", "lookup_mustfail")
FEED_STATS = ("fresh", "stale", "capped", "warm_kept")


def feed_plan_diff(pred, real, bw):
    """The draft-independent fields where the predicted (fast-path) plan and the real plan differ, plus the verify
    tokens' row 0 block (t0) and the real plan's draft count (every row must verify K drafts). Empty = equal."""
    bad = [f for f in FEED_PLAN_FIELDS if getattr(pred, f) != getattr(real, f)]
    if list(pred.verify_tokens[:bw]) != list(real.verify_tokens[:bw]):
        bad.append("verify_tokens[t0]")
    if any(int(n) != int(real.K) for n, a in zip(real.n_draft, real.active) if a):
        bad.append("n_draft<K")
    return bad


def da_plan_page(plan, reject_all=False, mustfail_mode=None, da_mustfail=False, exact=None):
    """The plan-only page of the device accept (written before V). Per row b < bw:
      n      usable drafts n_draft[b] (0 for an inactive row; 0 for every row under FORCE_REJECT, which is what
             accept() does to m after counting would_accept)
      active 1 / 0
      kvbase p_b - origin_b (MTP KV index of position p_b)
      rbase  p_b (absolute RoPE position)
      lim    kv_capacity_b - p_b + origin_b: draft step k is issued iff m_b + k < lim (accept(): pk - origin < cap)
      exact  1 where the served exact rule judges the row (QWEN36_MTP_SPEC_SAMPLE: rule exact and T > 0), else 0
    Word 0 carries the must-fail flags. Unused words are 0."""
    bw = plan.bw
    assert bw <= SLOT_LEN
    w = [0] * DA_PAGE_LEN
    w[DA_FLAGS] = (DA_FLAG_COMMIT_PLUS1 if mustfail_mode == "commit_plus1" else 0) | (DA_FLAG_MUSTFAIL if da_mustfail else 0)
    for b in range(bw):
        act = bool(plan.active[b])
        w[DA_N + b] = 0 if (reject_all or not act) else int(plan.n_draft[b])
        w[DA_ACTIVE + b] = 1 if act else 0
        if act:
            w[DA_KVBASE + b] = int(plan.p[b]) - int(plan.origin[b])
            w[DA_ROPEBASE + b] = int(plan.p[b])
            w[DA_LIM + b] = int(plan.kv_capacity[b]) - int(plan.p[b]) + int(plan.origin[b])
            w[DA_EXACT + b] = 1 if (exact is not None and exact[b]) else 0
    return w


def da_warmup_page(bw):
    """Warmup / compile values: every row active at position 0 with no usable draft, so the device accept writes
    the warmup inputs host_inputs() allocates (gslot 0, anchor b, d_pos = d_rpos = k), except t0n, which becomes the
    eager verify's own id of row b (a valid token id) instead of 0."""
    w = [0] * DA_PAGE_LEN
    for b in range(bw):
        w[DA_ACTIVE + b] = 1
        w[DA_LIM + b] = DA_LIM_OPEN
    return w


def _i32(v):
    """A 32-bit word as the int32 the device tensors hold (-1 = 0xFFFFFFFF)."""
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def da_accept_bit(bits):
    """The kernel's test of an fp32 accept flag, on its bits: the same truth value as the host's float(a) > 0.5 for
    EVERY fp32 bit pattern (positive normals/denormals above 0.5 and +inf pass; NaN, zeros, negatives, 0.5 and below
    do not)."""
    s = _i32(bits)
    return DA_F32_HALF < s <= DA_F32_INF


def da_reference(page, v_tok, ids, bw, T, accept=None, resid=None, noise=None):
    """The device accept kernel (kernels/mtp_accept/accept.cpp) in Python, word for word. page: da_plan_page;
    v_tok: the M = T*bw verify tokens (j-major); ids: the M verify ids; accept / resid (served exact rule): the rule's M
    accept flags (floats) and residual ids; noise (served exact rule): K tables of (K+1)*bw rows (da_noise_rows).
    Returns the outputs it writes: {gslot [32], anchor [bw], t0n [bw], d_pos [K][bw], d_rpos [K][bw], m [64]
    (m_b then t0n_b), and with noise: d_sgum [K][bw] rows} (int32 views, -1 = skip)."""
    K = T - 1
    flags = int(page[DA_FLAGS])
    u = [int(x) & 0xFFFFFFFF for x in ids]
    vt = [int(x) & 0xFFFFFFFF for x in v_tok]
    ab = [f32_bits(a) for a in accept] if accept is not None else None
    rs = [int(x) & 0xFFFFFFFF for x in resid] if resid is not None else None
    gslot, mout = [SLOT_SKIP] * SLOT_LEN, [0] * DA_M_LEN
    anchor, t0n = [0] * bw, [0] * bw
    d_pos = [[0] * bw for _ in range(K)]
    d_rpos = [[0] * bw for _ in range(K)]
    sg = [[None] * bw for _ in range(K)] if noise is not None else None
    for b in range(bw):
        n = int(page[DA_N + b]) & 0xFFFFFFFF
        act = int(page[DA_ACTIVE + b]) != 0
        ex = ab is not None and int(page[DA_EXACT + b]) != 0
        kvb, rb, lim = _i32(page[DA_KVBASE + b]), _i32(page[DA_ROPEBASE + b]), _i32(page[DA_LIM + b])
        m = 0
        if ex:
            while m < n and m < K and da_accept_bit(ab[m * bw + b]):
                m += 1
        else:
            while m < n and m < K and vt[(m + 1) * bw + b] == u[m * bw + b]:
                m += 1
        if b == 0 and act and flags & DA_FLAG_MUSTFAIL:
            m = m + 1 if m < T - 1 else m - 1
        commit = (min(m + 1, T - 1) if flags & DA_FLAG_COMMIT_PLUS1 else m) if act else 0
        gslot[b] = commit
        anchor[b] = m * bw + b
        tail = rs[m * bw + b] if (ex and m < n) else u[m * bw + b]
        t0n[b] = _i32(tail) if act else 0
        for k in range(K):
            ok = act and m + k < lim
            d_pos[k][b] = kvb + m + k if ok else -1
            d_rpos[k][b] = rb + m + k if ok else -1
            if sg is not None:
                sg[k][b] = noise[k][m * bw + b]
        mout[b] = m
        mout[32 + b] = t0n[b] & 0xFFFFFFFF
    out = {"gslot": gslot, "anchor": anchor, "t0n": t0n, "d_pos": d_pos, "d_rpos": d_rpos, "m": mout}
    if sg is not None:
        out["d_sgum"] = sg
    return out


def da_expected(plan, acc, d_sgum=None):
    """What the host-accept path writes into the same inputs today (host_inputs from accept()), in da_reference's
    format. m page: acc.m then t0_next for b < bw (0 for inactive rows), 0 past the bucket. d_sgum: the K [bw] rows of
    the host path's chain noise (served rule only)."""
    bw, T = plan.bw, plan.T
    m = [0] * DA_M_LEN
    for b in range(bw):
        m[b] = int(acc.m[b])
        m[32 + b] = int(acc.t0_next[b]) & 0xFFFFFFFF
    out = {"gslot": slot_index(acc.gdn_onehot, bw, T), "anchor": [int(x) for x in acc.anchor_idx],
           "t0n": [_i32(x) for x in acc.t0_next], "d_pos": [[int(x) for x in r] for r in acc.draft_pos],
           "d_rpos": [[int(x) for x in r] for r in acc.draft_rope_pos], "m": m}
    if d_sgum is not None:
        out["d_sgum"] = d_sgum
    return out


DA_FIELDS = ("gslot", "anchor", "t0n", "d_pos", "d_rpos", "m", "d_sgum")


def da_diff(got, exp):
    """Fields of two device-accept outputs that differ (empty = bitwise equal). d_sgum rows compare as lists of fp32
    BITS (so -0.0 != 0.0 and NaN == NaN by pattern)."""
    bad = []
    for k in DA_FIELDS:
        a, e = got.get(k), exp.get(k)
        if k == "d_sgum" and a is not None and e is not None:
            a = [[[f32_bits(x) for x in row] for row in dep] for dep in a]
            e = [[[f32_bits(x) for x in row] for row in dep] for dep in e]
        if a != e:
            bad.append(k)
    return bad


def da_check_page(page_words, acc, bw):
    """The drain check: the device's m page (one chip) against the host accept. Returns the rows that differ."""
    bad = []
    for b in range(bw):
        if int(page_words[b]) != int(acc.m[b]) or (int(page_words[32 + b]) & 0xFFFFFFFF) != (int(acc.t0_next[b]) & 0xFFFFFFFF):
            bad.append(b)
    return bad


def da_noise_positions(plan, K):
    """Chain-noise positions for EVERY possible accepted length: table row i*bw + b (i = 0..K) of depth k is the
    position p_b + i + 2 + k, i.e. the row _spec_d_block writes for m_b = i (base p + m + 2 + k; p = -1 rows give the
    same 1 + i + k as the host path). Returns [K][(K+1)*bw] positions."""
    bw = plan.bw
    out = []
    for k in range(K):
        row = []
        for i in range(K + 1):
            for b in range(bw):
                row.append(int(plan.p[b]) + i + 2 + k)
        out.append(row)
    return out


DA_CASES = ("all", "none", "partial", "mixed", "inactive", "inactive_row0", "fresh", "capped", "reject_all",
            "commit_plus1")


def da_case(name, bw, K, seed=0, vocab=1 << 20, exact_rows=None):
    """One synthetic device-accept case through the REAL host code: a scratch MTPBatchHost with seeded slots plans
    a step, ids are built to give the named acceptance pattern, and accept() runs. Returns (host, plan, ids, acc, spec)
      all / none / partial / mixed   every row accepts n_draft / 0 / a random prefix / a per-row mix
      inactive / inactive_row0       some rows (row 0 too) padded (position -1)
      fresh                          some slots empty or stale (n_draft 0)
      capped                         KV capacity a few positions past p, origin 0: fewer usable drafts and
                                     draft positions past the capacity (d_pos = -1)
      reject_all / commit_plus1      FORCE_REJECT / MUSTFAIL=commit_plus1 hosts
    exact_rows (a [bw] 0/1 list, or None = greedy): rows judged by the exact rule. Their acceptance comes from a
    synthetic accept-flag vector (fp32: 1.0 / 0.0 and the edge patterns 0.5, -0.0, NaN, +inf, a denormal) that gives
    the named pattern, and a residual id per row; their ids are random (no compare). spec is accept()'s rule data
    ({"accept", "resid", "exact", "greedy"}) or None."""
    import math
    import random

    if name not in DA_CASES:
        raise ValueError(f"device-accept case {name!r}: one of {DA_CASES}")
    if exact_rows is not None and any(exact_rows) and name == "reject_all":
        raise ValueError("reject_all with exact rows: FORCE_REJECT is refused under the served rule (no exact tail)")
    rng = random.Random(f"{name}/{bw}/{K}/{seed}/{exact_rows}")
    T = K + 1
    host = MTPBatchHost(bw, K, block_size=64, reject_all=(name == "reject_all"), trace="",
                        mustfail_mode=("commit_plus1" if name == "commit_plus1" else ""))
    tokens = [rng.randrange(vocab) for _ in range(bw)]
    positions = [rng.randrange(K + 8, 30000) for _ in range(bw)]
    caps = [32768] * bw
    if name == "inactive":
        for b in rng.sample(range(bw), max(1, bw // 2)):
            positions[b] = -1
        positions[0] = max(positions[0], K + 8)
    if name == "inactive_row0":
        positions[0] = -1
        if bw > 2:
            positions[bw - 1] = -1
    for b in range(bw):
        if positions[b] < 0:
            continue
        origin = positions[b] - rng.randrange(0, K + 8)
        if name == "capped":
            origin = 0
            caps[b] = positions[b] + 1 + rng.randrange(0, K + 1)
        st = SlotState(origin=origin, next_p=positions[b], t0=tokens[b],
                       drafts=[rng.randrange(vocab) for _ in range(K)])
        if name == "fresh" and rng.random() < 0.5:
            st = None if rng.random() < 0.5 else SlotState(origin=origin, next_p=positions[b] + 1, t0=tokens[b],
                                                             drafts=list(st.drafts))  # stale: re-origined
        host.slots[b] = st
    plan = host.plan(tokens, positions, bw, K, caps)
    ids = [rng.randrange(vocab) for _ in range(T * bw)]
    M = T * bw
    acc_f = [0.0] * M
    resid = [rng.randrange(vocab) for _ in range(M)]
    yes = (1.0, 1.0, 1.0, math.inf, 0.75, 0.50000006)  # 0.50000006 = the fp32 just above 0.5
    no = (0.0, 0.0, -0.0, 0.5, float("nan"), -1.0, 1e-45, -math.inf)
    for b in range(bw):
        ex = exact_rows is not None and bool(exact_rows[b])
        n = plan.n_draft[b] if plan.active[b] else 0
        if name in ("all", "reject_all", "commit_plus1", "capped"):
            want = n
        elif name == "none":
            want = 0
        elif name == "mixed":
            want = (0, n, rng.randrange(0, n + 1) if n else 0)[b % 3]
        else:
            want = rng.randrange(0, n + 1)
        for j in range(K):
            if ex:
                acc_f[j * bw + b] = rng.choice(yes) if j < want else rng.choice(no)
                if j >= want and rng.random() < 0.3:
                    ids[j * bw + b] = int(plan.drafts[b][j])  # an exact row ignores the compare: a match must not count
                continue
            if not plan.active[b]:
                continue
            d = int(plan.drafts[b][j])
            if j < want:
                ids[j * bw + b] = d
            elif ids[j * bw + b] == d:
                ids[j * bw + b] = (d + 1) % vocab  # break the prefix exactly at want
        if ex:
            for j in range(K, T):
                acc_f[j * bw + b] = rng.choice(yes + no)  # depth K has no draft: never read
    spec = None
    if exact_rows is not None:
        spec = {"accept": acc_f, "resid": resid, "exact": [int(bool(x)) for x in exact_rows],
                "greedy": [0] * bw}
    acc = host.accept(plan, ids, spec)
    return host, plan, ids, acc, spec


class PhaseTimer:
    """Per-bucket phase times (ms) of speculative steps. add() one step's phases; every `every`
    steps (all buckets together) lines() returns one log line per bucket with the window means and
    the running means, then opens a new window. Pure bookkeeping: no clock, no device.

    Each line also carries (after the running means) `t=` (the caller's clock at the window's last
    step), `flush=` (periodic | idle), the window's row counts (rows, emitted, accepted, drafted:
    accepted tokens per row-step = emitted / rows) and `samples_ms:` the per-step values of every
    phase, so per-step p50/p90 per cell are exact (tools/ttx/ttxlib/mtplog.py)."""

    def __init__(self, every=TIMING_EVERY, phases=TIMING_PHASES, mode=None):
        self.every = int(every)
        self.phases = tuple(phases)
        self.mode = mode  # "overlap": host wall clocks, no device syncs (mtplog carries a caveat for it)
        self.steps = 0
        self.win = {}  # bw -> [n, {phase: sum}, {phase: [samples]}, t_last, {count: sum}]
        self.tot = {}  # bw -> [n, {phase: sum}]

    def add(self, bw, times, at=None, counts=None):
        bw = int(bw)
        tn = self.tot.setdefault(bw, [0, {p: 0.0 for p in self.phases}])
        w = self.win.setdefault(bw, [0, {p: 0.0 for p in self.phases}, {p: [] for p in self.phases}, None,
                                     {c: 0 for c in TIMING_COUNTS}])
        for n_sums in (tn, w):
            n_sums[0] += 1
            for p in self.phases:
                n_sums[1][p] += float(times.get(p, 0.0))
        for p in self.phases:
            w[2][p].append(float(times.get(p, 0.0)))
        if at is not None:
            w[3] = float(at)
        for c in TIMING_COUNTS:
            w[4][c] += int((counts or {}).get(c, 0))
        self.steps += 1
        return self.every > 0 and self.steps % self.every == 0

    def lines(self, fallbacks=None, flush="periodic"):
        out = []
        for bw in sorted(self.win):
            n, sums, samples, t_last, counts = self.win[bw]
            tn, tsums = self.tot[bw]
            win = " ".join(f"{p}={sums[p] / max(1, n):.3f}" for p in self.phases)
            run = " ".join(f"{p}={tsums[p] / max(1, tn):.3f}" for p in self.phases)
            meta = (f"t={t_last:.3f} " if t_last is not None else "") + f"flush={flush} " + " ".join(
                f"{c}={counts[c]}" for c in TIMING_COUNTS)
            smp = ";".join(f"{p}=" + ",".join(f"{x:.2f}" for x in samples[p]) for p in self.phases)
            out.append(
                f"QWEN36_MTP_BATCH_TIMING step={self.steps} bw={bw} window_n={n}"
                + (f" mode={self.mode}" if self.mode else "")
                + f" mean_ms: {win} | "
                f"running_n={tn} mean_ms: {run} | {meta} | samples_ms: {smp}"
                + (f" fallbacks={fallbacks}" if fallbacks is not None else "")
            )
        self.win = {}
        return out


def step_counts(plan, acc):
    """One speculative step's row counts for PhaseTimer.add: active rows, tokens emitted (m_b + 1 per
    row, BEFORE the scheduler trims at a stop or max_tokens), accepted and usable drafts."""
    rows = [b for b in range(plan.bw) if plan.active[b]]
    return {"rows": len(rows), "emitted": sum(len(acc.emitted[b]) for b in rows),
            "accepted": sum(int(acc.m[b]) for b in rows), "drafted": sum(int(plan.n_draft[b]) for b in rows)}


def trace_filter():
    return _tt_public_getenv(ENV_TRACE, "") or None


def trace_match(filt, slot, req_id):
    """Does row `slot` (serving request `req_id`, or None when the runner gave no ids) match?"""
    if not filt:
        return False
    if filt == "*":
        return True
    if filt.startswith("slot:"):
        return str(slot) in filt[5:].split(",")
    return req_id is not None and filt in str(req_id)


ENV_K_BY_BUCKET = "QWEN36_MTP_BATCH_K_BY_BUCKET"


def parse_k_by_bucket(value, kcap):
    """QWEN36_MTP_BATCH_K_BY_BUCKET="8:2,4:3,2:3,1:3" -> {8: 2, 4: 3, 2: 3, 1: 3}; unset/""/"0" -> {}.

    Per decode-bucket draft depth. Every entry must hold 1 <= K_b <= kcap (the global QWEN36_MTP_BATCH_K, which
    also sizes the scheduler's lookahead pages, so a deeper bucket would write past its allocation) and the verify
    row cap Bw*(K_b+1) <= 32. Anything else refuses: a typo must not silently run the global K."""
    v = "" if value is None else str(value).strip()
    if v in ("", "0"):
        return {}
    out = {}
    for tok in v.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"{ENV_K_BY_BUCKET}={value!r}: entry {tok!r} is not <bucket>:<K>")
        b_s, k_s = tok.split(":", 1)
        b, k = int(b_s), int(k_s)
        if b < 1 or b & (b - 1):
            raise ValueError(f"{ENV_K_BY_BUCKET}={value!r}: bucket {b} is not a power of two")
        if b in out:
            raise ValueError(f"{ENV_K_BY_BUCKET}={value!r}: bucket {b} given twice")
        if not 1 <= k <= int(kcap):
            raise ValueError(f"{ENV_K_BY_BUCKET}={value!r}: bucket {b} K={k} outside [1, global K={kcap}]")
        if b * (k + 1) > TILE_ROWS:
            raise ValueError(f"{ENV_K_BY_BUCKET}={value!r}: bucket {b} K={k} needs {b * (k + 1)} verify rows > {TILE_ROWS}")
        out[b] = k
    return out


def k_by_bucket(kcap):
    return parse_k_by_bucket(_tt_public_getenv(ENV_K_BY_BUCKET), kcap)


def served_verify_cells(max_batch):
    """[(T, Bw)] of every speculating bucket this environment serves (T = K_b + 1, the bucket's verify depth), the
    cells a verify-path boot self-test must cover; [] when QWEN36_MTP_BATCH is off. Same rule as
    Qwen36MTPBatch.__init__ / compile: buckets(max_batch), k_cap(), k_by_bucket(), k_for_bucket()."""
    if not enabled():
        return []
    kcap = k_cap()
    kmap = k_by_bucket(kcap)
    out = set()
    for bw in buckets(int(max_batch)):
        k = k_for_bucket(bw, kcap, kmap)
        if k >= 1:
            out.add((k + 1, int(bw)))
    return sorted(out)


def k_by_bucket_banner(kmap, kcap):
    """B-independent banner text: the parsed map in the env's order of widths, not the buckets this boot has
    (a --seqs 8 4 sweep boots once per B; the B=4 boot has no bucket 8)."""
    return ",".join(f"{b}:{kmap[b]}" for b in sorted(kmap, reverse=True)) + f" (global cap {kcap})"


def k_for_bucket(bw, kcap, kmap=None):
    """Draft depth at bucket width bw: verify runs Bw*(K+1) rows through the decode stack, whose
    configs hold M <= 32. 0 means the bucket cannot speculate (plain decode). kmap (QWEN36_MTP_BATCH_K_BY_BUCKET,
    already validated) overrides the global cap for the widths it names."""
    if kmap and int(bw) in kmap:
        return int(kmap[int(bw)])
    return max(0, min(kcap, TILE_ROWS // bw - 1))


# ---------------------------------------------------------------------------------------------- warm tail
# QWEN36_MTP_WARM_TAIL=N (docs/MTP-BATCH.md section 13): at prefill, run the MTP layer over the last N prompt
# positions (KV-only) so the drafter starts with context instead of an empty cache (risk R3).
#
# Entry convention (the decode reseed's, section 2d): the MTP KV entry for absolute position s is
# (h_s, tok_{s+1}) with RoPE at s, h_s = the base model's final-norm output at s. Over the window
# [w0, L) of an L-token prompt that is (h_s, prompt[s+1]) for s < L-1 and (h_{L-1}, first) for s = L-1, where
# `first` is the greedy token after the prompt (the argmax of the prefill logits, which is the first decode
# step's t0). The MTP KV is indexed relative to the slot's origin (pos - origin), so the warm window is written
# at indices 0..n-1 with origin = w0: one fill at relative index 0, block-aligned by construction, and the first
# decode step (p = L) continues at index n with no gap. RoPE stays absolute (w0 + i), as at decode.
ENV_WARM = "QWEN36_MTP_WARM_TAIL"
ENV_WARM_MUSTFAIL = "QWEN36_MTP_WARM_TAIL_MUSTFAIL"
WARM_DEFAULT = 2048
WARM_BUCKETS = (128, 256, 512, 1024, 2048)  # the masked prefill buckets; 128 is the floor (k_chunk trap)
WARM_MAX = WARM_BUCKETS[-1]
WARM_MUSTFAIL_MODES = ("pair_minus1",)


def warm_tail():
    """N rows of prompt tail to warm (0 = off). "1" means the default 2048 (the chunk size: the window then never
    spans more than the last full chunk and the tail)."""
    v = _tt_public_getenv(ENV_WARM, "").strip()
    if v in ("", "0"):
        return 0
    n = int(v)
    if n == 1:
        return WARM_DEFAULT
    if not 1 <= n <= WARM_MAX:
        raise ValueError(f"{ENV_WARM}={v}: must be 0 (off), 1 (= {WARM_DEFAULT}) or in [2, {WARM_MAX}]")
    return n


def warm_mustfail():
    v = _tt_public_getenv(ENV_WARM_MUSTFAIL, "").strip()
    if not v:
        return None
    if v not in WARM_MUSTFAIL_MODES:
        raise ValueError(f"{ENV_WARM_MUSTFAIL}={v!r}: one of {WARM_MUSTFAIL_MODES}")
    return v


def warm_bucket(n):
    """Device row count of the warm pass for n real rows: the smallest masked bucket >= n."""
    for b in WARM_BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"warm rows {n} > {WARM_MAX}")


@dataclass
class WarmWindow:
    L: int  # prompt length (the first decode step runs at position L)
    n: int  # real warm rows = min(N, L)
    w0: int  # first warmed absolute position = the slot's MTP origin
    Wn: int  # device rows (warm_bucket(N), fixed per boot: one program set)


def warm_window(L, N):
    """The warm window of an L-token prompt for QWEN36_MTP_WARM_TAIL=N: positions [L-n, L), n = min(N, L).
    Wn is sized from N (not n), so every request runs the same shapes and no program compiles at request time."""
    L, N = int(L), int(N)
    if L < 1 or not 1 <= N <= WARM_MAX:
        raise ValueError(f"warm window needs L >= 1 and 1 <= N <= {WARM_MAX} (L={L}, N={N})")
    n = min(N, L)
    return WarmWindow(L=L, n=n, w0=L - n, Wn=warm_bucket(N))


def warm_tokens(win, prompt_ids, first_tok, mustfail_mode=None):
    """MTP input token per warm row i (absolute position w0+i), [Wn] with 0 padding:
    prompt[w0+i+1] for i < n-1 and first_tok for i = n-1 (the decode reseed's (h_s, tok_{s+1}) pairing).
    mustfail 'pair_minus1' pairs h_s with tok_s instead (one position early): the drafter context is then wrong
    and acceptance must DROP, while the emitted stream (greedy-verified) stays exact."""
    assert len(prompt_ids) == win.L, (len(prompt_ids), win.L)
    toks = [0] * win.Wn
    for i in range(win.n):
        s = win.w0 + i
        if mustfail_mode == "pair_minus1":
            toks[i] = int(prompt_ids[s])
        else:
            toks[i] = int(prompt_ids[s + 1]) if s + 1 < win.L else int(first_tok)
    return toks


def warm_rope_positions(win):
    """Absolute RoPE position per warm row: w0+i for real rows; pad rows repeat the last real position (their K/V
    land in the scratch block or past the frontier and are never attended, but the table slice must exist)."""
    last = win.w0 + win.n - 1
    return [win.w0 + i if i < win.n else last for i in range(win.Wn)]


def warm_gather_index(win, seg_start, seg_valid, zero_row):
    """Row gather of one prefill segment into the warm window. The segment's hidden rows r = 0..seg_valid-1 hold
    absolute positions seg_start + r. Returns (idx [Wn], covered): idx[i] = the segment row for window row i when
    position w0+i is in this segment, else `zero_row` (an all-zero row appended to the segment), so per-segment
    gathers ADD into the window exactly (x + 0). None when the segment and the window do not meet."""
    lo = max(int(seg_start), win.w0)
    hi = min(int(seg_start) + int(seg_valid), win.w0 + win.n)
    if lo >= hi:
        return None, 0
    idx = [int(zero_row)] * win.Wn
    for pos in range(lo, hi):
        idx[pos - win.w0] = pos - int(seg_start)
    return idx, hi - lo


# ------------------------------------------------------------------------------------------ warm tail span
# QWEN36_MTP_WARM_TAIL_SPAN=k (needs QWEN36_MTP_WARM_TAIL=N, N % 64 == 0): warm the last min(k*N, L) prompt positions
# instead of min(N, L), as up to k KV-only passes of N rows each (the SAME Wn = warm_bucket(N) programs as today, so
# nothing new compiles and no trace is added). Every chunk replay whose rows meet the window is gathered right after
# its replay (the chunk trace output holds one chunk only) into one of k persistent [Wn, D] window buffers.
# Sub-window j covers relative indices [j*N, min((j+1)*N, n)) of the slot (origin w0 = L - n), so every pass fills
# from a block-aligned relative start; the KV-only chain is row-local (norms, matmuls, RoPE, cast, fill), so how the
# rows are grouped into passes does not change any cache entry's inputs. Off (unset, 0 or 1) = today's single window.
ENV_WARM_SPAN = "QWEN36_MTP_WARM_TAIL_SPAN"
ENV_WARM_SPAN_MUSTFAIL = "QWEN36_MTP_WARM_TAIL_SPAN_MUSTFAIL"
WARM_SPAN_MAX = 16  # 16 x 2048 = a whole 32k prompt


def warm_span():
    """k window passes (0 = off). "", "0" and "1" are off: k=1 IS today's window."""
    v = _tt_public_getenv(ENV_WARM_SPAN, "").strip()
    if v in ("", "0", "1"):
        return 0
    k = int(v)
    if not 2 <= k <= WARM_SPAN_MAX:
        raise ValueError(f"{ENV_WARM_SPAN}={v}: must be 0/1 (off) or in [2, {WARM_SPAN_MAX}]")
    return k


def warm_span_mustfail():
    """QWEN36_MTP_WARM_TAIL_SPAN_MUSTFAIL=1: every span gather reads the hidden row one position late; the warmup
    self-test must REFUSE the boot."""
    v = _tt_public_getenv(ENV_WARM_SPAN_MUSTFAIL, "").strip()
    if v in ("", "0"):
        return False
    if v != "1":
        raise ValueError(f"{ENV_WARM_SPAN_MUSTFAIL}={v!r}: 1 or unset")
    return True


ENV_WARM_SPAN_EARLY = "QWEN36_MTP_WARM_TAIL_SPAN_EARLY"
ENV_WARM_SPAN_EARLY_MUSTFAIL = "QWEN36_MTP_WARM_TAIL_SPAN_EARLY_MUSTFAIL"


def warm_span_early():
    """QWEN36_MTP_WARM_TAIL_SPAN_EARLY=1: a span pass whose window is complete, and whose tokens are all prompt tokens
    (every pass but the last), runs its KV pass right after the gather, under the chunk loop (same CQ, between two
    non-blocking chunk replays), instead of serially after the prefill logits readback. The last pass pairs its last
    row with the prefill argmax, so it still waits for the logits. Same ops, same inputs, same per-pass order."""
    v = _tt_public_getenv(ENV_WARM_SPAN_EARLY, "").strip()
    if v in ("", "0"):
        return False
    if v != "1":
        raise ValueError(f"{ENV_WARM_SPAN_EARLY}={v!r}: 1 or unset")
    return True


def warm_span_early_mustfail():
    """QWEN36_MTP_WARM_TAIL_SPAN_EARLY_MUSTFAIL=1: the early passes pair h_s with tok_s (one position early, the
    'pair_minus1' rows) while the late path keeps the served pairing; the warmup self-test, which compares the KV
    block an early pass writes with the KV block the late path writes, must REFUSE the boot."""
    v = _tt_public_getenv(ENV_WARM_SPAN_EARLY_MUSTFAIL, "").strip()
    if v in ("", "0"):
        return False
    if v != "1":
        raise ValueError(f"{ENV_WARM_SPAN_EARLY_MUSTFAIL}={v!r}: 1 or unset")
    return True


def warm_span_early_ready(sp, covered, j):
    """May pass j run now (before the prefill logits)? Only if it is not the last pass (its tokens are then all prompt
    tokens: position w0+j*N+N <= L-1 is inside pass j+1) and passes 0..j are fully covered, in order: then the origin
    is w0 (warm_span_covered_from == 0 once the rest is covered), so relative index j*N is final."""
    j = int(j)
    if j >= sp.passes - 1:
        return False
    return all(int(covered[i]) == sp.sub(i)[1] for i in range(j + 1))


def warm_span_check(N, k, block_size=64):
    """Refusals for a span boot: N must be block-aligned (each pass fills from relative index j*N)."""
    if k and int(N) % int(block_size):
        raise ValueError(f"{ENV_WARM_SPAN}={k} needs {ENV_WARM} % {block_size} == 0 (got N={N}): each pass fills "
                         f"from relative index j*N, which must start a KV block")


@dataclass
class WarmSpan:
    L: int  # prompt length
    n: int  # warmed rows = min(k*N, L)
    w0: int  # first warmed absolute position = the slot's MTP origin
    N: int  # rows per pass
    Wn: int  # device rows per pass (warm_bucket(N))
    passes: int  # ceil(n / N) <= k

    def sub(self, j):
        """Sub-window j: (absolute start, real rows). Relative start = j*N."""
        a = j * self.N
        return self.w0 + a, min(self.N, self.n - a)

    def sub_window(self, j):
        """Sub-window j as a WarmWindow (warm_tokens / warm_rope_positions take it unchanged)."""
        s, rows = self.sub(j)
        return WarmWindow(L=self.L, n=rows, w0=s, Wn=self.Wn)


def warm_span_window(L, N, k):
    L, N, k = int(L), int(N), int(k)
    if L < 1 or not 1 <= N <= WARM_MAX or not 1 <= k <= WARM_SPAN_MAX:
        raise ValueError(f"warm span needs L >= 1, 1 <= N <= {WARM_MAX}, 1 <= k <= {WARM_SPAN_MAX} (L={L}, N={N}, k={k})")
    n = min(k * N, L)
    return WarmSpan(L=L, n=n, w0=L - n, N=N, Wn=warm_bucket(N), passes=-(-n // N))


def warm_span_gather_index(sp, j, seg_start, seg_valid, zero_row, shift=False):
    """Row gather of one prefill segment into sub-window j: (idx [Wn], covered) or (None, 0) when they do not meet.
    shift (the must-fail): read segment row r+1 for position seg_start+r (one position late) where it exists."""
    s, rows = sp.sub(j)
    lo = max(int(seg_start), s)
    hi = min(int(seg_start) + int(seg_valid), s + rows)
    if lo >= hi:
        return None, 0
    idx = [int(zero_row)] * sp.Wn
    for pos in range(lo, hi):
        r = pos - int(seg_start)
        if shift and r + 1 < int(seg_valid):
            r += 1
        idx[pos - s] = r
    return idx, hi - lo


def warm_span_covered_from(sp, covered):
    """First pass j* such that every pass j >= j* is fully covered (covered[j] == its rows); sp.passes if none.
    A request whose early passes were not covered (a resumed prefix, an eager chunk path) warms the covered suffix
    with the origin moved to pass j*'s start (still block-aligned)."""
    j_star = sp.passes
    for j in range(sp.passes - 1, -1, -1):
        if covered[j] != sp.sub(j)[1]:
            break
        j_star = j
    return j_star



def shifted_page_table(pt, shifts):
    """QWEN36_PREFIX_CACHING: the MTP head's page table [bw, nblk]: row b = pt[b, shifts[b]:] followed by shifts[b]
    zero (null-block) entries. Relative head position r of row b then lands in pt[b, r // block + shifts[b]], a block
    the request owns (its prefix-hit blocks are the first shifts[b]); the head's positions stay below
    alloc_capacity - shifts[b] * block because its origin is at or past the hit (warm windows and re-origins never
    start inside the cached prefix). All-zero shifts return pt itself (the unshifted table)."""
    import torch as _t

    if not any(int(x) for x in shifts):
        return pt
    bw, n = int(pt.shape[0]), int(pt.shape[-1])
    out = _t.zeros_like(pt)
    for b in range(bw):
        sh = int(shifts[b]) if b < len(shifts) else 0
        if sh < 0 or sh > n:
            raise ValueError(f"shifted_page_table: row {b} shift {sh} outside [0, {n}]")
        out[b, : n - sh] = pt[b, sh:]
    return out


def prefix_warm_window(total, hit, N, k, block_size):
    """QWEN36_PREFIX_CACHING + QWEN36_MTP_WARM_TAIL_SPAN: the warm span of a request whose prompt has `total` tokens
    and whose prefix-cache hit is `hit` (block-aligned; 0 = cold), with passes ANCHORED on the N grid instead of on
    the prompt end, so that vLLM's align-mode split of a long prompt (a head step ending at a block boundary, then the
    tail) completes every pass inside one step: the last pass holds [A, total) with A = ((total - 1) // N) * N, the
    earlier ones are whole N-blocks ending at A, at most k passes, none starting before `hit`.
    Returns (w0, passes): passes = [(start, rows)], w0 = passes[0][0] = the slot's MTP origin (block-aligned when
    N % block_size == 0). Pure host arithmetic (tests/test_prefix_s12.py)."""
    total, hit, N, k = int(total), int(hit), int(N), int(k)
    if total < 1 or hit < 0 or hit >= total or N < 1 or k < 1:
        raise ValueError(f"prefix_warm_window: total={total} hit={hit} N={N} k={k}")
    if N % int(block_size) or hit % N:
        raise ValueError(f"prefix_warm_window: N={N} must be a multiple of block {block_size} and hit={hit} of N")
    A = ((total - 1) // N) * N
    w0 = max(hit, A - (k - 1) * N, 0)
    passes = []
    s = w0
    while s < total:
        passes.append((s, min(N, total - s)))
        s += N
    return w0, passes


def buckets(max_batch):
    """Speculating bucket widths: QWEN36_MTP_BATCH_BUCKETS (default "4,8"), restricted to the plain
    decode bucket set (powers of two below max_batch, plus max_batch) and to K >= 1."""
    raw = _tt_public_getenv(ENV_BUCKETS, "4,8")
    plain = set()
    w = 1
    while w < max_batch:
        plain.add(w)
        w *= 2
    plain.add(max_batch)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        b = int(tok)
        if b in plain and k_for_bucket(b, MAX_K) >= 1 and b not in out:
            out.append(b)
    return sorted(out)


def bucket_for(num_active, width, slot_remap_set):
    """The plain decode bucket rule (qwen36_vllm.decode_forward): smallest power of two >= the
    active prefix, full width when a slot_remap is set this step."""
    num_active = max(1, min(int(num_active), int(width)))
    if slot_remap_set:
        return int(width)
    return min(int(width), 1 << max(0, (num_active - 1).bit_length()))


# ----------------------------------------------------------------- QWEN36_MTP_LOOKUP (step 1)
# Prompt-lookup draft source swap. Per row, when the last LOOKUP_NGRAM ids of the request's history (prompt + every
# committed id) occurred earlier and that earlier occurrence matches the current suffix for at least
# QWEN36_MTP_LOOKUP_MIN ids, the step verifies the ids that followed it instead of the MTP drafts. K, the verify
# width, the MTP drafter and its KV are untouched: the MTP head still drafts every step (its drafts are simply not
# verified on a lookup step) and the reseed writes the verify rows exactly as before. Acceptance stays greedy exact
# (accept() is unchanged), so the output is the MTP-only output by construction; only which drafts get verified moves.
# Read per plan() call (switch class runtime): the env decides on every step.
ENV_LOOKUP = "QWEN36_MTP_LOOKUP"
ENV_LOOKUP_MIN = "QWEN36_MTP_LOOKUP_MIN"
ENV_LOOKUP_MUSTFAIL = "QWEN36_MTP_LOOKUP_MUSTFAIL"
LOOKUP_MIN_DEFAULT = 12  # the offline gate (LC0 captures): flat for 6..16, 12 never lost on any doc type
LOOKUP_NGRAM = 3
LOOKUP_MATCH_CAP = 64
# shift: propose the ids one past the matched continuation (acceptance of lookup steps must collapse; the output stays
#        exact, which is what proves the swap cannot change tokens). trust: accept every lookup draft WITHOUT the
#        greedy compare (the output must diverge, so an equivalence check against MTP-only must fail).
LOOKUP_MUSTFAIL_MODES = ("shift", "trust")
_LK_BASE = 1 << 18  # > every Qwen3.6/3.8 id (vocab 248320): the 3-gram key packs into one int


def lookup_cfg():
    """(on, min_match, mustfail_mode) from the env, validated (fails closed on a bad value)."""
    on = _tt_public_getenv(ENV_LOOKUP, "0").strip() == "1"
    raw = _tt_public_getenv(ENV_LOOKUP_MIN, "").strip()
    try:
        mn = int(raw) if raw else LOOKUP_MIN_DEFAULT
    except ValueError:
        raise ValueError(f"{ENV_LOOKUP_MIN}={raw!r}: expected an integer") from None
    if not LOOKUP_NGRAM <= mn <= LOOKUP_MATCH_CAP:
        raise ValueError(f"{ENV_LOOKUP_MIN}={mn}: must be in [{LOOKUP_NGRAM}, {LOOKUP_MATCH_CAP}]")
    mf = _tt_public_getenv(ENV_LOOKUP_MUSTFAIL, "").strip() or None
    if mf is not None and mf not in LOOKUP_MUSTFAIL_MODES:
        raise ValueError(f"{ENV_LOOKUP_MUSTFAIL}={mf!r}: expected one of {LOOKUP_MUSTFAIL_MODES}")
    return on, mn, (mf if on else None)


def lookup_banner(cfg):
    on, mn, mf = cfg
    if not on:
        return None
    return (f"QWEN36_MTP_LOOKUP engaged: prompt-lookup draft source swap, min_match={mn} ngram={LOOKUP_NGRAM} "
            f"(K and verify width unchanged, greedy exact acceptance)") + (
        f" | QWEN36_MTP_LOOKUP_MUSTFAIL engaged: {mf}" if mf else "")


class LookupRow:
    """One request's id history (absolute position i = hist[i]) and a 3-gram index over it. idx maps the packed
    3-gram hist[i-3:i] to the LATEST continuation index i inserted so far; indices are inserted lazily up to (not
    including) the current end, so a lookup never finds the suffix itself."""

    __slots__ = ("hist", "idx", "upto")

    def __init__(self, prompt_ids):
        self.hist = [int(t) for t in prompt_ids]
        self.idx = {}
        self.upto = LOOKUP_NGRAM

    def _key(self, i):
        h = self.hist
        return (h[i - 3] * _LK_BASE + h[i - 2]) * _LK_BASE + h[i - 1]

    def index(self):
        """Insert every continuation index below the current end (the prompt's ~L inserts run once, at seed time,
        so the first decode step does not pay them)."""
        end = len(self.hist)
        for i in range(self.upto, end):
            self.idx[self._key(i)] = i
        self.upto = max(self.upto, end)

    def propose(self, K, min_match, shift=0):
        """Up to K ids following the latest earlier occurrence of the current 3-gram suffix, if that occurrence
        matches the suffix for >= min_match ids; else None. Returns (ids, match_len)."""
        end = len(self.hist)
        if end <= LOOKUP_NGRAM:
            return None
        self.index()
        j = self.idx.get(self._key(end))
        if j is None or j >= end:
            return None
        h = self.hist
        n = 0
        while n < LOOKUP_MATCH_CAP and j - 1 - n >= 0 and h[j - 1 - n] == h[end - 1 - n]:
            n += 1
        if n < min_match:
            return None
        ids = h[j + shift: min(j + shift + K, end)]
        return (ids, n) if ids else None


@dataclass
class SlotState:
    origin: int  # absolute position of this request's first speculative step (MTP KV index 0)
    next_p: int  # the position the next step must start at for the drafts to be usable
    t0: int  # the token the next step must start with
    drafts: list = field(default_factory=list)
    warm: bool = False  # seeded by QWEN36_MTP_WARM_TAIL, not yet continued by a speculative step
    # QWEN36_MTP_SPEC_SAMPLE: the drafter's candidate block from the last chain (the probs and the
    # ids), [K][C2] per depth: the rule needs q at the target's ids, and the drafter's block is
    # computed in the C trace (the chain) and consumed by the next step's verify.
    qprobs: list = field(default_factory=list)
    qids: list = field(default_factory=list)


@dataclass
class Plan:
    bw: int
    K: int
    T: int
    active: list  # [bw] bool
    p: list  # [bw] base position (or -1)
    t0: list  # [bw] token at position p
    n_draft: list  # [bw] usable drafts (0..K); acceptance never exceeds it
    drafts: list  # [bw][K] draft ids (dummies where unusable)
    origin: list  # [bw] MTP origin (or 0)
    verify_tokens: list  # [T*bw] j-major
    verify_pos: list  # [T][bw] KV position per verify depth (-1 = skip update and SDPA)
    verify_rope_pos: list  # [T*bw] j-major absolute position for RoPE (-1 rows are clamped on device)
    kv_capacity: list  # [bw] addressable KV positions per row (page-table width * block_size)
    source: list = field(default_factory=list)  # [bw] "mtp" | "lookup" (QWEN36_MTP_LOOKUP); empty = all mtp
    lookup_mustfail: str | None = None
    # QWEN36_MTP_SPEC_SAMPLE (None with the flag off): the drafter's distribution q behind each draft,
    # [K*bw, C2] torch fp32 probs and FULL-vocab ids, depth-major (row k*bw+b = draft k of row b), matching
    # the verify's first K*bw rows. A draft the chain did not sample (a lookup draft) gets a point mass.
    qprobs: object = None
    qids: object = None


@dataclass
class Accept:
    m: list  # [bw] accepted drafts per row
    emitted: list  # [bw] list of emitted token ids per row (empty for inactive rows)
    gdn_onehot: list  # [T][bw] 1.0 where m_b == j (GDN recurrent-state commit select)
    anchor_idx: list  # [bw] verify row whose hidden seeds draft step 0 (m_b*bw + b)
    t0_next: list  # [bw] ids[b, m_b]: the token after the anchor
    reseed_pos: list  # [T][bw] MTP KV index for reseed depth j (-1 = skip)
    reseed_rope_pos: list  # [T*bw] j-major absolute RoPE position
    draft_pos: list  # [K][bw] MTP KV index for draft step k (-1 = skip)
    draft_rope_pos: list  # [K][bw] absolute RoPE position for draft step k


class MTPBatchHost:
    """Per-slot speculative state for one mesh. Slots are the decode state slots (= decode rows
    after the plugin's slot_remap), the same index space as the GDN recurrent state."""

    def __init__(self, max_batch, kcap, block_size=64, reject_all=None, trace=None, mustfail_mode=None):
        self.mustfail = mustfail() if mustfail_mode is None else (mustfail_mode or None)
        self.max_batch = int(max_batch)
        self.kcap = int(kcap)
        self.block_size = int(block_size)
        self.slots = [None] * self.max_batch
        # QWEN36_PREFIX_CACHING (S12 prefix arm): per slot, how many leading page-table blocks the MTP head skips
        # (= the request's prefix-cache hit // block_size). The head addresses its own KV at positions RELATIVE to
        # its origin through the request's page table; with a prefix hit the leading blocks are SHARED cached blocks,
        # and sibling requests would overwrite each other's drafter KV there. Shifting the head's table past the hit
        # keeps every drafter write in the request's own blocks (shifted_page_table). 0 = unshifted (every
        # non-prefix arm: the head's table is the verify table itself).
        self.pt_shift = [0] * self.max_batch
        # QWEN36_MTP_OVERLAP: (plan, acc) of the step whose drafts are still on the device. Every method that
        # reads or moves slot state refuses while it is set, so a missed drain fails loudly instead of
        # silently re-origining rows (which would only cost acceptance, and so would never show in a gate).
        self.pending = None
        self.reject_all = force_reject() if reject_all is None else bool(reject_all)
        self.trace = trace_filter() if trace is None else (trace or None)
        self.row_ids = [None] * self.max_batch  # decode row -> request id (runner hook, trace only)
        self.stats = {
            "would_accept": 0,  # acceptance the drafts WOULD have had (differs from accepted under FORCE_REJECT)
            # QWEN36_MTP_SPEC_SAMPLE: the acceptance PER RULE (the exact rule vs sample-then-compare),
            # so one boot can A/B the two rules: the drafted and the accepted counts of each.
            "exact_drafted": 0,
            "exact_accepted": 0,
            "compare_drafted": 0,
            "compare_accepted": 0,
            "steps": 0,
            "rows": 0,
            "drafted": 0,
            "accepted": 0,
            "emitted": 0,
            "fresh": 0,
            "stale": 0,
            "capped": 0,
            "warm": 0,  # slots seeded by QWEN36_MTP_WARM_TAIL (origin = first warmed position)
            "warm_kept": 0,  # first speculative steps that continued a warm origin (else it re-origined: stale)
        }
        # QWEN36_MTP_LOOKUP: per-slot request history (None = no history: not seeded, or it broke); counters are
        # printed on the stats line only once a lookup step ran, so CONTROL's line is unchanged.
        self.lookup = [None] * self.max_batch
        # QWEN36_MTP_SPEC_SAMPLE: the drafter's candidate width C2 (= 2 * the per-shard K) while the
        # served rule is on, else None (then plan() builds no q blocks and accept() takes no rule data).
        self.spec_c2 = None
        # QWEN36_MTP_SPEC_SAMPLE_DUAL: exact rows whose drafts had no q (came from a greedy-family chain) and were
        # planned without drafts (a guard: a T > 0 row never runs a greedy step)
        self.dual_qdrop = 0
        self.lstats = {"lookup_steps": 0, "lookup_drafted": 0, "lookup_accepted": 0, "lookup_seeded": 0,
                       "lookup_gap": 0}
        self.depth_try = [0] * MAX_K
        self.depth_hit = [0] * MAX_K
        self.mhist = [0] * (MAX_K + 1)
        self.pos_rows, self.pos_drafted, self.pos_accepted = [], [], []
        self.capture = {} if capture_on() else None  # slot -> {"p0", "ids", "m", "nd", "gaps", "trunc"}

    # ----------------------------------------------------------------- deferred finish (QWEN36_MTP_OVERLAP)
    def _settled(self, what):
        if self.pending is not None:
            raise RuntimeError(f"QWEN36_MTP_OVERLAP: {what} while a speculative step is unfinished (drain first)")

    def defer(self, plan, acc):
        """The step's tokens are final (accept ran); its drafts are still on the device."""
        self._settled("defer")
        self.pending = (plan, acc)

    def complete(self, draft_ids, blocks=None):
        """Finish the deferred step with its drafts (finish()); returns (plan, acc), or None if none."""
        if self.pending is None:
            return None
        plan, acc = self.pending
        self.pending = None
        self.finish(plan, acc, draft_ids, blocks)
        return plan, acc

    # ----------------------------------------------------------------- slot lifecycle
    def reset_slot(self, s):
        self._settled("reset_slot")
        if 0 <= int(s) < self.max_batch:
            self.slots[int(s)] = None
            self.lookup[int(s)] = None
            self.pt_shift[int(s)] = 0
        if self.capture is not None:
            self.capture.pop(int(s), None)  # pop_capture() logs it first on the release paths

    def set_pt_shift(self, s, blocks):
        """QWEN36_PREFIX_CACHING: slot s's MTP head skips its first `blocks` page-table blocks (its prefix hit)."""
        self._settled("set_pt_shift")
        s, blocks = int(s), int(blocks)
        if not 0 <= s < self.max_batch or blocks < 0:
            raise ValueError(f"set_pt_shift: slot {s} / blocks {blocks} out of range")
        self.pt_shift[s] = blocks

    def reset_all(self):
        self._settled("reset_all")
        self.slots = [None] * self.max_batch
        self.lookup = [None] * self.max_batch
        self.pt_shift = [0] * self.max_batch
        if self.capture is not None:
            self.capture = {}

    @staticmethod
    def lookup_enabled():
        return lookup_cfg()[0]

    def lookup_seed(self, s, prompt_ids):
        """QWEN36_MTP_LOOKUP: slot s now serves a request whose prompt is prompt_ids (positions 0..len-1). The
        first decode step's t0 sits at position len(prompt_ids). No-op when the flag is off."""
        self._settled("lookup_seed")
        s = int(s)
        if not 0 <= s < self.max_batch:
            raise ValueError(f"lookup_seed: slot {s} outside [0, {self.max_batch})")
        if not lookup_cfg()[0]:
            self.lookup[s] = None
            return
        self.lookup[s] = LookupRow(prompt_ids)
        self.lookup[s].index()
        self.lstats["lookup_seeded"] += 1

    def warm_slot(self, s, origin, next_p, t0):
        """QWEN36_MTP_WARM_TAIL: slot s's MTP KV holds the prompt window [origin, next_p) at indices 0..n-1.
        The first decode step continues it only if it starts at next_p with token t0 (plan()'s usual check);
        a host-sampled or logits-processed first token that differs re-origins it like any stale slot."""
        self._settled("warm_slot")
        s, origin, next_p = int(s), int(origin), int(next_p)
        if not 0 <= s < self.max_batch:
            raise ValueError(f"warm_slot: slot {s} outside [0, {self.max_batch})")
        if not 0 <= origin < next_p:
            raise ValueError(f"warm_slot: origin {origin} must be in [0, next_p={next_p})")
        self.slots[s] = SlotState(origin=origin, next_p=next_p, t0=int(t0), drafts=[], warm=True)
        self.stats["warm"] += 1

    def idle(self):
        return all(s is None for s in self.slots)

    def set_row_ids(self, req_ids):
        """Decode row i serves req_ids[i] this step (rows past the list are padding)."""
        ids = list(req_ids or [])[: self.max_batch]
        self.row_ids = ids + [None] * (self.max_batch - len(ids))

    def traced_rows(self, bw):
        return [b for b in range(min(bw, self.max_batch)) if trace_match(self.trace, b, self.row_ids[b])]

    def remap(self, remap):
        """Slot i takes the state previously at slot remap[i] (plugin condense), like
        TPGatedDeltaNet.remap_slots does for the GDN state."""
        self._settled("remap")
        idx = [int(remap[i]) for i in range(self.max_batch)]
        if all(idx[i] == i for i in range(self.max_batch)):
            return
        old = list(self.slots)
        self.slots = [old[j] if 0 <= j < self.max_batch else None for j in idx]
        oldl = list(self.lookup)
        self.lookup = [oldl[j] if 0 <= j < self.max_batch else None for j in idx]
        olds = list(self.pt_shift)
        self.pt_shift = [olds[j] if 0 <= j < self.max_batch else 0 for j in idx]
        if self.capture is not None:
            oc = self.capture
            self.capture = {i: oc[j] for i, j in enumerate(idx) if j in oc}

    # ----------------------------------------------------------------- one step
    def feed_predict(self, tokens, positions, bw, K, kv_capacity, need_q=None, spec=True, pad=False):
        """QWEN36_MTP_DEVFEED: (plan, None) when this step is a FAST step, else (None, reason). Called while the previous
        step is still pending (its drafts on the device): a fast step is one whose plan() after the drain is determined
        now except for the drafts and their q blocks, which the device feed writes. The prediction carries placeholder
        drafts ([t0] * K) and zero q blocks; feed_plan_diff checks it against the real plan after the drain.
        Conditions (each a branch of plan() that is NOT taken): the pending step ran the same bucket width and depth and
        the sampled family (its chain has q blocks), every row is active in both steps and continues exactly
        (position p + m + 1 and token t0_next: not fresh, not stale), no warm origin, no lookup, every row has room for
        K verify depths (not capped), and (sampled family) every row's drafts have their q (no dual_qdrop).
        pad (QWEN36_MTP_DEVFEED_PAD): a row padded NOW (position -1) is allowed whatever it was in the pending step; it is
        planned exactly as plan() plans a padded row (draft-independent). A row padded in the pending step and active now
        is still refused ("inactive row"). At least one row must be active."""
        if self.pending is None:
            return None, "no pending step"
        if not spec or not self.spec_c2:
            return None, "family"
        pplan, pacc = self.pending
        if pplan.bw != bw or pplan.K != K:
            return None, "bucket"
        if pplan.qprobs is None:
            return None, "family"
        lk_on, _, lk_mf = lookup_cfg()
        if lk_on:
            return None, "lookup"
        T = K + 1
        p, t0, origin, caps = [], [], [], []
        act = []
        for b in range(bw):
            pb, tb, cb = int(positions[b]), int(tokens[b]), int(kv_capacity[b])
            if pad and pb < 0:
                act.append(False)
                p.append(-1)
                t0.append(0)
                origin.append(0)
                caps.append(cb)
                continue
            if pb < 0 or not pplan.active[b]:
                return None, "inactive row"
            act.append(True)
            st = self.slots[b]
            if st is None or st.warm:
                return None, "fresh"
            if pb != int(pplan.p[b]) + int(pacc.m[b]) + 1 or tb != int(pacc.t0_next[b]):
                return None, "stale"
            if pb >= cb or cb - pb - 1 < K:
                return None, "capped"
            p.append(pb)
            t0.append(tb)
            origin.append(int(st.origin))
            caps.append(cb)
        import torch

        C2 = int(self.spec_c2)
        if not any(act):
            return None, "inactive row"
        # Active rows: rows j >= 1 are placeholders (fed); p + j < capacity for every j <= K (not capped). Padded rows:
        # plan()'s padded row (token 0, positions -1, no drafts, drafts [token] * K).
        verify_tokens = [t0[b] if act[b] else 0 for j in range(T) for b in range(bw)]
        verify_pos = [[p[b] + j if act[b] else -1 for b in range(bw)] for j in range(T)]
        verify_rope_pos = [p[b] + j if act[b] else -1 for j in range(T) for b in range(bw)]
        return Plan(bw=bw, K=K, T=T, active=list(act), p=p, t0=t0, n_draft=[K if a else 0 for a in act],
                    drafts=[[t0[b]] * K if act[b] else [int(tokens[b])] * K for b in range(bw)],
                    origin=origin, verify_tokens=verify_tokens, verify_pos=verify_pos, verify_rope_pos=verify_rope_pos,
                    kv_capacity=caps, source=["mtp"] * bw, lookup_mustfail=lk_mf,
                    qprobs=torch.zeros((K * bw, C2), dtype=torch.float32),
                    qids=torch.zeros((K * bw, C2), dtype=torch.float32)), None

    def plan(self, tokens, positions, bw, K, kv_capacity, need_q=None, spec=True):
        """Build the verify inputs for the first bw rows.

        tokens/positions: per-row last token and its position (position -1 = padded row).
        kv_capacity: per-row count of addressable KV positions (page-table width * block_size);
        verify depth j is only issued while p + j < capacity, and drafts past it are unusable.
        QWEN36_MTP_SPEC_SAMPLE_DUAL: spec=False (a greedy-family step) builds no q blocks; need_q ([bw] 0/1, dual_family)
        lets a row whose drafts have no q go with a zero block unless it needs one, and an exact row that needs one
        and has none is planned with no drafts (counted in dual_qdrop). need_q None: every chain draft needs its q
        (the single-family path, unchanged).
        """
        assert 1 <= K <= MAX_K and bw * (K + 1) <= TILE_ROWS, (bw, K)
        self._settled("plan")
        T = K + 1
        lk_on, lk_min, lk_mf = lookup_cfg()
        source = []
        active, p, t0, n_draft, drafts, origin = [], [], [], [], [], []
        for b in range(bw):
            pb = int(positions[b])
            tb = int(tokens[b])
            if pb < 0:
                active.append(False)
                p.append(-1)
                t0.append(0)
                n_draft.append(0)
                drafts.append([tb] * K)
                origin.append(0)
                source.append("mtp")
                continue
            if pb >= int(kv_capacity[b]):
                raise ValueError(f"row {b}: position {pb} is past its page table ({kv_capacity[b]} positions)")
            st = self.slots[b]
            usable = 0
            dr = [tb] * K
            if st is not None and (st.next_p != pb or st.t0 != tb):
                # The row moved on without us (a plain-decode fallback step, a preempt/resume, a
                # missed reset): the MTP KV between origin and pb has a gap or foreign entries.
                # Re-origin so the drafter only ever attends to entries this request wrote.
                self.stats["stale"] += 1
                st = None
            if st is None:
                st = SlotState(origin=pb, next_p=pb, t0=tb, drafts=[])
                self.slots[b] = st
                self.stats["fresh"] += 1
            elif st.warm:
                st.warm = False  # the warm window continues: origin stays at the first warmed position
                self.stats["warm_kept"] += 1
            if len(st.drafts) >= 1:
                usable = min(K, len(st.drafts))
                dr = [int(d) for d in st.drafts[:usable]] + [tb] * (K - usable)
            src = "mtp"
            lr = self.lookup[b] if lk_on else None
            if lr is not None:
                # History invariant: hist[i] is the id at absolute position i, through pb (= tb).
                n_h = len(lr.hist)
                if n_h == pb:
                    lr.hist.append(tb)  # the request's first decode step: t0 came from the prefill logits
                elif not (n_h == pb + 1 and lr.hist[pb] == tb):
                    self.lookup[b] = lr = None  # a step the host did not see (plain-decode fallback): stop for this request
                    self.lstats["lookup_gap"] += 1
                if lr is not None:
                    got = lr.propose(K, lk_min, shift=1 if lk_mf == "shift" else 0)
                    if got is not None:
                        ids_lk = [int(x) for x in got[0][:K]]
                        usable = len(ids_lk)
                        dr = ids_lk + [tb] * (K - usable)
                        src = "lookup"
            cap = int(kv_capacity[b]) - pb - 1  # deepest verify depth j with p + j < capacity
            if cap < usable:
                self.stats["capped"] += 1
                usable = max(0, cap)
            if (need_q is not None and spec and self.spec_c2 and need_q[b] and src == "mtp" and usable > 0
                    and len(st.qprobs) < usable):
                self.dual_qdrop += 1  # QWEN36_MTP_SPEC_SAMPLE_DUAL guard: no q for these drafts -> verify none
                usable = 0
            active.append(True)
            p.append(pb)
            t0.append(tb)
            n_draft.append(usable)
            drafts.append(dr)
            origin.append(st.origin)
            source.append(src)

        qprobs, qids = self._spec_q_blocks(bw, K, active, n_draft, drafts, source, need_q) if spec else (None, None)
        verify_tokens, verify_rope_pos = [], []
        verify_pos = []
        for j in range(T):
            row_pos = []
            for b in range(bw):
                if not active[b]:
                    verify_tokens.append(0)
                    verify_rope_pos.append(-1)
                    row_pos.append(-1)
                    continue
                verify_tokens.append(t0[b] if j == 0 else drafts[b][j - 1])
                pj = p[b] + j
                ok = pj < int(kv_capacity[b])
                verify_rope_pos.append(pj if ok else -1)
                row_pos.append(pj if ok else -1)
            verify_pos.append(row_pos)
        return Plan(
            bw=bw,
            K=K,
            T=T,
            active=active,
            p=p,
            t0=t0,
            n_draft=n_draft,
            drafts=drafts,
            origin=origin,
            verify_tokens=verify_tokens,
            verify_pos=verify_pos,
            verify_rope_pos=verify_rope_pos,
            kv_capacity=[int(kv_capacity[b]) for b in range(bw)],
            source=source,
            lookup_mustfail=lk_mf,
            qprobs=qprobs,
            qids=qids,
        )

    def _spec_q_blocks(self, bw, K, active, n_draft, drafts, source, need_q=None):
        """QWEN36_MTP_SPEC_SAMPLE: q behind each usable draft, [K*bw, C2] probs and ids (depth-major).

        The exact rule needs the distribution each draft was SAMPLED from. A chain draft has it (the
        chain's candidate block, kept by finish()). A lookup draft is deterministic, so its q is a point
        mass at the draft: the rule then accepts it with probability p(x) and a rejection draws from
        norm((p - 1[x])^+) = p without x, which is the exact rule for a deterministic drafter. A chain
        draft WITHOUT its block cannot be judged exactly (its q is unknown): that is refused, not
        guessed. Rows past n_draft and inactive rows stay zero; accept() ignores the rule there.
        need_q (QWEN36_MTP_SPEC_SAMPLE_DUAL): a row with need_q[b] == 0 whose drafts came without a block (a
        greedy-family chain) gets the zero block: its rule output is not read (not exact)."""
        if not self.spec_c2:
            return None, None
        import torch

        C2 = int(self.spec_c2)
        qp = torch.zeros((K, bw, C2), dtype=torch.float32)
        qi = torch.zeros((K, bw, C2), dtype=torch.float32)
        for b in range(bw):
            nd = int(n_draft[b])
            if not active[b] or nd == 0:
                continue
            st = self.slots[b]
            if source and source[b] == "lookup":
                for k in range(nd):
                    qp[k, b, 0] = 1.0
                    qi[k, b, 0] = float(drafts[b][k])
                continue
            if need_q is not None and not need_q[b] and (st is None or len(st.qprobs) < nd):
                continue
            if st is None or len(st.qprobs) < nd:
                raise RuntimeError(
                    f"QWEN36_MTP_SPEC_SAMPLE: row {b} has {nd} chain drafts but {0 if st is None else len(st.qprobs)} "
                    f"drafter blocks; the exact rule needs q for every draft (refusing rather than accepting blind)")
            for k in range(nd):
                qp[k, b] = st.qprobs[k]
                qi[k, b] = st.qids[k]
        return qp.reshape(K * bw, C2), qi.reshape(K * bw, C2)

    def accept(self, plan, ids, spec=None):
        """Greedy exact acceptance. ids: the verify argmax, T*bw j-major (ids[j*bw+b] is the base
        model's greedy token after row (b, j)). Row b accepts the longest prefix with
        drafts[b][j] == ids[j*bw+b], capped by n_draft[b]; it emits ids[b, 0..m_b] -- the first is
        exactly the plain-decode token, the rest equal the accepted drafts. With spec (the exact
        rule's device outputs), an exact row accepts by the device's test instead (see below)."""
        bw, K, T = plan.bw, plan.K, plan.T
        assert len(ids) >= T * bw, (len(ids), T, bw)
        # QWEN36_MTP_SPEC_SAMPLE: spec = {"accept": [M], "resid": [M], "exact": [bw], "greedy": [bw]} from the
        # verify (None with the flag off). An EXACT row (rule flag 1 and T > 0) accepts by the device's
        # Leviathan/Chen test (accept[j*bw+b] for draft j) and emits its accepted drafts, then the residual
        # at the first rejection, or the target's own sample (ids, the verify's device sample) at row m when
        # no draft is left. Every other row is the compare path below (greedy rows: byte-identical to off).
        m_list, emitted = [], []
        for b in range(bw):
            if not plan.active[b]:
                m_list.append(0)
                emitted.append([])
                continue
            exact = spec is not None and bool(spec["exact"][b]) and not bool(spec["greedy"][b])
            nd = int(plan.n_draft[b])
            m = 0
            if exact:
                while m < nd and float(spec["accept"][m * bw + b]) > 0.5:
                    m += 1
            else:
                while m < nd and int(plan.drafts[b][m]) == int(ids[m * bw + b]):
                    m += 1
            if spec is not None and not bool(spec["greedy"][b]):
                key = "exact" if exact else "compare"
                self.stats[key + "_drafted"] += nd
                self.stats[key + "_accepted"] += m
            lk = bool(plan.source) and plan.source[b] == "lookup"
            if lk:
                self.lstats["lookup_steps"] += 1
                self.lstats["lookup_drafted"] += nd
                self.lstats["lookup_accepted"] += m
                if plan.lookup_mustfail == "trust" and not exact:
                    m = nd  # MUST-FAIL: lookup drafts committed without the greedy compare
            self.stats["would_accept"] += m
            if self.reject_all:
                m = 0  # FORCE_REJECT: the verify still ran over every draft; commit depth 0 only
            m_list.append(m)
            if exact:
                # (FORCE_REJECT with the exact rule is refused at boot: a forced rejection has no exact tail here.)
                tail = int(spec["resid"][m * bw + b]) if m < nd else int(ids[m * bw + b])
                emitted.append([int(t) for t in plan.drafts[b][:m]] + [tail])
            elif lk and plan.lookup_mustfail == "trust":
                # MUST-FAIL: the committed ids are the lookup drafts themselves, then the verify id after them.
                emitted.append([int(t) for t in plan.drafts[b][:m]] + [int(ids[m * bw + b])])
            else:
                emitted.append([int(ids[j * bw + b]) for j in range(m + 1)])
            lr = self.lookup[b]
            if lr is not None and len(lr.hist) == plan.p[b] + 1:
                lr.hist.extend(emitted[-1])  # positions p+1 .. p+m+1; the last is the next step's t0

        commit = m_list
        if self.mustfail == "commit_plus1":
            # Must-fail: the state commit runs one verify depth past the accepted length (a
            # rejected draft enters the recurrence); emitted tokens, anchor and next position stay m_b.
            commit = [min(m + 1, T - 1) for m in m_list]
        gdn_onehot = [[1.0 if (plan.active[b] and commit[b] == j) or (not plan.active[b] and j == 0) else 0.0
                       for b in range(bw)] for j in range(T)]
        anchor_idx = [m_list[b] * bw + b for b in range(bw)]
        # the next step starts from the LAST EMITTED token (= ids[m] on the compare path; under the exact rule
        # the residual or the target's sample, which the chain must continue from)
        t0_next = [int(emitted[b][-1]) if plan.active[b] else 0 for b in range(bw)]

        reseed_pos, reseed_rope = self.reseed_positions(plan)
        draft_pos, draft_rope = [], []
        for k in range(K):
            row, rrow = [], []
            for b in range(bw):
                pk = plan.p[b] + m_list[b] + k
                # MTP KV index = pk - origin < pk: bounded by the row's page table like any base write.
                if plan.active[b] and plan.verify_pos[0][b] >= 0 and pk - plan.origin[b] < plan.kv_capacity[b]:
                    row.append(pk - plan.origin[b])
                    rrow.append(pk)
                else:
                    row.append(-1)
                    rrow.append(-1)
            draft_pos.append(row)
            draft_rope.append(rrow)
        return Accept(
            m=m_list,
            emitted=emitted,
            gdn_onehot=gdn_onehot,
            anchor_idx=anchor_idx,
            t0_next=t0_next,
            reseed_pos=reseed_pos,
            reseed_rope_pos=reseed_rope,
            draft_pos=draft_pos,
            draft_rope_pos=draft_rope,
        )

    @staticmethod
    def reseed_positions(plan):
        """Reseed: the MTP layer's KV entry at position p+j is (verify hidden of row (b, j), ids[b, j]).
        All T depths are written (the verify rows as they are, no slicing); depths past m_b hold
        rejected context and are overwritten by this step's draft steps (p+m_b .. p+m_b+K-1) or by
        the next step's reseed (which starts at p+m_b+1) before anything attends to them. A function of
        the plan alone, which is what lets QWEN36_MTP_EARLY_RESEED run it before the accept.
        Returns ([T][bw] MTP KV index or -1, [T*bw] j-major absolute RoPE position or -1)."""
        bw, T = plan.bw, plan.T
        reseed_pos, reseed_rope = [], []
        for j in range(T):
            row = []
            for b in range(bw):
                pj = plan.p[b] + j
                if plan.active[b] and plan.verify_pos[j][b] >= 0:
                    row.append(pj - plan.origin[b])
                    reseed_rope.append(pj)
                else:
                    row.append(-1)
                    reseed_rope.append(-1)
            reseed_pos.append(row)
        return reseed_pos, reseed_rope

    def finish(self, plan, acc, draft_ids, blocks=None):
        """Store the next drafts. draft_ids: K*bw k-major (draft_ids[k*bw+b] = draft k+1 of row b),
        or None when the drafter did not run (then the next step verifies without drafts)."""
        bw, K = plan.bw, plan.K
        self.stats["steps"] += 1
        for b in range(bw):
            if not plan.active[b]:
                continue
            st = self.slots[b]
            m = acc.m[b]
            self.stats["rows"] += 1
            self.stats["drafted"] += plan.n_draft[b]
            self.stats["accepted"] += m
            self.stats["emitted"] += m + 1
            nd = plan.n_draft[b]
            for k in range(min(nd, m + 1)):  # drafts 1..min(nd, m+1) were compared; the first m matched
                self.depth_try[k] += 1
                if k < m:
                    self.depth_hit[k] += 1
            self.mhist[m] += 1
            pb = plan.p[b] // POS_BIN
            while len(self.pos_rows) <= pb:
                self.pos_rows.append(0)
                self.pos_drafted.append(0)
                self.pos_accepted.append(0)
            self.pos_rows[pb] += 1
            self.pos_drafted[pb] += nd
            self.pos_accepted[pb] += m
            if self.capture is not None:
                self._capture_step(b, plan.p[b], acc.emitted[b], m, nd)
            st.next_p = plan.p[b] + m + 1
            st.t0 = acc.t0_next[b]
            st.drafts = [] if draft_ids is None else [int(draft_ids[k * bw + b]) for k in range(K)]
            # QWEN36_MTP_SPEC_SAMPLE: the chain's candidate block (the probs and the ids), kept per
            # depth so the next step's verify has the drafter's q.
            if blocks is None:
                st.qprobs, st.qids = [], []
            else:
                qpb, qib = blocks
                st.qprobs = [qpb[k][b] for k in range(K)]
                st.qids = [qib[k][b] for k in range(K)]

    # ----------------------------------------------------------------- QWEN36_MTP_BATCH_CAPTURE
    def _capture_step(self, b, p, emitted, m, nd):
        c = self.capture.get(b)
        if c is None:
            c = self.capture[b] = {"p0": int(p), "next": int(p), "ids": [], "m": [], "nd": [], "gaps": [], "trunc": 0}
        if int(p) != c["next"]:  # a plain-decode step (or a re-origin) in between: its token is not in ids
            c["gaps"].append((len(c["m"]), int(p)))
        if len(c["ids"]) < CAPTURE_MAX_TOKENS:
            c["ids"].extend(int(t) for t in emitted)
            c["m"].append(int(m))
            c["nd"].append(int(nd))
        else:
            c["trunc"] = 1
        c["next"] = int(p) + int(m) + 1

    def pop_capture(self, s, t=None):
        """The finished request's capture line for slot s (None when capture is off or the slot ran no
        speculative step). Call BEFORE reset_slot. ids: little-endian uint32, base64; m / nd: one digit (hex
        for K > 9) per step; gaps: step index and base position where steps were not contiguous."""
        if self.capture is None:
            return None
        c = self.capture.pop(int(s), None)
        if not c or not c["m"]:
            return None
        ids = base64.b64encode(struct.pack(f"<{len(c['ids'])}I", *c["ids"])).decode()
        hx = "0123456789abcdef"
        gaps = ";".join(f"{i}:{p}" for i, p in c["gaps"]) or "-"
        tt = f" t={t:.3f}" if t is not None else ""
        return (f"QWEN36_MTP_BATCH capture: slot={int(s)} p0={c['p0']} n={len(c['ids'])} steps={len(c['m'])}{tt} "
                f"truncated={c['trunc']} m={''.join(hx[x] for x in c['m'])} nd={''.join(hx[x] for x in c['nd'])} "
                f"gaps={gaps} ids={ids}")

    def lc_summary(self):
        """The long-context counters (see POS_BIN above), '/'-joined lists ttx parses (mtplog.LC_KEYS)."""
        kmax = max([k + 1 for k in range(MAX_K) if self.depth_try[k]] or [0])
        mmax = max([m + 1 for m in range(MAX_K + 1) if self.mhist[m]] or [1])
        return (f"depth_try={_ints(self.depth_try[:kmax]) or 0} depth_hit={_ints(self.depth_hit[:kmax]) or 0} "
                f"mhist={_ints(self.mhist[:mmax])} pos_bin={POS_BIN} pos_rows={_ints(self.pos_rows) or 0} "
                f"pos_drafted={_ints(self.pos_drafted) or 0} pos_accepted={_ints(self.pos_accepted) or 0}")

    def summary(self):
        s = self.stats
        rows = max(1, s["rows"])
        drafted = max(1, s["drafted"])
        return (
            f"steps={s['steps']} rows={s['rows']} tokens/row-step={s['emitted'] / rows:.3f} "
            f"accept_rate={s['accepted'] / drafted:.3f} would_accept_rate={s['would_accept'] / drafted:.3f} "
            f"emitted={s['emitted']} drafted={s['drafted']} accepted={s['accepted']} "
            f"exact_accept_rate={s['exact_accepted'] / max(1, s['exact_drafted']):.3f} "
            f"exact_rows={s['exact_drafted']} compare_accept_rate="
            f"{s['compare_accepted'] / max(1, s['compare_drafted']):.3f} "
            f"compare_rows={s['compare_drafted']} "
            f"fresh={s['fresh']} stale={s['stale']} capped={s['capped']} force_reject={int(self.reject_all)}"
        ) + (f" warm={s['warm']} warm_kept={s['warm_kept']}" if s["warm"] else "") + (f" mustfail={self.mustfail}" if self.mustfail else "") + (
            " " + " ".join(f"{k}={v}" for k, v in self.lstats.items()) if self.lstats["lookup_seeded"] else "") + (
            f" {self.lc_summary()}" if s["rows"] else "")

    # ----------------------------------------------------------------- QWEN36_MTP_BATCH_TRACE
    def trace_step(self, step, plan, acc, ids, draft_ids):
        """One line per traced row of a speculative step: what it verified, what it accepted, what
        it committed, and where the next step must start. Pure formatting (no state change)."""
        bw, T, K = plan.bw, plan.T, plan.K
        lines = []
        for b in self.traced_rows(bw):
            if not plan.active[b]:
                continue
            st = self.slots[b]
            nxt = [] if draft_ids is None else [int(draft_ids[k * bw + b]) for k in range(K)]
            lines.append(
                f"QWEN36_MTP_BATCH trace step={step} kind=spec bw={bw} K={K} row={b} req={self.row_ids[b]} "
                f"pos={plan.p[b]} t0={plan.t0[b]} origin={plan.origin[b]} n_draft={plan.n_draft[b]} "
                f"drafts={list(plan.drafts[b][:plan.n_draft[b]])} verify_ids={[int(ids[j * bw + b]) for j in range(T)]} "
                f"m={acc.m[b]} emitted={acc.emitted[b]} commit_depth={next((j for j in range(T) if acc.gdn_onehot[j][b] == 1.0), acc.m[b])} anchor={acc.anchor_idx[b]} "
                f"next_p={st.next_p if st else None} next_t0={st.t0 if st else None} next_drafts={nxt} "
                f"force_reject={int(self.reject_all)}"
            )
        return lines

    def trace_plain(self, step, tokens, positions, bw, why):
        """One line per traced row of a step that ran PLAIN decode instead (the row still advances
        one token; the next speculative step then re-origins or continues)."""
        lines = []
        for b in self.traced_rows(bw):
            if int(positions[b]) < 0:
                continue
            lines.append(
                f"QWEN36_MTP_BATCH trace step={step} kind=plain bw={bw} row={b} req={self.row_ids[b]} "
                f"pos={int(positions[b])} t0={int(tokens[b])} why={why}"
            )
        return lines


def kv_capacity_from_page_table(page_table_rows, block_size):
    """Addressable positions per row: the page-table width times block_size. The table is the
    runner's fixed-width block table, so this is the hard bound past which a KV write would index
    outside the row; unallocated entries point at vLLM's null block (harmless padding)."""
    return [len(r) * int(block_size) for r in page_table_rows]
