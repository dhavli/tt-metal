# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side (pure torch) value builders for the traced masked-bucket prefill.

The eager masked-bucket prefill rebuilds ~151 small host tensors per request (GDN beta/g and
q/k/v validity masks in every GDN layer, the FIR decode-window one-hot in every GDN layer, the
KV-fill page table, the logits one-hot). Each of those is a ``ttnn.from_torch`` host write, which
is illegal inside a captured trace ("Writes are not supported during trace capture") and is also
the bulk of the per-request host dispatch cost.

The traced path hoists all of them into ONE model-owned set of persistent device buffers per
bucket whose SHAPES depend only on the bucket, so a single captured program set serves every
``actual_len``; only the VALUES change per request and are DMA'd in with
``ttnn.copy_host_to_device_tensor`` before ``ttnn.execute_trace``.

Everything here is pure torch (no ttnn import) so the value logic — the part that is easy to get
subtly wrong — is unit-testable on a host without a Tenstorrent device.
"""

import torch

# Default paged-KV block size for the Qwen3.6 serving path (get_block_size(kv_caches)).
DEFAULT_BLOCK_SIZE = 64


def host_masks(actual_len, bucket):
    """GDN validity mask [1, bucket, 1] float32: 1.0 for t < actual_len else 0.0.

    One tensor feeds BOTH device buffers: the fp32 beta/g mask and the bf16 q/k/v mask (the eager
    path in gdn/fused_chunk.py builds the same float32 ``_mt`` and uploads it twice, once as
    float32 and once as ``q.dtype``). 0.0/1.0 are exact in bf16, so the traced values are
    bit-identical to the eager ones.
    """
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    m = torch.zeros(1, bucket, 1, dtype=torch.float32)
    m[:, :actual_len, :] = 1.0
    return m


def host_conv_sel(actual_len, bucket, kernel_size):
    """FIR decode-window one-hot [1, K-1, bucket+K-1] float32 (sel[0, j, actual_len+j] = 1.0).

    Selects rows ``x_padded[:, actual_len : actual_len+(K-1)]`` — i.e. the last K-1 REAL conv
    inputs ``x[actual_len-(K-1) : actual_len]`` — via a matmul instead of a static slice, so the
    program depends only on shapes (fixed per bucket) and only the values depend on actual_len.
    Mirrors the one-hot built inside ``_causal_conv1d_fir`` for the eager int-valid_len path.
    """
    assert kernel_size >= 2, f"kernel_size {kernel_size} must be >= 2"
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    total_len = (kernel_size - 1) + bucket
    sel = torch.zeros(1, kernel_size - 1, total_len, dtype=torch.float32)
    for j in range(kernel_size - 1):
        sel[:, j, actual_len + j] = 1.0
    return sel


def host_logit_sel(actual_len, bucket):
    """Last-real-row one-hot [1, 1, 1, bucket] float32 for the TP logits select (row actual_len-1)."""
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
    sel[0, 0, 0, actual_len - 1] = 1.0
    return sel


def fill_pt_row(page_table, chunk_start, actual_len, bucket, pad_block, block_size=DEFAULT_BLOCK_SIZE, mapped_pad=True):
    """FIXED-WIDTH KV-fill page table [1, ceil(bucket/block_size)] int32 for the traced bucket body.

    The eager path sizes this table to the REAL blocks only (``page_table[:, blk0:blkN]`` with
    blkN = ceil((chunk_start+actual_len)/block_size)), so its width — and hence the paged_fill_cache
    program plus the ``ttnn.slice`` that trims K/V to page_len — changes with actual_len. A trace
    needs one fixed shape, so the traced body always fills the whole bucket: width = bucket/block_size
    makes ``page_len == S``, which also removes the slice entirely.

    Entries ``[0, nreal)`` are the request's real blocks. Trailing entries hold K/V for PAD rows,
    which causal SDPA never reads (k <= q < actual_len) and decode later overwrites via
    paged_update_cache; they must nevertheless point somewhere harmless:
      * the request's OWN mapped block at that index when it is non-zero (the natural full-bucket
        fill — no extra block needed, and the demo/tests' arange page tables always hit this), else
      * ``pad_block``, a scratch physical block no request owns (default: the last KV block, env
        QWEN36_PREFILL_BUCKET_PAD_BLOCK). Never 0, because block 0 is a real request's block for a
        zero-padded page-table row.
    mapped_pad=False (what the served replay passes): every entry past the real blocks is ``pad_block``,
    whatever the row holds there. A vLLM block-table row can carry entries past the request's own blocks
    that the request does not own, and the eager path never writes past the real blocks either.
    """
    # A bucket SHORTER than one block (block 2048, the prefix-caching arm: buckets 128..1024) is one block wide:
    # paged_fill_cache writes its S rows from the block start, and chunk_start is block-aligned, so the rows past S
    # in that block are untouched (ttsim: bench/probes/sim_prefix_block2048.py). Either the bucket is a whole number
    # of blocks (block 64, the shipped arm: unchanged) or one block holds the whole bucket.
    assert bucket % block_size == 0 or block_size % bucket == 0, (
        f"bucket {bucket} must be a multiple or a divisor of block_size {block_size}")
    assert chunk_start % block_size == 0, f"chunk_start {chunk_start} must be block-aligned"
    assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
    assert int(pad_block) != 0, "pad_block must not be block 0 (it aliases a real request's block)"
    width = -(-bucket // block_size)
    blk0 = chunk_start // block_size
    nreal = -(-(chunk_start + actual_len) // block_size) - blk0  # ceil div
    nreal = max(0, min(nreal, width))
    pt = page_table.reshape(1, -1)
    assert blk0 + nreal <= pt.shape[1], (
        f"page table row of {pt.shape[1]} blocks does not cover the {nreal} real block(s) at "
        f"offset {blk0} for chunk_start={chunk_start}, actual_len={actual_len}"
    )
    row = torch.full((1, width), int(pad_block), dtype=torch.int32)
    for j in range(width):
        idx = blk0 + j
        if idx >= pt.shape[1]:
            continue  # page-table row ends before the bucket -> scratch block
        v = int(pt[0, idx])
        if j < nreal or (mapped_pad and v != 0):
            row[0, j] = v
    return row


def parse_bucket_trace_gate(value, all_buckets):
    """Parse QWEN36_PREFILL_BUCKET_TRACE into a sorted tuple of buckets to trace.

    "0"/""/unset -> () (today's eager masked path, byte-for-byte). "1" -> every bucket.
    A comma list ("128", "128,256") -> exactly those buckets, so the 128 bucket can ship alone
    while the trace-region footprint of the larger ones is measured.
    """
    if value is None:
        return ()
    value = str(value).strip()
    if value in ("", "0", "off", "false"):
        return ()
    if value in ("1", "all", "true"):
        return tuple(sorted(all_buckets))
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        b = int(part)
        assert b in all_buckets, f"QWEN36_PREFILL_BUCKET_TRACE bucket {b} is not one of {tuple(all_buckets)}"
        out.append(b)
    return tuple(sorted(set(out)))


def parse_bucket_trace_mustfail(value):
    """QWEN36_PREFILL_BUCKET_TRACE_MUSTFAIL: "" / unset -> None; "stale" -> "stale". Anything else raises
    (a typo in a must-fail rung must not silently run the correct path)."""
    v = "" if value is None else str(value).strip()
    if v in ("", "0"):
        return None
    if v == "stale":
        return v
    raise ValueError(f"QWEN36_PREFILL_BUCKET_TRACE_MUSTFAIL={value!r}: only 'stale' is defined")


def parse_bucket_kda(value, mustfail, bucket_trace_on):
    """QWEN36_PREFILL_BUCKET_KDA (+ _MUSTFAIL) -> (on, mustfail). "" / "0" = off; "1" = on. Anything else raises,
    and so do the flag without QWEN36_PREFILL_BUCKET_TRACE (it only changes the TRACED masked bucket) and a
    must-fail without the flag (a typo in a rung must not silently run the plain path)."""
    v = "" if value is None else str(value).strip()
    mf = "" if mustfail is None else str(mustfail).strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"QWEN36_PREFILL_BUCKET_KDA={value!r}: expected unset, 0 or 1")
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_PREFILL_BUCKET_KDA_MUSTFAIL={mustfail!r}: expected unset, 0 or 1")
    on, mfon = v == "1", mf == "1"
    if mfon and not on:
        raise ValueError("QWEN36_PREFILL_BUCKET_KDA_MUSTFAIL=1 needs QWEN36_PREFILL_BUCKET_KDA=1")
    if on and not bucket_trace_on:
        raise ValueError("QWEN36_PREFILL_BUCKET_KDA=1 needs QWEN36_PREFILL_BUCKET_TRACE (it changes only the traced "
                         "masked bucket): refusing")
    return on, mfon


def parse_bucket_kda_max_len(value, mustfail, bucket_kda_on):
    """QWEN36_PREFILL_BUCKET_KDA_MAX_LEN (+ _MUSTFAIL) -> (max_len or None, mustfail).

    Unset / "" / "0" = off: QWEN36_PREFILL_BUCKET_KDA runs exactly as before (the kda body for every tail). A positive
    integer N keeps BOTH masked-bucket bodies per bucket (the shipped MAC FIR one and the kda one) and replays the kda
    body only for a tail that ends at prompt position <= N; every longer prompt replays the FIR body, i.e. it prefills
    exactly as the arm without QWEN36_PREFILL_BUCKET_KDA. Anything else raises, and so do the flag without
    QWEN36_PREFILL_BUCKET_KDA=1 and a must-fail without the flag."""
    v = "" if value is None else str(value).strip()
    mf = "" if mustfail is None else str(mustfail).strip()
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_PREFILL_BUCKET_KDA_MAX_LEN_MUSTFAIL={mustfail!r}: expected unset, 0 or 1")
    if v in ("", "0"):
        if mf == "1":
            raise ValueError("QWEN36_PREFILL_BUCKET_KDA_MAX_LEN_MUSTFAIL=1 needs QWEN36_PREFILL_BUCKET_KDA_MAX_LEN")
        return None, False
    try:
        n = int(v)
    except ValueError:
        n = -1
    if n <= 0 or str(n) != v:
        raise ValueError(f"QWEN36_PREFILL_BUCKET_KDA_MAX_LEN={value!r}: expected unset, 0 or a positive integer")
    if not bucket_kda_on:
        raise ValueError("QWEN36_PREFILL_BUCKET_KDA_MAX_LEN needs QWEN36_PREFILL_BUCKET_KDA=1: refusing")
    return n, mf == "1"


def bucket_kda_selects(tail_end, max_len):
    """True = replay the kda body for a masked-bucket tail ending at prompt position tail_end (chunk_start +
    actual_len). max_len None (QWEN36_PREFILL_BUCKET_KDA_MAX_LEN off) is never asked: the caller keeps one body."""
    return int(tail_end) <= int(max_len)


def bucket_bodies_gate(fir, fir2, kda, rel_max=0.25):
    """In-boot self-test of QWEN36_PREFILL_BUCKET_KDA_MAX_LEN for ONE device, on the eager masked-bucket body's hidden
    output: fir and fir2 = two FIR-body passes on the same inputs, kda = the kda-body pass. Passes when
      fir == fir2 bitwise   (the long-prompt body is deterministic, so a long prompt prefills exactly as without bkda),
      kda != fir            (the two bodies really differ: a set captured twice with one body would be blind),
      rel-L2(kda, fir) <= rel_max (a sanity bound on the 64-layer hidden: the tier-2 conv rounding compounds through
                            the layers; a wrong-history or garbage body lands far above it). The numerics gate
                            is the tier-2 equiv + gsm8k of an internal record, not this bound.
    Returns a dict with ok and the three parts."""
    f, f2, k = fir.float(), fir2.float(), kda.float()
    same_fir = bool(f.shape == f2.shape and torch.equal(f.reshape(-1).view(torch.int32), f2.reshape(-1).view(torch.int32)))
    n_diff = int((k != f).sum()) if k.shape == f.shape else k.numel()
    den = float(f.double().norm())
    rel = float((k.double() - f.double()).norm() / den) if den > 0 and k.shape == f.shape else float("inf")
    distinct, close = n_diff > 0, rel <= rel_max
    return dict(ok=bool(same_fir and distinct and close), fir_deterministic=same_fir, distinct=bool(distinct),
                diff_elems=n_diff, rel_l2=rel, rel_max=rel_max)


def conv_ref64(x, hist, taps):
    """fp64 reference of the GDN prefill causal conv + SiLU for ONE device: x [T, C], hist [K-1, C] (the previous
    K-1 inputs, oldest first), taps [K, C] (taps[0] multiplies the oldest input). out[t] = silu(sum_k xp[t+k]*taps[k])
    with xp = cat(hist, x); the MAC FIR (_causal_conv1d_fir) and the kda op both compute this."""
    x, hist, taps = x.double(), hist.double(), taps.double()
    K, T = taps.shape[0], x.shape[0]
    xp = torch.cat([hist, x], dim=0)
    acc = torch.zeros_like(x)
    for k in range(K):
        acc += xp[k : k + T] * taps[k]
    return torch.nn.functional.silu(acc)


def kda_conv_gate(ref64, fir, cand, wrong, rel_floor=2.0**-8):
    """Tier-2 gate of QWEN36_PREFILL_BUCKET_KDA's in-boot self-test for ONE device. ref64 = conv_ref64 (exact), fir
    = the shipped MAC FIR output, cand = the kda output, wrong = the kda output with a WRONG history (the in-boot
    wrong-input branch). The candidate passes when its max error vs the exact reference is at most twice the FIR's
    (or a floor of rel_floor * max|ref|, for an exact FIR); the wrong branch must FAIL the same test, otherwise the
    test cannot see a history error and the boot is refused. Returns a dict with ok, cand_ok, wrong_fails."""
    ref = ref64.double()
    scale = float(ref.abs().max())
    err = {k: float((t.double() - ref).abs().max()) for k, t in (("fir", fir), ("cand", cand), ("wrong", wrong))}
    bound = max(2.0 * err["fir"], rel_floor * scale)
    cand_ok = err["cand"] <= bound
    wrong_fails = err["wrong"] > bound
    return dict(ok=bool(cand_ok and wrong_fails and scale > 0), cand_ok=bool(cand_ok), wrong_fails=bool(wrong_fails),
                scale=scale, bound=bound, err_fir=err["fir"], err_cand=err["cand"], err_wrong=err["wrong"])


class PrefillPhaseTimer:
    """Host wall-clock phase split of ONE request's prefill (QWEN36_PREFILL_PHASE_TIMING=1).

    mark(name) closes the phase that ran since the previous mark (the caller synchronizes the device
    first, so a phase holds its own device work). Repeated names accumulate, in first-seen order.
    line() adds a final "rest" phase up to now, so the phases sum to total_ms. Pure stdlib."""

    def __init__(self, clock=None):
        import time

        self._clock = clock or time.perf_counter
        self._t0 = self._last = self._clock()
        self.ms = {}
        self.info = {}

    def mark(self, name):
        now = self._clock()
        self.ms[name] = self.ms.get(name, 0.0) + (now - self._last) * 1000.0
        self._last = now

    def line(self, prefix="QWEN36_PREFILL_PHASE_TIMING engaged:"):
        self.mark("rest")
        total = (self._last - self._t0) * 1000.0
        info = " ".join(f"{k}={v}" for k, v in self.info.items())
        phases = " ".join(f"{k}={v:.2f}" for k, v in self.ms.items())
        return f"{prefix} {info} total_ms={total:.2f} | {phases}".replace("  ", " ")
