# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""patches/tt-metal/0058: the fused GDN prefill prep + scan on upstream's merged math, and its fail-closed warmup
self-test.

QWEN36_GDN_FUSED_UPMATH=1         the prep/scan compute kernels and the receiver reader from kernels_fused_um/ (upstream
                                  #57440 math as of the #58927 head: C3/C5 fused chains in DEST, nkd + dl*I hand-off,
                                  the scan's o/state adds accumulated in DEST, the #58925-#58927 prep-item hunks).
QWEN36_GDN_FUSED_UPMATH=prep3     patches/tt-metal/0059: the same kernels with the prep compute at >= HiFi3 and the scan at
                                  the arm's fidelity (the served-hifi2 form).
QWEN36_GDN_FUSED_UPMATH=mustfail  ttsim only (also prep3-mustfail): dl*I handed off as I; this self-test must REFUSE.

Read per call by the C++ op (banner "QWEN36_GDN_FUSED_UPMATH engaged"), hashed into the program. NOT bit-exact
(numerics tier 2: roundings move), so the self-test is a closeness bound against the CONTROL program at HiFi4, not a bitwise
compare. The C++ op requires QWEN36_GDN_WY_SFPU=1, QWEN36_GDN_PREP_ONEBLOCK=1 and QWEN36_GDN_PREP_COL_EXP/DST4 off.
QWEN36_GDN_UPMATH_SELFTEST=0 skips the self-test (op probes only; it logs "self-test SKIPPED", which served arms list
as absent).
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

from contextlib import contextmanager
import os

from loguru import logger
import torch
import ttnn


FLAG = "QWEN36_GDN_FUSED_UPMATH"
# Closeness bound per chip: relative Frobenius distance of (o, final state) to the CONTROL program -- flag off, at HiFi4
# whatever the arm's fidelity, so the reference is ~5x closer to fp64 than a hifi2 CONTROL and the distance measures the
# candidate's own error, not the CONTROL's. Calibrated on the served warmup's live inputs (layer 0, 2048 x token 0,
# zero state; host replay [source validation record]), where a hifi2 CONTROL is 7-11% off fp64 and the old
# same-fidelity bound refused prep3 on silicon (card [[0.064, 0.006], [0.022, 0.025]] vs 0.035):
#   prep3 at hifi2 vs the HiFi4 CONTROL: model inputs o 0.014-0.017, state 0.046-0.051; random 0.007-0.008
#   mode 1 at hifi3: model o 0.002-0.003, state 0.011-0.012; random <= 0.003
#   wrong-input rung g' = g - 1 (fp64): model o >= 0.156, state >= 0.29; random >= 0.66; must-fail state >= 16
# -> scan at hifi2: bound 0.08 (1.6x the largest model-input distance), wrong >= 0.12 (1.3x under the weakest rung);
#    hifi3/hifi4: bound 0.03, wrong >= 0.1. QWEN36_GDN_UPMATH_BOUND overrides the bound (wrong stays 1.5x/3.3x).
def bounds():
    fid = _tt_public_getenv("QWEN36_GDN_SCAN_FID", "hifi4").strip().lower()
    default = 0.08 if fid == "hifi2" else 0.03
    b = float(_tt_public_getenv("QWEN36_GDN_UPMATH_BOUND", str(default)))
    return b, (1.5 if fid == "hifi2" else 10.0 / 3.0) * b


@contextmanager
def control_fidelity():
    """The CONTROL reference runs at HiFi4 (QWEN36_GDN_SCAN_FID is read per call)."""
    saved = _tt_public_getenv("QWEN36_GDN_SCAN_FID")
    try:
        os.environ["QWEN36_GDN_SCAN_FID"] = "hifi4"
        yield
    finally:
        if saved is None:
            os.environ.pop("QWEN36_GDN_SCAN_FID", None)
        else:
            os.environ["QWEN36_GDN_SCAN_FID"] = saved
_DONE = set()


SIM_ONLY = ("mustfail", "prep3-mustfail", "diagprep", "diagscan", "exact")


def mode():
    """'' (off), '1' (0058), 'prep3' (0059: the prep compute at >= HiFi3, the HiFi2-safe variant) or a ttsim-only value (the must-fails and the
    per-kernel fidelity diagnostics), strictly parsed before any launch."""
    v = _tt_public_getenv(FLAG, "").strip()
    if v in ("", "0"):
        return ""
    if v not in ("1", "prep3") + SIM_ONLY:
        raise ValueError(f"{FLAG}={v!r}: expected unset, 0, 1, prep3 or (ttsim only) {', '.join(SIM_ONLY)}")
    if v in SIM_ONLY and not _tt_public_getenv("TT_METAL_SIMULATOR"):
        raise RuntimeError(f"{FLAG}={v} is allowed in ttsim only")
    return v


@contextmanager
def forced(value):
    """Set only this lever's switch ('' = off); pool, ONEBLOCK, WY and scan fidelity stay as the arm set them."""
    saved = _tt_public_getenv(FLAG)
    try:
        if value:
            os.environ[FLAG] = value
        else:
            os.environ[FLAG] = "0"
        yield
    finally:
        if saved is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = saved


def per_chip_rel(a, b):
    """Relative Frobenius distance ||a - b|| / ||b|| per device (inf on a shape mismatch or a non-finite value)."""
    out = []
    for x, y in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
        xx, yy = ttnn.to_torch(x).double(), ttnn.to_torch(y).double()
        if xx.shape != yy.shape or not bool(torch.isfinite(xx).all()):
            out.append(float("inf"))
            continue
        out.append(float((xx - yy).norm() / yy.norm().clamp_min(1e-30)))
    return out


def selftest(run, v, g, candidate, desc):
    """Fail-closed warmup self-test, once per process per mode, on the first fused call (the eager chunk-prefill
    warmup, before any trace capture). `run(v, g)` runs the served op (the arm's pool/ONEBLOCK/WY/fidelity) on these
    live inputs; `candidate` is the (o, final_state) this call produced with the flag on.

      reference  the flag off at HiFi4: the CONTROL program on the same inputs.
      close      candidate vs reference, o and final state, every device: relative distance <= BOUND.
      wrong      the flag-on program on g' = g - 1 (the decay path the upstream math rewrites) vs the
                 reference: relative distance >= WRONG_MIN on every device in o and final state; otherwise the
                 bound cannot see a wrong result and the boot is refused.
    Raises -> the boot is refused. QWEN36_GDN_FUSED_UPMATH=mustfail must end here."""
    m = mode()
    if not m or m in _DONE:
        return
    if _tt_public_getenv("QWEN36_GDN_UPMATH_SELFTEST", "1") == "0":
        logger.warning(f"QWEN36_GDN_FUSED_UPMATH self-test SKIPPED (QWEN36_GDN_UPMATH_SELFTEST=0) mode={m}")
        _DONE.add(m)
        return
    owned = []
    try:
        with forced(""), control_fidelity():
            reference = run(v, g)
        owned.extend(reference)
        g_wrong = ttnn.add(g, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(g_wrong)
        wrong = run(v, g_wrong)  # flag as the arm set it
        owned.extend(wrong)
        close = [per_chip_rel(c, r) for c, r in zip(candidate, reference)]
        far = [per_chip_rel(w, r) for w, r in zip(wrong, reference)]
        BOUND, WRONG_MIN = bounds()
        ok = (len(close) == 2 and all(d <= BOUND for pair in close for d in pair)
              and all(d >= WRONG_MIN for pair in far for d in pair))
        fmt = lambda xs: [[round(d, 6) for d in pair] for pair in xs]  # noqa: E731
        if not ok:
            raise RuntimeError(
                f"QWEN36_GDN_FUSED_UPMATH self-test REFUSED mode={m} {desc}: (o, state) per chip rel. distance to "
                f"CONTROL {fmt(close)} (must be <= {BOUND}); wrong-input rung {fmt(far)} (must be >= {WRONG_MIN})")
        logger.warning(
            f"QWEN36_GDN_FUSED_UPMATH self-test pass mode={m} {desc}: (o, state) per chip rel. distance to CONTROL "
            f"{fmt(close)} (bound {BOUND}); wrong-input rung {fmt(far)} (>= {WRONG_MIN})")
        _DONE.add(m)
    finally:
        for tensor in owned:
            ttnn.deallocate(tensor)
