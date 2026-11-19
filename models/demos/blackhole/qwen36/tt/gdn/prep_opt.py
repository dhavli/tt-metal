# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""patches/tt-metal/0056: two independently gated edits to the fused GDN prep compute kernel, and their fail-closed
warmup self-test.

QWEN36_GDN_PREP_COL_EXP=1  decay_exp / decayfac exp on the live column faces only (VectorMode::C).
QWEN36_GDN_PREP_DST4=1     the prep's independent broadcasts, q^2/k^2 and the k_dec transpose in four-tile DEST groups.

Both are read per call by the C++ op (banner "<FLAG> engaged"), hashed into the program, and bit-exact by construction
on every live lane. <FLAG>_MUSTFAIL=1 corrupts that edit's kernel path and is legal under ttsim only (C++ and here).
QWEN36_GDN_PREP_SELFTEST=0 skips the self-test (op probes only; it logs "self-test SKIPPED", which served arms list as
absent).
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

from contextlib import contextmanager
import os

from loguru import logger
import torch
import ttnn


FLAGS = ("QWEN36_GDN_PREP_COL_EXP", "QWEN36_GDN_PREP_DST4")
FAULTS = tuple(k + "_MUSTFAIL" for k in FLAGS)
_DONE = set()


def enabled():
    """The engaged flags, strictly parsed before any launch (the simulator-only faults included)."""
    values = {}
    for key in FLAGS + FAULTS:
        value = _tt_public_getenv(key, "0")
        if value not in ("0", "1"):
            raise ValueError(f"{key}={value!r}: expected unset, 0 or 1")
        values[key] = value == "1"
    for flag, fault in zip(FLAGS, FAULTS):
        if values[fault]:
            if not _tt_public_getenv("TT_METAL_SIMULATOR"):
                raise RuntimeError(f"{fault} is allowed in ttsim only")
            if not values[flag]:
                raise ValueError(f"{fault} requires {flag}=1")
    return tuple(flag for flag in FLAGS if values[flag])


@contextmanager
def variant(flags, faults=()):
    """Set only this lever's switches; pool, ONEBLOCK, WY and scan fidelity stay as the arm set them."""
    saved = {key: _tt_public_getenv(key) for key in FLAGS + FAULTS}
    try:
        for key in FLAGS + FAULTS:
            os.environ[key] = "1" if key in flags or key in faults else "0"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def per_chip_bits(a, b):
    """Differing raw-bit elements per device (a shape/dtype mismatch counts every element)."""
    out = []
    for x, y in zip(ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)):
        xx, yy = ttnn.to_torch(x).contiguous(), ttnn.to_torch(y).contiguous()
        if xx.shape != yy.shape or xx.dtype != yy.dtype:
            out.append(int(xx.numel()))
            continue
        iv = torch.int32 if xx.element_size() == 4 else torch.int16
        out.append(int((xx.view(iv) != yy.view(iv)).sum()))
    return out


def selftest(run, v, g, candidate, desc):
    """Fail-closed warmup self-test, once per process per flag set, on the first fused call (the eager chunk-prefill
    warmup, before any trace capture). `run(v, g)` runs the served op (the arm's pool/ONEBLOCK/WY/fidelity) on these
    live inputs; `candidate` is the (o, final_state) this call produced with the arm's flags.

      reference  both 0056 flags off: the CONTROL program on the same inputs.
      per rung   each engaged flag alone, then both together (the last is `candidate`): o and final state must be
                 BITWISE equal to the reference on every device.
      wrong      the rung's program on g' = 0.5*g - 0.125 (the decay path both edits sit on) must DIFFER from the
                 reference on every device, in both o and final state; otherwise the comparison is void and the boot
                 is refused.
    Raises -> the boot is refused. Every *_MUSTFAIL must end here."""
    active = enabled()
    if not active:
        return
    faults = tuple(key for key in FAULTS if _tt_public_getenv(key) == "1")
    key = (active, faults)
    if key in _DONE:
        return
    if _tt_public_getenv("QWEN36_GDN_PREP_SELFTEST", "1") == "0":
        logger.warning(f"QWEN36_GDN_PREP self-test SKIPPED (QWEN36_GDN_PREP_SELFTEST=0) flags={active}")
        _DONE.add(key)
        return
    rungs = [(flag,) for flag in active]
    if len(active) == 2:
        rungs.append(active)
    owned = []
    try:
        with variant(()):
            reference = run(v, g)
        owned.extend(reference)
        g_wrong = ttnn.add(ttnn.multiply(g, 0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG), -0.125,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(g_wrong)
        for flags in rungs:
            rung_faults = tuple(f for f in faults if f.removesuffix("_MUSTFAIL") in flags)
            with variant(flags, rung_faults):
                if flags == active:
                    result = candidate
                else:
                    result = run(v, g)
                    owned.extend(result)
                wrong = run(v, g_wrong)
                owned.extend(wrong)
            equal = [per_chip_bits(a, b) for a, b in zip(reference, result)]
            different = [per_chip_bits(a, b) for a, b in zip(reference, wrong)]
            if len(equal) != 2 or any(n for pair in equal for n in pair) or not all(
                    n > 0 for pair in different for n in pair):
                raise RuntimeError(
                    f"QWEN36_GDN_PREP self-test REFUSED flags={flags} faults={rung_faults} {desc}: differing (o, state) "
                    f"elements per chip vs CONTROL {equal} (must be 0); wrong-input rung {different} (must be > 0)")
            logger.warning(
                f"QWEN36_GDN_PREP self-test exact flags={flags} {desc}: (o, state) per chip vs CONTROL {equal}; "
                f"wrong-input rung {different}")
        _DONE.add(key)
    finally:
        for tensor in owned:
            ttnn.deallocate(tensor)
