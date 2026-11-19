"""pp/host: per-request prefill host-gap levers (an internal record).

Three boot-time flags, each default off. Off means the served path runs as before, byte for byte: every hook in
model.py tests one attribute that these parsers set to False/None.

QWEN36_PREFILL_RESET_TRACE=1
    The per-request GDN reset (48 layers x (3 conv taps + recurrent state + conv carry) = 240 eager ttnn.copy from
    persistent zero sources into the persistent B=1 prefill scratch) is captured once at warmup as ONE trace and
    replayed per request (one execute_trace instead of 240 host dispatches, ~10 ms of host time with the device
    idle at request start; pht 10-15: gdn_reset 9.5-22 ms/request). Replayed only while the scratch the trace baked
    is bound (identity guard, else the eager reset runs). In-boot self-test on both chips: junk is copied into every
    target, read back (wrong-input branch: it must be junk, else the readback is blind), the trace is replayed and
    every target must read back exactly zero. _MUSTFAIL=1 captures the trace without the last layer's recurrent-state
    copy, so the self-test must refuse the boot (ttsim rung only).

QWEN36_PREFILL_TAIL_NOSYNC=1
    prefill_paged_slots only. The drain before the masked-bucket tail and the synchronize after the tail's bucket
    replay are dropped: the host enqueues the tail's input DMAs, its replay, the eager logits select and the MTP warm
    gathers while the device still runs the chunk replays. The host input tensors stay referenced (HostRefs) until
    the blocking logits readback has returned, so no in-flight DMA reads freed memory. Same CQ, same order: the
    device runs the same commands in the same sequence.

QWEN36_PREFILL_DEFER_FINISH=1
    prefill_paged_slots only. Each user's MTP warm-tail finish (host argmax, warm token/RoPE/fill writes and the
    KV-only MTP pass, pht 10-19 mtp_finish 23-53 ms host with the device idle) is parked instead of run inline. An
    inner user's runs from the next user's first warm gather (_mtp_warm_seg / _mtp_warm_span_seg flush first, before
    they overwrite the window buffers), i.e. behind that user's first replay. The last user's runs at the next model
    entry (prefill_forward / decode_forward / release_request), after the step published its first tokens.
    _MUSTFAIL=1 drops the deferred finish (the slot starts cold), so the in-boot equivalence self-test must refuse
    (ttsim rung only).

The TAIL_NOSYNC / DEFER_FINISH self-test (model.prefill_host_selftest, run once at the end of the decode warmup on
both chips) prefills two synthetic prompts in one call through prefill_paged_slots with the levers off, then on,
and compares both users' host logits, the GDN scratch state, the MTP warm windows each finish read and both slot
records, bitwise. Its wrong-input branch changes user 0's last token through the lever path: its logits must
differ, else the comparison is blind and the boot is refused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_RESET_TRACE = "QWEN36_PREFILL_RESET_TRACE"
ENV_RESET_TRACE_MF = "QWEN36_PREFILL_RESET_TRACE_MUSTFAIL"
ENV_TAIL_NOSYNC = "QWEN36_PREFILL_TAIL_NOSYNC"
ENV_DEFER_FINISH = "QWEN36_PREFILL_DEFER_FINISH"
ENV_DEFER_FINISH_MF = "QWEN36_PREFILL_DEFER_FINISH_MUSTFAIL"


def _on(env, name):
    v = (env.get(name) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"{name}={v!r}: only 0 or 1 is defined")
    return v == "1"


@dataclass(frozen=True)
class HostFlags:
    reset_trace: bool = False
    reset_trace_mf: bool = False
    tail_nosync: bool = False
    defer_finish: bool = False
    defer_finish_mf: bool = False

    @property
    def any_request_path(self):
        """A lever that changes prefill_paged_slots' request-time flow (the equivalence self-test covers these)."""
        return self.tail_nosync or self.defer_finish


def parse_flags(env=None, *, bucket_trace=False, num_devices=2):
    """Read the three flags (boot time). Refuses inconsistent combinations instead of silently serving another arm."""
    env = os.environ if env is None else env
    f = HostFlags(
        reset_trace=_on(env, ENV_RESET_TRACE),
        reset_trace_mf=_on(env, ENV_RESET_TRACE_MF),
        tail_nosync=_on(env, ENV_TAIL_NOSYNC),
        defer_finish=_on(env, ENV_DEFER_FINISH),
        defer_finish_mf=_on(env, ENV_DEFER_FINISH_MF),
    )
    if f.reset_trace_mf and not f.reset_trace:
        raise RuntimeError(f"{ENV_RESET_TRACE_MF}=1 needs {ENV_RESET_TRACE}=1. Refusing to boot.")
    if f.defer_finish_mf and not f.defer_finish:
        raise RuntimeError(f"{ENV_DEFER_FINISH_MF}=1 needs {ENV_DEFER_FINISH}=1. Refusing to boot.")
    if (f.reset_trace or f.tail_nosync or f.defer_finish) and num_devices <= 1:
        raise RuntimeError("QWEN36_PREFILL_{RESET_TRACE,TAIL_NOSYNC,DEFER_FINISH} are TP (num_devices>1) levers. Refusing.")
    if f.tail_nosync and not bucket_trace:
        raise RuntimeError(f"{ENV_TAIL_NOSYNC}=1 needs QWEN36_PREFILL_BUCKET_TRACE (the traced tail). Refusing to boot.")
    return f


class HostRefs:
    """Host tensors whose non-blocking host->device copies may still be in flight. Released only after a blocking
    readback (or a synchronize) has drained the command queue they were enqueued on."""

    def __init__(self):
        self._refs = []
        self.peak = 0

    def add(self, items):
        self._refs.extend(items)
        self.peak = max(self.peak, len(self._refs))

    def __len__(self):
        return len(self._refs)

    def release(self):
        n = len(self._refs)
        self._refs.clear()
        return n


@dataclass
class DeferredFinish:
    """The last user's MTP warm finish, parked until the next model entry. `req` is the warm pass's per-request state
    (taken off the _MTPWarm so its abort() in prefill_paged_slots' finally is a no-op); `args` are finish()'s."""

    req: object
    args: tuple


def reset_selftest(ttnn, mesh, device, pairs, replay, num_devices, mustfail=False, junk_value=1.5):
    """QWEN36_PREFILL_RESET_TRACE self-test (shared by the boot and the ttsim probe). pairs: [(layer, zero source,
    target)]. Junk is copied into every target with the SAME copy program as the reset (junk built with the zero
    source's exact spec, so nothing compiles after a trace is parked), every target is read back over the whole mesh
    and must be junk (the wrong-input branch: a blind readback is refused), then `replay()` runs the reset and every
    target must read back exactly zero. Raises RuntimeError (REFUSED) on any failure; returns the target count."""
    import torch

    rep = ttnn.ReplicateTensorToMesh(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    junk = {}

    def _junk_for(z):
        key = id(z)
        if key not in junk:
            j = ttnn.from_torch(torch.full(tuple(z.shape), junk_value, dtype=torch.float32), dtype=z.dtype,
                                layout=z.layout, device=mesh, mesh_mapper=rep, memory_config=z.memory_config())
            if (j.dtype, j.layout, tuple(j.shape), j.memory_config()) != (
                z.dtype, z.layout, tuple(z.shape), z.memory_config()
            ):
                ttnn.deallocate(j)
                raise RuntimeError("QWEN36_PREFILL_RESET_TRACE self-test REFUSED: junk spec != zero-source spec")
            junk[key] = j
        return junk[key]

    def _read(t):
        v = ttnn.to_torch(t, mesh_composer=comp)
        if int(v.shape[0]) < num_devices * int(t.shape[0]):
            raise RuntimeError("QWEN36_PREFILL_RESET_TRACE self-test REFUSED: readback did not cover every device")
        return v

    blind, bad = [], []
    try:
        for _li, z, t in pairs:
            ttnn.copy(_junk_for(z), t)
        ttnn.synchronize_device(device)
        for li, _z, t in pairs:
            if not bool((_read(t).float() == junk_value).all()):
                blind.append(li)
        replay()
        ttnn.synchronize_device(device)
        for li, _z, t in pairs:
            if bool((_read(t) != 0).any()):
                bad.append(li)
    finally:
        for j in junk.values():
            ttnn.deallocate(j)
    if blind:
        raise RuntimeError(
            f"QWEN36_PREFILL_RESET_TRACE self-test REFUSED: the wrong-input branch (junk, no reset) read back as "
            f"non-junk on layers {sorted(set(blind))} (the readback is blind)"
        )
    if bad:
        raise RuntimeError(
            f"QWEN36_PREFILL_RESET_TRACE self-test REFUSED: targets not zero after the reset replay on layers "
            f"{sorted(set(bad))} (mustfail={mustfail})"
        )
    return len(pairs)
