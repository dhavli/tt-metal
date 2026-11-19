# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFIX_SNAP_FAST: the S12P GDN checkpoint readback without the whole-batch host untilize.

Diagnosis (cold-pp, server logs of ab20261026T060022Z and ab20261030T152205Z): every align-mode head step of a cold
prompt is followed by a 0.75-1.4 s engine stall before the tail step starts (S12 has 0.06-0.16 s there). The stall is
`prefix_cache_snapshot`, which for EVERY step with a checkpoint reads the WHOLE batched GDN state back, whatever the
number of slots wanted: per GDN layer ttnn.to_torch of rec_state [B=8, Nv=24, 128, 128] bf16 TILE on each of the two
cards (12.6 MB/layer, 604 MB/step) plus the K conv taps [1, B->32 tile rows, 5120], all untilized on the host, then a
host clone of the wanted rows. One checkpointed row is 75.75 MiB.

This module reads only what is wanted and moves the untilize to the device:
  * slots few (2 * len(slots) <= B): per wanted slot a device ttnn.slice of row `slot` (dim 0, a data-movement
    program, any dtype) -> [1, Nv, Dk, Dv] per card;
  * slots many: the whole tensor, as before (the volume is the same, only the untilize moves);
  * untilize on device (ttnn.to_layout ROW_MAJOR) when the tensor is bfloat16 (bf16 passes the unpacker unchanged);
    any other dtype (fp32 recurrent state, QWEN35_GDN_STATE_BF16 unset) is read in TILE and untilized on the host as
    before (an fp32 device untilize is not guaranteed bit-exact);
  * conv taps: device untilize of the [1, B, D] tap (bf16), host row pick.
Every value returned is the same bytes the legacy readback returns (data movement only): the ttsim probe
bench/probes/sim_prefix_snap_fast.py checks it bitwise against `legacy_snapshot` on a 2-chip mesh, and the served
self-test (`selftest_compare`) checks it again on the first real checkpoint after traffic.

QWEN36_PREFIX_SNAP_PACK (below) packs the reads of a group of layers into two blocking reads.

Shapes returned (per slot, per GDN layer): (rec [num_devices, Nv, Dk, Dv], [K x conv [num_devices, 1, D]]) -- what
model._validate_gdn_snapshot accepts and PrefixCache stores.

NO ttsim number is a performance number.
"""
from __future__ import annotations
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os

ENV = "QWEN36_PREFIX_SNAP_FAST"
ENV_MUSTFAIL = "QWEN36_PREFIX_SNAP_FAST_MUSTFAIL"  # "wrongslot": read slot (s+1) % B -> the self-test must refuse


def enabled() -> bool:
    return _tt_public_getenv(ENV, "0") == "1"


def mustfail() -> str:
    return _tt_public_getenv(ENV_MUSTFAIL, "").strip().lower()


def legacy_snapshot(ttnn, mesh_device, num_devices, dn_states, slots):
    """The shipped path (model.prefix_cache_snapshot before this lever), op for op. dn_states: list of
    (rec_state, conv_states, B, Nv, Dk, Dv) per GDN layer."""
    result = [[] for _ in slots]
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    for rec_state, conv_states, B, Nv, Dk, Dv in dn_states:
        rec = ttnn.to_torch(rec_state, mesh_composer=comp).reshape(num_devices, B, Nv, Dk, Dv)
        convs = [ttnn.to_torch(c, mesh_composer=comp) for c in conv_states]
        for out, slot in zip(result, slots):
            out.append((rec[:, slot].clone(), [c[:, slot:slot + 1].clone() for c in convs]))
    return result


def _to_host(ttnn, t, comp):
    """ttnn.to_torch of a device tensor, untilized on the device when that is exact (bf16)."""
    if t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT:
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        try:
            return ttnn.to_torch(rm, mesh_composer=comp)
        finally:
            ttnn.deallocate(rm)
    return ttnn.to_torch(t, mesh_composer=comp)


# ---- QWEN36_PREFIX_SNAP_NB -------------------------------------------------------------------------------------------
# fast_snapshot's reads, non-blocking. The shipped readback (and SNAP_FAST alone) calls ttnn.to_torch once per tensor:
# 48 GDN layers x (1 recurrent state + K=4 conv taps) = 240 blocking device reads per checkpoint step, each a full
# command-queue round trip (enqueue, wait for the device, wait for the PCIe completion) before the next op can even be
# enqueued. NB enqueues every device op and every read (ttnn.from_device(blocking=False), the KV host tier's store
# pattern) for a group of layers, then ONE ttnn.synchronize_device per group, then the host-only to_torch of each host
# tensor. Same ops, same bytes, same order on the one command queue; only the host waits are batched. The device
# temporaries (row slices, row-major copies) stay alive until their group's synchronize, which caps the transient DRAM
# at `nb_layers` layers (default 8: 8 x 786 KB per card per slot on the per-slot path, 8 x 6.3 MB on the whole-tensor
# path). Needs QWEN36_PREFIX_SNAP_FAST (it is that path's read schedule); the served self-test vs the legacy readback
# covers it.
ENV_NB = "QWEN36_PREFIX_SNAP_NB"
ENV_NB_LAYERS = "QWEN36_PREFIX_SNAP_NB_LAYERS"


def nb_enabled() -> bool:
    return _tt_public_getenv(ENV_NB, "0") == "1"


def nb_layers() -> int:
    v = _tt_public_getenv(ENV_NB_LAYERS, "").strip()
    n = int(v) if v else 8
    if not 1 <= n <= 48:
        raise ValueError(f"{ENV_NB_LAYERS}={v}: must be in [1, 48]")
    return n


class _Reads:
    """One group of device reads. blocking=True: ttnn.to_torch per tensor at add() (the SNAP_FAST read schedule).
    blocking=False: ttnn.from_device(blocking=False) at add(), one synchronize_device and the host to_torch at
    finish(). add() returns an index into finish()'s list."""

    def __init__(self, ttnn, mesh_device, comp, blocking):
        self.ttnn, self.mesh, self.comp, self.blocking = ttnn, mesh_device, comp, blocking
        self.items, self.temps = [], []

    def add(self, t, owned=False):
        ttnn = self.ttnn
        src = t
        if t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT:
            src = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            if owned:
                self.temps.append(t)
            owned = True
        if self.blocking:
            self.items.append(ttnn.to_torch(src, mesh_composer=self.comp))
            if owned:
                ttnn.deallocate(src)
            for x in self.temps:
                ttnn.deallocate(x)
            self.temps = []
        else:
            self.items.append(ttnn.from_device(src, blocking=False))
            if owned:
                self.temps.append(src)
        return len(self.items) - 1

    def finish(self):
        ttnn = self.ttnn
        if not self.blocking:
            ttnn.synchronize_device(self.mesh)
            self.items = [ttnn.to_torch(h, mesh_composer=self.comp) for h in self.items]
        for x in self.temps:
            ttnn.deallocate(x)
        self.temps = []
        return self.items


# ---- QWEN36_PREFIX_SNAP_PACK -----------------------------------------------------------------------------------------
# fast_snapshot's reads, packed. Measured (S12PHV-1M B1 sweep, ab20261031T160454Z): SNAP_FAST costs 264-281 ms for one
# slot and ~752 ms for 7, while one slot is only 37.7 MB per chip (~13-15 ms of PCIe at 5.3-6.6 GB/s). The rest is per-op
# overhead: per GDN layer a slice + to_layout + BLOCKING to_torch of rec, and a to_layout + blocking to_torch per conv tap,
# i.e. 48 x 5 = 240 blocking round trips. PACK gathers a group of `pack_layers` layers on the device first:
#   * rec: the wanted rows' slices of every layer in the group (or, when 2 * len(slots) > B, the layers' whole rec
#     tensors) concatenated on dim 0 (ttnn.concat of TILE tensors on an untiled outer dim: tile copies, data movement),
#     ONE to_layout ROW_MAJOR when bf16 (fp32 stays TILE and is untilized on the host, as _to_host decides), ONE blocking
#     to_torch, split on the host;
#   * conv taps: all K taps [1, B, D] of all layers in the group concatenated on dim 0, one to_layout, one blocking
#     read, host split and row pick as before.
# Two blocking reads per group instead of 5 per layer. Every device temporary is freed right after its read. Each chip
# packs its own values, so ConcatMeshToTensor(dim=0) returns [num_devices * rows, ...] device-major: the host split views
# it as [num_devices, rows, ...], which keeps the device order. The whole-tensor path packs at most max(1, G // 2) layers
# so that no group holds more than G * B / 2 rec rows: the transient DRAM is then <= 2 x G x 4 x 786 KB = 50 MB per chip
# at G = 8 on either path (packed + its row-major copy; the slices are freed before the untilize), the same bound SNAP_NB
# accepted. Same bytes as SNAP_FAST (data movement only); the served self-test vs the legacy readback covers it.
# Needs QWEN36_PREFIX_SNAP_FAST; mutually exclusive with QWEN36_PREFIX_SNAP_NB (non-blocking reads correlated with an
# unexplained engine stall; PACK stays blocking).
ENV_PACK = "QWEN36_PREFIX_SNAP_PACK"
ENV_PACK_LAYERS = "QWEN36_PREFIX_SNAP_PACK_LAYERS"
ENV_PACK_MUSTFAIL = "QWEN36_PREFIX_SNAP_PACK_MUSTFAIL"  # "swap": two layers' pieces swapped in the host split


def pack_enabled() -> bool:
    """QWEN36_PREFIX_SNAP_PACK=1. Raises RuntimeError on a configuration it cannot serve."""
    if _tt_public_getenv(ENV_PACK, "0") != "1":
        return False
    if not enabled():
        raise RuntimeError(f"{ENV_PACK}=1 needs {ENV}=1 (it is that path's read schedule)")
    if nb_enabled():
        raise RuntimeError(f"{ENV_PACK}=1 and {ENV_NB}=1 are mutually exclusive (PACK keeps blocking reads)")
    return True


def pack_layers() -> int:
    v = _tt_public_getenv(ENV_PACK_LAYERS, "").strip()
    n = int(v) if v else 8
    if not 1 <= n <= 48:
        raise ValueError(f"{ENV_PACK_LAYERS}={v}: must be in [1, 48]")
    return n


def pack_mustfail() -> str:
    return _tt_public_getenv(ENV_PACK_MUSTFAIL, "").strip().lower()


def _read_packed(ttnn, comp, parts, owned):
    """parts (device tensors, same shape past dim 0) -> one host tensor: concat on dim 0 on the device, one untilize when
    bf16 TILE (the _to_host rule), ONE blocking to_torch. owned: the parts are temporaries, freed here. A single part
    is read as is: ttnn.concat of one tensor may return that very tensor (the served state)."""
    t, t_owned = parts[0], owned
    try:
        if len(parts) > 1:
            t, t_owned = ttnn.concat(parts, dim=0), True
    finally:
        if owned and len(parts) > 1:
            for p in parts:
                ttnn.deallocate(p)
    try:
        if t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT:
            rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            if t_owned:
                ttnn.deallocate(t)
                t_owned = False
            try:
                return ttnn.to_torch(rm, mesh_composer=comp)
            finally:
                ttnn.deallocate(rm)
        return ttnn.to_torch(t, mesh_composer=comp)
    finally:
        if t_owned:
            ttnn.deallocate(t)


def _own(x):
    """A host tensor that owns its bytes (never a view that would keep the whole packed read alive)."""
    import torch

    return x.clone(memory_format=torch.contiguous_format)


def _pack_snapshot(ttnn, mesh_device, num_devices, dn_states, slots, wrong_slot, layers, swap):
    result = [[] for _ in slots]
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    n = len(slots)
    g0 = 0
    while g0 < len(dn_states):
        _, conv0, B, Nv, Dk, Dv = dn_states[g0]
        per_slot = 2 * n <= B
        step = layers if per_slot else max(1, layers // 2)
        group = dn_states[g0:g0 + step]
        g0 += len(group)
        K = len(conv0)
        if any((b, nv, dk, dv, len(cs)) != (B, Nv, Dk, Dv, K) for _, cs, b, nv, dk, dv in group):
            raise ValueError("QWEN36_PREFIX_SNAP_PACK: GDN layers of one group differ in shape")
        L = len(group)
        src = [((s + 1) % B) if wrong_slot else s for s in slots]
        # rec: [L * n] row slices (layer-major) or [L * B] whole rows, one read
        if per_slot:
            parts = []
            try:
                for rec_state, *_ in group:
                    for s in src:
                        parts.append(ttnn.slice(rec_state, (s, 0, 0, 0), (s + 1, Nv, Dk, Dv)))
            except BaseException:
                for p in parts:
                    ttnn.deallocate(p)
                raise
            host = _read_packed(ttnn, comp, parts, owned=True).reshape(num_devices, L * n, Nv, Dk, Dv)
            recs = [[host[:, li * n + j] for j in range(n)] for li in range(L)]
        else:
            host = _read_packed(ttnn, comp, [g[0] for g in group], owned=False).reshape(num_devices, L * B, Nv, Dk, Dv)
            recs = [[host[:, li * B + s] for s in src] for li in range(L)]
        # conv taps: [L * K, B, D] (layer-major, tap-minor), one read
        ch = _read_packed(ttnn, comp, [c for g in group for c in g[1]], owned=False)
        ch = ch.reshape(num_devices, L * K, *ch.shape[1:])
        convs = [[ch[:, li * K + k] for k in range(K)] for li in range(L)]
        if swap:  # MUSTFAIL: a host-split error the self-test must refuse
            if L >= 2:
                recs[0], recs[1] = recs[1], recs[0]
            else:
                convs[0][0], convs[0][1] = convs[0][1], convs[0][0]
        for li in range(L):
            for out, s, rec in zip(result, src, recs[li]):
                out.append((_own(rec), [_own(c[:, s:s + 1]) for c in convs[li]]))
    return result


def fast_snapshot(ttnn, mesh_device, num_devices, dn_states, slots, wrong_slot=False, nb=False, group_layers=8,
                  pack=False, pack_layers=8, pack_swap=False):
    """Same result as legacy_snapshot, reading only the wanted rows (few slots) and untilizing bf16 on the device.
    wrong_slot (MUSTFAIL): read row (s + 1) % B instead of s. nb (QWEN36_PREFIX_SNAP_NB): the reads of `group_layers`
    layers at a time are enqueued non-blocking and waited for once. pack (QWEN36_PREFIX_SNAP_PACK): the reads of
    `pack_layers` layers are concatenated on the device and read with two blocking reads; pack_swap is its MUSTFAIL."""
    if pack:
        if nb:
            raise ValueError("QWEN36_PREFIX_SNAP_PACK and QWEN36_PREFIX_SNAP_NB are mutually exclusive")
        if not 1 <= int(pack_layers) <= 48:
            raise ValueError(f"pack_layers={pack_layers}: must be in [1, 48]")
        return _pack_snapshot(ttnn, mesh_device, num_devices, dn_states, slots, wrong_slot, int(pack_layers),
                              pack_swap)
    result = [[] for _ in slots]
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    step = group_layers if nb else 1
    for g0 in range(0, len(dn_states), step):
        group = dn_states[g0:g0 + step]
        reads = _Reads(ttnn, mesh_device, comp, blocking=not nb)
        plan = []
        for rec_state, conv_states, B, Nv, Dk, Dv in group:
            src = [((s + 1) % B) if wrong_slot else s for s in slots]
            if 2 * len(slots) <= B:
                rec_idx = [reads.add(ttnn.slice(rec_state, (s, 0, 0, 0), (s + 1, Nv, Dk, Dv)), owned=True)
                           for s in src]
                whole = None
            else:
                rec_idx, whole = None, reads.add(rec_state)
            conv_idx = [reads.add(c) for c in conv_states]  # [num_devices, B, D] (the tile padding is dropped)
            plan.append((src, rec_idx, whole, conv_idx, B, Nv, Dk, Dv))
        host = reads.finish()
        for src, rec_idx, whole, conv_idx, B, Nv, Dk, Dv in plan:
            if rec_idx is not None:
                recs = [host[i].reshape(num_devices, Nv, Dk, Dv) for i in rec_idx]
            else:
                w = host[whole].reshape(num_devices, B, Nv, Dk, Dv)
                recs = [w[:, s].clone() for s in src]
            convs = [host[i] for i in conv_idx]
            for out, s, rec in zip(result, src, recs):
                out.append((rec.contiguous(), [c[:, s:s + 1].clone() for c in convs]))
    return result


def compare(a, b):
    """Bitwise equality of two snapshot lists (per slot, per layer, rec + conv taps). Returns (ok, detail)."""
    import torch

    if len(a) != len(b):
        return False, f"slot count {len(a)} vs {len(b)}"
    for si, (sa, sb) in enumerate(zip(a, b)):
        if len(sa) != len(sb):
            return False, f"slot {si}: layer count {len(sa)} vs {len(sb)}"
        for li, ((ra, ca), (rb, cb)) in enumerate(zip(sa, sb)):
            pairs = [(ra, rb)] + list(zip(ca, cb))
            if len(ca) != len(cb):
                return False, f"slot {si} layer {li}: conv tap count"
            for ti, (x, y) in enumerate(pairs):
                if x.shape != y.shape or x.dtype != y.dtype:
                    return False, f"slot {si} layer {li} tensor {ti}: {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}"
                if not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
                    return False, f"slot {si} layer {li} tensor {ti}: bytes differ"
    return True, "bitwise equal"


def selftest_compare(fast, legacy):
    """The served self-test on a real checkpoint: fast must equal legacy bitwise, and a one-bit flip of the legacy copy
    (the wrong-input branch, run every time) must be caught. Returns (ok, detail); ok=False refuses the step."""
    import torch

    ok, detail = compare(fast, legacy)
    if not ok:
        return False, f"fast vs legacy: {detail}"
    if not legacy or not legacy[0]:
        return False, "nothing to compare"
    flipped = [[(r.clone(), [c.clone() for c in cs]) for r, cs in slot] for slot in legacy]
    r0 = flipped[0][0][0]
    v = r0.view(-1).view(torch.uint8) if r0.is_contiguous() else None
    if v is None:
        return False, "non-contiguous legacy rec"
    v[0] ^= 1
    bad, _ = compare(fast, flipped)
    if bad:
        return False, "wrong-input branch passed (a flipped bit was not caught)"
    return True, "bitwise equal; wrong-input branch refused"


# ---- QWEN36_PREFIX_SCRATCH_RESUME ------------------------------------------------------------------------------------
# An align-mode cold prompt is prefilled in two steps: the head [0, floor(L / 2048) * 2048) and the tail. The tail step
# restores the head's GDN state from its HOST snapshot into the bound B=1 prefill scratch (48 layers x from_torch with a
# host tilize of rec [2, 24, 128, 128] + 4 conv taps + the conv carry) before replaying the chunk path from `start`.
# When no other row ran on the scratch since the head step's row ended, the scratch still holds exactly that state:
# rec_state is the very tensor the slot copy read (the snapshot is a byte copy of it), and conv_carry is what the
# snapshot's taps 1..K-1 were written from. The prefill chunk path reads only rec_state and conv_carry (conv_states are
# outputs: the finalize / the folded slot copy write them). So the restore is skipped for that row, and within a step
# that row is run first. Eligible: a continuation (hit < start: the state is the request's own live state, never a
# cached checkpoint of another chunking) whose (request id, start) equals the scratch's resident tag. The tag is set
# only after a non-final row completed its slot copy, and cleared before any row runs and by every other scratch bind.
ENV_RESUME = "QWEN36_PREFIX_SCRATCH_RESUME"
ENV_RESUME_MUSTFAIL = "QWEN36_PREFIX_SCRATCH_RESUME_MUSTFAIL"  # "stale": treat any continuation as resident


def scratch_resume_enabled() -> bool:
    return _tt_public_getenv(ENV_RESUME, "0") == "1"


def resume_mustfail() -> str:
    return _tt_public_getenv(ENV_RESUME_MUSTFAIL, "").strip().lower()


def resident_order(n, req_ids, starts, hits, resident):
    """Row order for one prefill step: the resident continuation (if any) first, the rest in their order.
    Returns (order, resident_row or None). resident: the scratch tag (req_id, end) or None."""
    row = None
    if resident is not None:
        for u in range(n):
            if req_ids[u] == resident[0] and int(starts[u]) == int(resident[1]) and int(hits[u]) < int(starts[u]):
                row = u
                break
    if row is None:
        return list(range(n)), None
    return [row] + [u for u in range(n) if u != row], row


def scratch_digest(ttnn, mesh_device, tensors):
    """Host bytes of the given device tensors (rec_state / conv_carry of every GDN layer), for the self-test."""
    import torch

    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    return [ttnn.to_torch(t, mesh_composer=comp).contiguous().view(-1).view(torch.uint8).clone() for t in tensors]


def digests_equal(a, b):
    import torch

    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def restore_rec(ttnn, mesh_device, mapper, rec, dst, wrong=False):
    """Upload one layer's host recurrent state rec [num_devices, Nv, Dk, Dv] into the device tensor dst (TILE) in place.
    bf16: row-major upload + device tilize (exact: tile-aligned, no padding); other dtypes: the legacy host tilize.
    wrong (MUSTFAIL): upload the state with its first two value heads swapped."""
    if wrong:
        rec = rec.clone()
        rec[:, [0, 1]] = rec[:, [1, 0]]
    if dst.dtype == ttnn.bfloat16 and tuple(rec.shape[-2:]) == (128, 128):
        rm = ttnn.from_torch(rec, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
                             mesh_mapper=mapper)
        t = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
    else:
        t = ttnn.from_torch(rec, dtype=dst.dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
    try:
        ttnn.copy(t, dst)
    finally:
        ttnn.deallocate(t)


# ---- QWEN36_PREFIX_CKPT_DEFER (cold-prefix-s10 lever L1) -------------------------------------------------------------
# The cold 4k decomposition (an internal record, w4c) charges the GDN checkpoint readback +0.26 s (B1) and
# +0.97 s (B8) of TTFT: the packed reads block the engine step that emits the first token. This lever takes the
# checkpoint OFF that path: inside the step each checkpointed slot's GDN state (per layer: the recurrent-state row, and
# the K conv-tap rows untilized) is COPIED on the device into a staging pool, and the pool is read back at the start of
# the NEXT engine step (the runner resolves it before consume / prepare / the host-tier stores, so every consumer of a
# checkpoint still sees it before it can need it). Same bytes: slice / untilize / copy / concat are data movement (the
# ttsim cells of bench/probes/sim_prefix_snap_fast.py ckpt_defer_* check staged == legacy bitwise, and that the staged
# copy survives the source being overwritten); the served self-test compares the first staged checkpoint of a boot
# against the immediate packed read of the same slot, with its wrong-input branch.
#
# The pool is allocated ONCE, before any trace capture (qwen36_vllm.warmup_model_prefill): a device buffer allocated
# after capture and kept alive across a trace replay can alias the replay's baked intermediates (the ttnn.clone hazard
# noted at prefill_traced_bucket_batched). Every temporary of stage() / read() is freed before it returns. A step with
# more checkpoints than free pool entries reads the rest immediately (the shipped path).
ENV_DEFER = "QWEN36_PREFIX_CKPT_DEFER"
ENV_DEFER_SLOTS = "QWEN36_PREFIX_CKPT_DEFER_SLOTS"  # pool entries (checkpointed slots in flight), default 4
ENV_DEFER_MUSTFAIL = "QWEN36_PREFIX_CKPT_DEFER_MUSTFAIL"  # "wrongslot": stage slot (s+1) % B -> the self-test refuses


def defer_enabled() -> bool:
    """QWEN36_PREFIX_CKPT_DEFER=1. Raises RuntimeError on a configuration it cannot serve."""
    if _tt_public_getenv(ENV_DEFER, "0") != "1":
        return False
    if not enabled():
        raise RuntimeError(f"{ENV_DEFER}=1 needs {ENV}=1")
    if not pack_enabled():
        raise RuntimeError(f"{ENV_DEFER}=1 needs {ENV_PACK}=1 (the pool is read with the packed reads)")
    return True


def defer_slots() -> int:
    v = _tt_public_getenv(ENV_DEFER_SLOTS, "").strip()
    n = int(v) if v else 4
    if not 1 <= n <= 64:
        raise ValueError(f"{ENV_DEFER_SLOTS}={v}: must be in [1, 64]")
    return n


def defer_mustfail() -> str:
    return _tt_public_getenv(ENV_DEFER_MUSTFAIL, "").strip().lower()


class StagedRef:
    """A checkpoint staged in the pool (entry `idx`), not yet read back. Stands in for a host snapshot until
    StagePool.read replaces it."""

    __slots__ = ("idx", "slot", "selftest_ref")

    def __init__(self, idx, slot, selftest_ref=None):
        self.idx, self.slot, self.selftest_ref = int(idx), int(slot), selftest_ref

    def __repr__(self):
        return f"StagedRef(idx={self.idx}, slot={self.slot})"


def is_staged(x) -> bool:
    return isinstance(x, StagedRef)


DRAM_BANKS = 8  # Blackhole P150a: 8 GDDR6 banks per chip (QWEN36_DRAM_LEDGER logs the allocator's num_banks)
DRAM_ALIGN = 64  # interleaved DRAM page alignment on Blackhole
TILE_BF16 = 32 * 32 * 2


def interleaved_bytes(pages: int, page_bytes: int, banks: int = DRAM_BANKS) -> int:
    """Device bytes one DRAM-interleaved buffer takes per card: the allocator reserves the same address range in
    every bank, ceil(pages / banks) aligned pages each. A one-page buffer therefore costs `banks` pages
    (2026-11-09: the conv taps are [1, 1, D] row-major = ONE 10,240 B page -> 81,920 B per card)."""
    page = -(-int(page_bytes) // DRAM_ALIGN) * DRAM_ALIGN
    return -(-int(pages) // int(banks)) * page * int(banks)


def entry_bytes(layers, banks: int = DRAM_BANKS) -> int:
    """Device bytes per card of one staging-pool entry as StagePool allocates it: per GDN layer one bf16 TILE
    recurrent state [Nv, Dk, Dv] (tile pages) plus K separate bf16 ROW_MAJOR conv buffers [1, 1, D] (one page each).
    The one formula StagePool and the KV sizer's reservation (kv_reserve_tokens) share."""
    out = 0
    for Nv, Dk, Dv, K, D in layers:
        tiles = int(Nv) * -(-int(Dk) // 32) * -(-int(Dv) // 32)
        out += interleaved_bytes(tiles, TILE_BF16, banks) + int(K) * interleaved_bytes(1, int(D) * 2, banks)
    return out


# bytes per element of a paged-KV dtype, keyed by every QWEN36_KV_DTYPE alias (tp_common.KV_DTYPE_ALIASES; a test
# keeps the key sets equal -- tp_common imports ttnn, this module must stay importable without it)
_KV_BPE = {"bf16": 2.0, "bfloat16": 2.0,
           "bf8": 1088 / 1024, "bfp8": 1088 / 1024, "bfp8_b": 1088 / 1024, "bfloat8_b": 1088 / 1024,
           "bf4": 576 / 1024, "bfp4": 576 / 1024, "bfp4_b": 576 / 1024, "bfloat4_b": 576 / 1024}
_RESERVED = {}  # set by kv_reserve_tokens in the worker process: what the KV sizer gave up, for the pool's check


def reserved() -> dict:
    """{'tokens', 'kv_bytes', 'pool_bytes'} per card that the KV sizer reserved for the pool ({} if it did not)."""
    return dict(_RESERVED)


def gdn_layer_shapes(cfg: dict, num_devices: int) -> list:
    """Per-card [(Nv, Dk, Dv, K, D)] of every GDN layer, from the HF text config (as gdn/tp.py derives them)."""
    nd = int(num_devices)
    nk, nv = int(cfg["linear_num_key_heads"]), int(cfg["linear_num_value_heads"])
    dk, dv, k = int(cfg["linear_key_head_dim"]), int(cfg["linear_value_head_dim"]), int(cfg["linear_conv_kernel_dim"])
    n = sum(1 for t in cfg["layer_types"] if t != "full_attention")
    return [(nv // nd, dk, dv, k, (2 * nk * dk + nv * dv) // nd)] * n


def kv_spec() -> tuple:
    """(K, V) alias names of the paged KV dtypes, read like tp_common.resolve_kv_dtypes(bf16). Raises on an alias
    tp_common would refuse."""
    spec = _tt_public_getenv("QWEN36_KV_DTYPE", "").strip().lower()
    if spec:
        parts = [x.strip() for x in spec.split("/")]
        parts = parts * 2 if len(parts) == 1 else parts
        if len(parts) != 2 or any(x not in _KV_BPE for x in parts):
            raise ValueError(f"QWEN36_KV_DTYPE={spec!r} is not '<K>/<V>' over {sorted(_KV_BPE)}")
        return parts[0], parts[1]
    return ("bf8", "bf8") if _tt_public_getenv("QWEN_SDPA_BF8", "0") == "1" else ("bf16", "bf16")


def kv_reserve_tokens(cfg: dict, num_devices: int, block: int = 2048) -> tuple:
    """QWEN36_PREFIX_CKPT_DEFER x the fixed KV pool (QWEN36_MAX_TOKENS_ALL_USERS): the staging pool is allocated at
    warmup, AFTER the KV cache, so without a reservation it eats the DRAM headroom the MTP warm tail needs (pp32k w1b
    2026-11-09: TT_FATAL OOM at max_model_len 262144, 8.0 MiB free per bank but the largest free block 3.75 MiB
    against a 5.0 MiB request -- fragmented headroom). Returns (tokens, pool_bytes, kv_bytes_per_token) per card: the
    pool's allocated bytes (entry_bytes, per-bank pages) as KV tokens, rounded UP to whole `block`s. Only the
    full-attention layers' paged K/V count as KV bytes (the MTP layer's KV shrinks with the same block count: the
    true reservation is larger, erring safe). 0 tokens when the lever is off."""
    _RESERVED.clear()
    if _tt_public_getenv(ENV_DEFER, "0") != "1":
        return 0, 0, 0
    nd = int(num_devices)
    pool = entry_bytes(gdn_layer_shapes(cfg, nd)) * defer_slots()
    kd, vd = kv_spec()
    kvh = max(1, int(cfg["num_key_value_heads"]) // nd)
    n_attn = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    hd = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))
    per_tok = n_attn * kvh * hd * (_KV_BPE[kd] + _KV_BPE[vd])
    blocks = -(-pool // int(per_tok * block))
    tok = int(blocks * block)
    _RESERVED.update(tokens=tok, kv_bytes=int(tok * per_tok), pool_bytes=int(pool))
    return tok, int(pool), float(per_tok)


class StagePool:
    """Pre-allocated device staging for GDN checkpoints. layers: [(Nv, Dk, Dv, K, D)] per GDN layer (per-card shapes);
    rec entries [1, Nv, Dk, Dv] bf16 TILE, conv entries [1, 1, D] bf16 ROW_MAJOR, sharded over the mesh on dim 0 like
    the live state (each card stages its own shard)."""

    def __init__(self, ttnn, mesh_device, num_devices, layers, entries, pack_layers=8):
        import torch

        self.ttnn, self.mesh, self.nd = ttnn, mesh_device, int(num_devices)
        self.layers = [tuple(int(v) for v in l) for l in layers]
        self.entries, self.pack_layers = int(entries), int(pack_layers)
        self.comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        self.rec, self.conv = [], []
        try:
            for _ in range(self.entries):
                recs, convs = [], []
                for Nv, Dk, Dv, K, D in self.layers:
                    recs.append(ttnn.from_torch(torch.zeros(self.nd, Nv, Dk, Dv, dtype=torch.bfloat16),
                                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                                mesh_mapper=shard))
                    convs.append([ttnn.from_torch(torch.zeros(self.nd, 1, D, dtype=torch.bfloat16),
                                                  dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                  device=mesh_device, mesh_mapper=shard) for _ in range(K)])
                self.rec.append(recs)
                self.conv.append(convs)
        except BaseException:
            self.free_all()
            raise
        self.free = list(range(self.entries))
        self.bytes_per_entry = entry_bytes(self.layers)  # per card

    def free_all(self):
        for e in self.rec:
            for t in e:
                self.ttnn.deallocate(t)
        for e in self.conv:
            for l in e:
                for t in l:
                    self.ttnn.deallocate(t)
        self.rec, self.conv, self.free = [], [], []

    def stage(self, dn_states, slot, wrong_slot=False):
        """Copy decode slot `slot` of every GDN layer into a free entry. Returns a StagedRef, or None when the pool is
        full (the caller reads immediately). dn_states: [(rec_state, conv_states, B, Nv, Dk, Dv)] in GDN layer order."""
        ttnn = self.ttnn
        if not self.free:
            return None
        if len(dn_states) != len(self.layers):
            raise ValueError(f"{ENV_DEFER}: {len(dn_states)} GDN layers vs a pool of {len(self.layers)}")
        j = self.free.pop(0)
        try:
            for li, (rec_state, conv_states, B, Nv, Dk, Dv) in enumerate(dn_states):
                if rec_state.dtype != ttnn.bfloat16 or (Nv, Dk, Dv, len(conv_states)) != self.layers[li][:4]:
                    raise ValueError(f"{ENV_DEFER}: layer {li} state {rec_state.dtype} {(Nv, Dk, Dv)} does not match "
                                     f"the pool (bf16 recurrent state only)")
                s = ((int(slot) + 1) % B) if wrong_slot else int(slot)
                t = ttnn.slice(rec_state, (s, 0, 0, 0), (s + 1, Nv, Dk, Dv))
                try:
                    ttnn.copy(t, self.rec[j][li])
                finally:
                    ttnn.deallocate(t)
                D = self.layers[li][4]
                for k, c in enumerate(conv_states):
                    rm = ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT) if c.layout == ttnn.TILE_LAYOUT else c
                    try:
                        r = ttnn.slice(rm, (0, s, 0), (1, s + 1, D))
                        try:
                            ttnn.copy(r, self.conv[j][li][k])
                        finally:
                            ttnn.deallocate(r)
                    finally:
                        if rm is not c:
                            ttnn.deallocate(rm)
        except BaseException:
            self.free.insert(0, j)
            raise
        return StagedRef(j, slot)

    def release(self, ref):
        if ref.idx in self.free:
            raise RuntimeError(f"{ENV_DEFER}: pool entry {ref.idx} released twice")
        self.free.append(ref.idx)

    def read(self, refs, swap=False):
        """Read staged entries back (packed: per group of pack_layers GDN layers, one rec read and one conv read for
        all refs), release them, and return one host snapshot per ref in the legacy format: per GDN layer
        (rec [num_devices, Nv, Dk, Dv], [K x conv [num_devices, 1, D]])."""
        ttnn, nd = self.ttnn, self.nd
        refs = list(refs)
        if not refs:
            return []
        if len({r.idx for r in refs}) != len(refs):
            raise RuntimeError(f"{ENV_DEFER}: duplicate pool entries in one read")
        n = len(refs)
        result = [[] for _ in refs]
        try:
            g0 = 0
            while g0 < len(self.layers):
                group = list(range(g0, min(g0 + self.pack_layers, len(self.layers))))
                g0 += len(group)
                Nv, Dk, Dv, K, D = self.layers[group[0]]
                if any(self.layers[l] != self.layers[group[0]] for l in group):
                    raise ValueError(f"{ENV_DEFER}: GDN layers of one group differ in shape")
                L = len(group)
                rec_parts = [self.rec[r.idx][l] for l in group for r in refs]  # layer-major, ref-minor
                host = _read_packed(ttnn, self.comp, rec_parts, owned=False).reshape(nd, L * n, Nv, Dk, Dv)
                conv_parts = [self.conv[r.idx][l][k] for l in group for r in refs for k in range(K)]
                ch = _read_packed(ttnn, self.comp, conv_parts, owned=False).reshape(nd, L * n * K, 1, D)
                for li in range(L):
                    for ri in range(n):
                        rec = host[:, li * n + ri]
                        convs = [ch[:, (li * n + ri) * K + k] for k in range(K)]
                        if swap and li == 0 and n >= 1 and K >= 2:  # MUSTFAIL for the ttsim cells: taps 0/1 swapped
                            convs[0], convs[1] = convs[1], convs[0]
                        result[ri].append((_own(rec), [_own(c) for c in convs]))
        finally:
            for r in refs:
                self.release(r)
        return result
