# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFILL_RS_PKT4=1: the 1x2 (Linear) reduce-scatter writer sends up to 4 tiles per fabric packet, not 2.

Why: the prefill MLP down-proj reduce-scatter (reduce_scatter_minimal_async, line topology, QWEN36_PREFILL_RS_CFG
2,10,4,2) costs ~250 us/call, 16.0 ms/chunk over 64 layers at 32k ([source validation record]),
while it moves ~5.6 MB per chip over 2 links. The line writer packs at most 2 tiles per packet (host
max_scatter_write_pages = 2): 2.2 KB packets at bf8, the size where tt-fabric runs at ~19 GB/s per link against
37-46 GB/s at 4-15 KB (third-party P150x4 golden table, docs/PREFILL-KERNEL-DEEPDIVE.md). The c69 sweep is
link-bound (2 links ~1.8x 1 link), and bf8 (half the bytes, the same packet count) cut it only 303 -> ~250 us:
a packet-rate wall. ESTIMATE, not a measurement.

What: a kernel-only overlay of line_reduce_scatter_minimal_async_writer.cpp (bench/probes/rs_pkt4/line_writer.pkt4.cpp,
mounted by the arm over /opt/tt-metal-src): 3- and 4-chunk scatter writes (the ring writer's maximum), clamped to
the mux channel buffer. Transport only; every tile lands at the same remote address as before, so the RS output is
bit-identical by construction (the ttsim gate checks it against the stock kernel on random inputs).

Identity: the flag and the mounted kernel must agree. Flag on with the stock kernel, or the pkt4 kernel without the
flag, REFUSES the boot (check()). Banner "QWEN36_PREFILL_RS_PKT4 engaged".

Self-test (fail closed, ensure_selftest, eager, before any trace, both chips): the served prefill_all_reduce on a
per-chip partial pair (A, 0) and (0, B), bf8 and bf16, at T = 32 / 128 / 160 / chunk. The zero side makes the sum
exact, so each chip's output must equal the sender's tiles bitwise: a pure transport check of every packet shape
(4-tile, 3/2/1-tile tails). Wrong-input branch at every boot: the same output against the other half MUST differ.
The must-fail kernel (tiles 1 and 2 swapped in 3/4-tile packets) is a ttsim-only rung (bench/probes/rs_pkt4).

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import hashlib
import os

import torch
import ttnn
from loguru import logger

ENV = "QWEN36_PREFILL_RS_PKT4"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")

KERNEL_REL = ("ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
              "line_reduce_scatter_minimal_async_writer.cpp")
MARKER = "#define QWEN36_PREFILL_RS_PKT4_KERNEL 1"
MF_MARKER = "constexpr bool pkt4_mustfail = true;"
SELFTEST_BANNER = "QWEN36_PREFILL_RS_PKT4 self-test exact"
_state = {"checked": None, "tested": set()}


def enabled():
    return _MODE == "1"


def _kernel_paths():
    roots = []
    for r in (_tt_public_getenv("TT_METAL_HOME"), "/opt/tt-metal-src"):
        if r and r not in roots:
            roots.append(r)
    return [os.path.join(r, KERNEL_REL) for r in roots]


def check():
    """Flag <-> mounted kernel. Returns the kernel sha256[:12] (or None when off and stock). Raises on a mismatch."""
    if _state["checked"] is not None:
        return _state["checked"] or None
    found = [(p, open(p, "rb").read()) for p in _kernel_paths() if os.path.exists(p)]
    if not found:
        if enabled():
            raise RuntimeError(f"{ENV}=1 but no line RS writer kernel at {_kernel_paths()}: REFUSED")
        _state["checked"] = ""
        return None
    marked = [p for p, b in found if MARKER.encode() in b]
    if enabled() and len(marked) != len(found):
        raise RuntimeError(f"{ENV}=1 but the mounted line RS writer is not the pkt4 kernel "
                           f"({[p for p, _ in found if p not in marked]}): the arm must mount "
                           f"bench/probes/rs_pkt4/line_writer.pkt4.cpp. REFUSED")
    if not enabled() and marked:
        raise RuntimeError(f"the pkt4 line RS writer is mounted ({marked}) but {ENV} is unset: an arm that does not "
                           f"say what it serves. REFUSED")
    if not enabled():
        _state["checked"] = ""
        return None
    body = found[0][1]
    sha = hashlib.sha256(body).hexdigest()[:12]
    mf = MF_MARKER.encode() in body
    logger.info(f"{ENV} engaged: line reduce-scatter writer packs up to 4 tiles per fabric packet (kernel "
                f"{found[0][0]} sha256={sha}{', MUSTFAIL build' if mf else ''})")
    _state["checked"] = sha
    return sha


def _run_rs(mesh, tt_ccl, topology, host, dtype):
    """host: [2, 1, T, N] torch, one partial per chip -> (input as the device holds it, output), both [2, 1, T, *]."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    x = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    xin = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    out = tpc.prefill_all_reduce(x, mesh, tt_ccl, topology)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    ttnn.deallocate(out)
    return xin, got


def selftest(mesh, tt_ccl, topology, Ts, N, dtypes=(ttnn.bfloat8_b, ttnn.bfloat16)):
    """Per (T, dtype, sender): exact = every element of the receiving chip's output equals the sender's tile;
    wrong = the same output against the sender's OTHER half (must differ)."""
    res = []
    g = torch.Generator().manual_seed(20261020)
    half = N // 2
    for T in Ts:
        for dt in dtypes:
            for sender in (0, 1):
                a = torch.randn(1, 1, T, N, generator=g) * 4
                z = torch.zeros(1, 1, T, N)
                host = torch.cat([a, z] if sender == 0 else [z, a], dim=0)
                xin, got = _run_rs(mesh, tt_ccl, topology, host, dt)
                src = xin[sender]
                recv = 1 - sender
                exp_recv = src[..., recv * half:(recv + 1) * half]  # the half the OTHER chip owns, sent over fabric
                exp_own = src[..., sender * half:(sender + 1) * half]  # the sender's own half + the peer's zeros
                wrong = src[..., sender * half:(sender + 1) * half]  # the half that never crossed the link
                d_recv = int((got[recv] != exp_recv).sum())
                d_own = int((got[sender] != exp_own).sum())
                d_wrong = int((got[recv] != wrong).sum())
                res.append({"T": T, "dtype": str(dt), "sender": sender, "diff_recv": d_recv, "diff_own": d_own,
                            "wrong_differs": d_wrong, "ok": d_recv == 0 and d_own == 0 and d_wrong > 0})
    return res


def ensure_selftest(mesh, tt_ccl, topology, Ts, N):
    """Once per process per T: fail closed. Logs the engaged banner first (check())."""
    if not enabled():
        check()  # still refuses a pkt4 kernel mounted without the flag
        return
    check()
    todo = sorted({int(t) for t in Ts if int(t) % 32 == 0 and int(t) not in _state["tested"]})
    if not todo:
        return
    res = selftest(mesh, tt_ccl, topology, todo, int(N))
    bad = [r for r in res if not r["ok"]]
    if bad:
        raise RuntimeError(f"{SELFTEST_BANNER.replace(' exact', '')} REFUSED: {bad}")
    _state["tested"].update(todo)
    logger.info(f"{SELFTEST_BANNER}: {len(res)} cells T={todo} N={N} bf8+bf16 both senders, every received tile "
                f"bitwise equal to the sender's on both chips; wrong-input branch (the other half) differs in every "
                f"cell")
