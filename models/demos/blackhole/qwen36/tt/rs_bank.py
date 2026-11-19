# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_PREFILL_RS_BANK=1: the 1x2 (Linear) reduce-scatter sends BANK-CONTIGUOUS fabric packets (8 bf8 tiles each).

Why: the prefill MLP down-proj reduce-scatter (reduce_scatter_minimal_async, line, QWEN36_PREFILL_RS_CFG 2,10,4,2)
costs ~16 ms per 32k chunk (opmark S3, [source validation record]). The static anatomy
([source validation record], branch anatomy/rs-line) puts the bound on the per-packet
cost of each fabric link: 1 link x 4 workers -> 2 links x 2 workers = 1.75x, 4x fewer read rounds only -9..-11%,
rs4 (2 -> 4 tiles per packet) -22%/call. The rs4 writer still sends 4 one-tile scatter chunks per packet (640
packets and 2560 remote NoC writes per link per call at T=2048), because consecutive tiles of the DRAM-INTERLEAVED
intermediate live in different banks.

What: kernel-only overlays of the line RS reader AND writer (bench/probes/rs_bank/line_reader.bank.cpp,
line_writer.bank.cpp, mounted by the arm over /opt/tt-metal-src). Inside one slice row the columns c, c+8, ...,
c+72 sit in ONE DRAM bank at consecutive per-bank pages (InterleavedAddrGen: bank = id mod NUM_DRAM_BANKS, page
offset = id div NUM_DRAM_BANKS; dataflow_api_addrgen.h:18-41, 292-297). The sender reader fills its CB in that bank
order inside each sync group (the tiles between two out_ready increments, so the reducer still sees the stock set
when its semaphore wait returns); the writer recomputes every destination with the stock address generator and
merges only address-contiguous tiles: one packet = up to 4 contiguous chunks, at most (mux channel buffer - header)
/ page tiles (8 bf8 tiles = 8704 B at QWEN36_FABRIC_PAYLOAD=8704). Served T=2048: 320 packets and 512 remote NoC
writes per link per call (host model, bench/probes/rs_bank/host_order_test.sh). Transport only, tier 1: every tile
lands at the same remote address as before; the reduction is untouched. It REPLACES the rs4 writer overlay, so it
refuses QWEN36_PREFILL_RS_PKT4=1.

Identity: the flag and BOTH mounted kernels must agree (markers + the shared bank-order version). Flag on with a
stock/rs4 kernel, a bank kernel without the flag, or a reader/writer version mismatch REFUSES the boot (check()).
Banner "QWEN36_PREFILL_RS_BANK engaged".

Self-test (fail closed, ensure_selftest, eager, before any trace, both chips): the served prefill_all_reduce on a
per-chip partial pair (A, 0) and (0, B), bf8 and bf16, at T = 32 / 128 / 160 / chunk. The zero side makes the sum
exact, so each chip's output must equal the sender's tiles bitwise: a pure transport check of every packet shape
(8-tile unicast, 2..4-chunk scatter, ragged groups). Wrong-input branch at every boot: the same output against the
other half MUST differ. The must-fail writer (line_writer.bankmf.cpp: merges tiles regardless of their address, i.e.
it corrupts the bank mapping) is a ttsim-only rung (bench/probes/rs_bank).

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import hashlib
import os
import re

import torch
import ttnn
from loguru import logger

ENV = "QWEN36_PREFILL_RS_BANK"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")

_KDIR = "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
WRITER_REL = _KDIR + "line_reduce_scatter_minimal_async_writer.cpp"
READER_REL = _KDIR + "line_reduce_scatter_minimal_async_reader.cpp"
WRITER_MARKER = "#define QWEN36_PREFILL_RS_BANK_KERNEL 1"
READER_MARKER = "#define QWEN36_PREFILL_RS_BANK_READER 1"
_VERSION_RE = re.compile(rb"#define QWEN36_PREFILL_RS_BANK_ORDER_VERSION (\d+)")
MF_MARKER = "constexpr bool rs_bank_mustfail = true;"
SELFTEST_BANNER = "QWEN36_PREFILL_RS_BANK self-test exact"
_state = {"checked": None, "tested": set()}


def enabled():
    return _MODE == "1"


def _roots():
    roots = []
    for r in (_tt_public_getenv("TT_METAL_HOME"), _tt_public_getenv("TT_METAL_RUNTIME_ROOT"), "/opt/tt-metal-src"):
        if r and r not in roots:
            roots.append(r)
    return roots


def _found(rel):
    return [(p, open(p, "rb").read()) for p in (os.path.join(r, rel) for r in _roots()) if os.path.exists(p)]


def check():
    """Flag <-> mounted reader+writer. Returns the kernels' sha256[:12] pair (or None when off and stock). Raises on
    any mismatch."""
    if _state["checked"] is not None:
        return _state["checked"] or None
    writers, readers = _found(WRITER_REL), _found(READER_REL)
    w_marked = [p for p, b in writers if WRITER_MARKER.encode() in b]
    r_marked = [p for p, b in readers if READER_MARKER.encode() in b]
    if not enabled():
        if w_marked or r_marked:
            raise RuntimeError(f"the rs_bank line RS kernels are mounted ({w_marked + r_marked}) but {ENV} is unset: an "
                               f"arm that does not say what it serves. REFUSED")
        _state["checked"] = ""
        return None
    if _tt_public_getenv("QWEN36_PREFILL_RS_PKT4", "0") != "0":
        raise RuntimeError(f"{ENV}=1 replaces the rs4 writer overlay; QWEN36_PREFILL_RS_PKT4 must be 0. REFUSED")
    if not writers or not readers:
        raise RuntimeError(f"{ENV}=1 but no line RS reader/writer kernel under {_roots()}: REFUSED")
    if len(w_marked) != len(writers) or len(r_marked) != len(readers):
        raise RuntimeError(f"{ENV}=1 but the mounted line RS kernels are not BOTH the rs_bank overlays "
                           f"(writer marked {len(w_marked)}/{len(writers)}, reader marked {len(r_marked)}/{len(readers)}): "
                           f"the arm must mount bench/probes/rs_bank/line_writer.bank.cpp and line_reader.bank.cpp. "
                           f"REFUSED")
    versions = {m.group(1) for _, b in writers + readers for m in [_VERSION_RE.search(b)] if m}
    if len(versions) != 1 or any(_VERSION_RE.search(b) is None for _, b in writers + readers):
        raise RuntimeError(f"{ENV}=1 but the reader and writer bank-order versions disagree ({versions}): REFUSED")
    wsha = hashlib.sha256(writers[0][1]).hexdigest()[:12]
    rsha = hashlib.sha256(readers[0][1]).hexdigest()[:12]
    mf = MF_MARKER.encode() in writers[0][1]
    logger.info(f"{ENV} engaged: line reduce-scatter sends bank-contiguous fabric packets (order v{versions.pop().decode()}, "
                f"writer {writers[0][0]} sha256={wsha}, reader sha256={rsha}"
                f"{', QWEN36_PREFILL_RS_BANK_MUSTFAIL build' if mf else ''})")
    _state["checked"] = (wsha, rsha)
    return _state["checked"]


def _served_rs(x, mesh, tt_ccl, topology):
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    return tpc.prefill_all_reduce(x, mesh, tt_ccl, topology)


def _run_rs(mesh, tt_ccl, topology, host, dtype, rs_fn=None):
    """host: [2, 1, T, N] torch, one partial per chip -> (input as the device holds it, output), both [2, 1, T, *]."""
    rs_fn = rs_fn or _served_rs
    x = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    xin = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    out = rs_fn(x, mesh, tt_ccl, topology)
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    ttnn.deallocate(out)
    return xin, got


def selftest(mesh, tt_ccl, topology, Ts, N, dtypes=(ttnn.bfloat8_b, ttnn.bfloat16), rs_fn=None):
    """Per (T, dtype, sender): exact = every element of the receiving chip's output equals the sender's tile;
    wrong = the same output against the sender's OTHER half (must differ)."""
    res = []
    g = torch.Generator().manual_seed(20261021)
    half = N // 2
    for T in Ts:
        for dt in dtypes:
            for sender in (0, 1):
                a = torch.randn(1, 1, T, N, generator=g) * 4
                z = torch.zeros(1, 1, T, N)
                host = torch.cat([a, z] if sender == 0 else [z, a], dim=0)
                xin, got = _run_rs(mesh, tt_ccl, topology, host, dt, rs_fn=rs_fn)
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


def ensure_selftest(mesh, tt_ccl, topology, Ts, N, rs_fn=None):
    """Once per process per T: fail closed. Logs the engaged banner first (check())."""
    check()  # off: still refuses rs_bank kernels mounted without the flag
    if not enabled():
        return
    todo = sorted({int(t) for t in Ts if int(t) % 32 == 0 and int(t) not in _state["tested"]})
    if not todo:
        return
    res = selftest(mesh, tt_ccl, topology, todo, int(N), rs_fn=rs_fn)
    bad = [r for r in res if not r["ok"]]
    if bad:
        raise RuntimeError(f"{SELFTEST_BANNER.replace(' exact', '')} REFUSED: {bad}")
    _state["tested"].update(todo)
    logger.info(f"{SELFTEST_BANNER}: {len(res)} cells T={todo} N={N} bf8+bf16 both senders, every received tile "
                f"bitwise equal to the sender's on both chips; wrong-input branch (the other half) differs in every "
                f"cell")
