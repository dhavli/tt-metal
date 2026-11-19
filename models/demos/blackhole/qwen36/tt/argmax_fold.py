# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_ARGMAX_FOLD=1: the device greedy argmax combine (model.py _greedy_argmax_device) in 5 programs, was ~18.

Why (desk, docs/OP-DATASHEETS.md D7, [source validation record]): the vocab-sharded greedy pick runs
to_layout, argmax, gather, to_layout, typecast, to_layout, TWO all-gathers, 4 slices, gt, typecast, subtract, add,
multiply, add, typecast: ~18 tiny programs, each a launch and a DRAM round trip of one tile, and two CCL
handshakes. It runs 4x per decode cycle (the verify and the K=3 draft steps, QWEN36_MTP_DRAFT_VOCAB included).

What: to_layout and ttnn.argmax stay (the shard argmax itself). Then kernels/argmax_fold/argmax_fold_pack.cpp (one
core) reads each row's shard index and its logit straight from the row-major logits, and writes ONE [B, 2] float32
tile of raw bits (index, bf16 max << 16); ONE all-gather carries it; argmax_fold_combine.cpp (one core) applies the
served tie rule in integers (shard 1 wins only when its max is strictly greater as a float; NaN never wins) and
writes the [1, 1, B, 1] uint32 TILE ids the consumers take (mtp_batch commit, draft_vocab.map_ids). Same values:
the served fp32 chain is exact below 2^24, so integer arithmetic reproduces it bit for bit.

Applies to: every _greedy_argmax_device call whose (B, per) the boot self-test passed; else today's chain.

Self-test (fail closed, once per process at the MTP compile, before any trace): per (B, per) the server runs
(draft widths, verify widths M = T*Bw, per = vocab/TP and the draft vocab's N/TP), random sharded logits with
planted in-shard duplicate maxima and cross-shard exact ties, the fold against the served chain (served_reference,
the old code op for op), bitwise on every chip, plus the host torch argmax as a report. Wrong-input branch at every
boot: the fold with the must-fail (+1) MUST differ, else the boot is refused. QWEN36_ARGMAX_FOLD_MUSTFAIL=1 serves
the +1: the comparison differs and the boot is REFUSED.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_ARGMAX_FOLD"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")

SELFTEST_BANNER = "QWEN36_ARGMAX_FOLD self-test exact"
MAX_ROWS = 32
TILE = 32
F32_TILE = 4096
CHUNK = 64
CB_SCR = 0

_KDIR = Path(__file__).resolve().parent / "kernels" / "argmax_fold"
_SRC = {}
_tested = {}  # (B, per, ndev) -> "exact"
_logged = set()


def enabled():
    return _MODE == "1"


def mustfail():
    return _MF == "1"


def _src(name):
    if name not in _SRC:
        _SRC[name] = (_KDIR / name).read_text()
    return _SRC[name]


def _shape(t):
    return [int(x) for x in t.shape]


def _ndev(dev):
    return int(dev.get_num_devices()) if hasattr(dev, "get_num_devices") else 1


def key(B, per, ndev):
    return (int(B), int(per), int(ndev))


def usable(B, per, ndev):
    if not enabled():
        return False
    if key(B, per, ndev) in _tested:
        return True
    k = ("fallback", key(B, per, ndev))
    if k not in _logged:
        _logged.add(k)
        logger.warning(f"QWEN36_ARGMAX_FOLD fallback: B={B} per={per} not self-tested this process; today's combine chain serves it")
    return False


def _check(logits, B, per):
    if not 1 <= B <= MAX_ROWS:
        raise RuntimeError(f"{ENV}: B={B} outside 1..{MAX_ROWS}")
    s = _shape(logits)
    if len(s) != 4 or s[:3] != [1, 1, B] or s[3] != per or logits.dtype != ttnn.bfloat16:
        raise RuntimeError(f"{ENV}: logits {s} {logits.dtype}; expected [1, 1, {B}, {per}] bf16")
    if per % TILE:
        raise RuntimeError(f"{ENV}: per={per} must be a tile multiple (row pages are 64-byte multiples)")


def greedy(logits, per, mesh, tt_ccl, topology, mf=None):
    """Replicated [1, 1, B, 1] uint32 global greedy ids from the vocab-sharded [1, 1, B, per] bf16 logits.
    Deallocates logits (as the served chain does). Refuses a (B, per) the self-test did not pass (mf=None)."""
    B = int(logits.shape[2])
    per = int(per)
    ndev = _ndev(mesh)
    _check(logits, B, per)
    if mf is None:
        k = key(B, per, ndev)
        if k not in _tested:
            raise RuntimeError(f"{ENV}: (B={B}, per={per}) was not self-tested this process (tested: "
                               f"{sorted(_tested)}); refusing to serve it")
        mf = mustfail()
        if k not in _logged:
            _logged.add(k)
            logger.info(f"QWEN36_ARGMAX_FOLD engaged: B={B} per={per}: greedy combine in 5 programs (was ~18), one all-gather "
                        f"(was 2){' MUSTFAIL +1' if mf else ''}")
    return _run(logits, B, per, mesh, tt_ccl, topology, bool(mf))


def greedy_rm(rm, per, mesh, tt_ccl, topology, mf=None):
    """greedy() on logits that are ALREADY row-major (QWEN36_MTP_SPEC_SAMPLE_RMSHARE: the verify untilizes its
    logits once and hands the same tensor to the sampler's wide top-K and to this argmax).  The programs after
    the untilize are greedy()'s own (argmax, pack, one all-gather, combine) on the same row-major values, so the
    ids are the served ones bit for bit.  rm is NOT freed (the caller owns it).  Same refusal as greedy()."""
    B = int(rm.shape[2])
    per = int(per)
    ndev = _ndev(mesh)
    _check(rm, B, per)
    if rm.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"{ENV}: greedy_rm needs ROW_MAJOR logits; got {rm.layout}")
    if mf is None:
        k = key(B, per, ndev)
        if k not in _tested:
            raise RuntimeError(f"{ENV}: (B={B}, per={per}) was not self-tested this process (tested: "
                               f"{sorted(_tested)}); refusing to serve it")
        mf = mustfail()
    from models.tt_transformers.tt.ccl import tt_all_gather

    idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # [1,1,B,1] uint32, per shard (lowest index within the shard)
    packed = _pack(rm, idx, B, mesh)
    ttnn.deallocate(idx)
    g = tt_all_gather(packed, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    tok = _combine(g, B, per, ndev, mesh, bool(mf))
    if g is not packed:
        ttnn.deallocate(packed)
    ttnn.deallocate(g)
    return tok


def _one_core(dev):
    c = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])


def _kern(name, crs, ct, rt):
    return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                 core_ranges=crs, compile_time_args=ct, runtime_args=rt,
                                 config=ttnn.ReaderConfigDescriptor())


def _scr(crs, nbytes):
    return ttnn.CBDescriptor(total_size=nbytes, core_ranges=crs, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.float32, page_size=nbytes)])


def _pack(rm, idx, B, dev):
    crs = _one_core(dev)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, 2]), ttnn.float32, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    ix_page = int(idx.buffer_aligned_page_size())
    if ix_page > CHUNK:
        raise RuntimeError(f"{ENV}: index page {ix_page} > {CHUNK}")
    rm_page = int(rm.buffer_aligned_page_size())  # one logits row: the accessor's page geometry (bank interleave)
    if rm_page % CHUNK:
        raise RuntimeError(f"{ENV}: logits row page {rm_page} is not a {CHUNK}-byte multiple")
    ct = [CB_SCR, B, ix_page, rm_page] + acc(idx) + acc(rm) + acc(out)
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [idx.buffer_address(), rm.buffer_address(), out.buffer_address()]
    desc = ttnn.ProgramDescriptor(kernels=[_kern("argmax_fold_pack.cpp", crs, ct, rt)], semaphores=[],
                                  cbs=[_scr(crs, F32_TILE + 2 * B * CHUNK)])
    ttnn.generic_op([idx, rm, out], desc)
    return out


def _combine(g, B, per, ndev, dev, mf):
    crs = _one_core(dev)
    nt = int(g.padded_shape[-1]) // TILE
    if nt not in (1, ndev) or int(g.shape[-1]) != 2 * ndev:
        raise RuntimeError(f"{ENV}: gathered {list(g.shape)} padded {list(g.padded_shape)}: expected width {2 * ndev} "
                           f"in 1 or {ndev} tiles")
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, 1]), ttnn.uint32, ttnn.TILE_LAYOUT, dev,
                                         ttnn.DRAM_MEMORY_CONFIG)
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    ct = [CB_SCR, B, ndev, nt, per, 1 if mf else 0] + acc(g) + acc(out)
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [g.buffer_address(), out.buffer_address()]
    desc = ttnn.ProgramDescriptor(kernels=[_kern("argmax_fold_combine.cpp", crs, ct, rt)], semaphores=[],
                                  cbs=[_scr(crs, (nt + 1) * F32_TILE)])
    ttnn.generic_op([g, out], desc)
    return out


def _run(logits, B, per, mesh, tt_ccl, topology, mf):
    from models.tt_transformers.tt.ccl import tt_all_gather

    ndev = _ndev(mesh)
    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # [1,1,B,1] uint32, per shard (lowest index within the shard)
    packed = _pack(rm, idx, B, mesh)
    ttnn.deallocate(rm)
    ttnn.deallocate(idx)
    g = tt_all_gather(packed, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    tok = _combine(g, B, per, ndev, mesh, mf)
    if g is not packed:
        ttnn.deallocate(packed)
    ttnn.deallocate(g)
    ttnn.deallocate(logits)
    return tok


# ---------------------------------------------------------------------------------------------- reference + self-test
def served_reference(logits, per, mesh, tt_ccl, topology):
    """model.py _greedy_argmax_device, op for op (the flag-off path). Deallocates logits."""
    from models.tt_transformers.tt.ccl import tt_all_gather

    B = logits.shape[2]
    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(rm, dim=-1, keepdim=True)
    val = ttnn.gather(rm, dim=-1, index=idx)
    idx_f = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
    val_t = ttnn.to_layout(val, ttnn.TILE_LAYOUT)
    gi = tt_all_gather(idx_f, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.float32)
    gv = tt_all_gather(val_t, mesh, tt_ccl, cluster_axis=None, dim=3, topology=topology, dtype=ttnn.bfloat16)
    i0 = ttnn.slice(gi, [0, 0, 0, 0], [1, 1, B, 1]); i1 = ttnn.slice(gi, [0, 0, 0, 1], [1, 1, B, 2])  # noqa: E702
    v0 = ttnn.slice(gv, [0, 0, 0, 0], [1, 1, B, 1]); v1 = ttnn.slice(gv, [0, 0, 0, 1], [1, 1, B, 2])  # noqa: E702
    m = ttnn.typecast(ttnn.gt(v1, v0), ttnn.float32)
    tok = ttnn.add(i0, ttnn.multiply(m, ttnn.add(ttnn.subtract(i1, i0), float(per))))
    tok_u = ttnn.typecast(tok, ttnn.uint32)
    for x in (rm, idx, val, idx_f, val_t, gi, gv, i0, i1, v0, v1, m, tok):
        ttnn.deallocate(x)
    ttnn.deallocate(logits)
    return tok_u


def plant(x, B, per, ndev, g):
    """In place on torch x [1,1,B,ndev*per] (bf16): row 0 a cross-shard exact tie (lowest global index must win),
    row 1 (B >= 2) a 3-way in-shard tie in shard ndev-1, row 2 (B >= 3) a strict shard-1 win by one bf16 ulp."""
    import torch

    big = 40.0
    if ndev > 1:
        j = int(torch.randint(0, per, (1,), generator=g))
        for d in range(ndev):
            x[0, 0, 0, d * per + j] = big
    if B >= 2:
        d = ndev - 1
        js = sorted(int(v) for v in torch.randperm(per, generator=g)[:3])
        for j in js:
            x[0, 0, 1, d * per + j] = big
    if B >= 3 and ndev > 1:
        j0 = int(torch.randint(0, per, (1,), generator=g))
        j1 = int(torch.randint(0, per, (1,), generator=g))
        x[0, 0, 2, j0] = big
        x[0, 0, 2, per + j1] = torch.tensor(big, dtype=torch.bfloat16) + torch.finfo(torch.bfloat16).eps * big
    return x


def _to_rows(mesh, t, B):
    chips = ttnn.get_device_tensors(t) if _ndev(mesh) > 1 else [t]
    return [ttnn.to_torch(c).reshape(-1)[:B].to(__import__("torch").int64) for c in chips]


def selftest(mesh, tt_ccl, topology, cells, seed=4242):
    """Fold vs served_reference per (B, per) on sharded random logits with planted ties, bitwise on every chip; the
    served MUSTFAIL honoured as the candidate; the explicit must-fail MUST differ. Returns
    {f"B{B}_P{per}": "exact" | "DIFF n", f"..._mustfail": n (> 0), f"..._torch": n (report only)}."""
    import torch

    g = torch.Generator().manual_seed(seed)
    ndev = _ndev(mesh)
    mapper = {"mesh_mapper": ttnn.ShardTensorToMesh(mesh, dim=3)} if ndev > 1 else {}
    res = {}
    for B, per in sorted(set((int(b), int(p)) for b, p in cells)):
        x = (torch.randn(1, 1, B, ndev * per, generator=g) * 4).to(torch.bfloat16)
        plant(x, B, per, ndev, g)
        host = torch.argmax(x.float(), dim=-1).reshape(-1)

        def dev_logits():
            return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, **mapper)

        ref = served_reference(dev_logits(), per, mesh, tt_ccl, topology)
        cand = greedy(dev_logits(), per, mesh, tt_ccl, topology, mf=mustfail())
        bad = greedy(dev_logits(), per, mesh, tt_ccl, topology, mf=True)
        r, c, w = _to_rows(mesh, ref, B), _to_rows(mesh, cand, B), _to_rows(mesh, bad, B)
        n = sum(int((a != b).sum()) for a, b in zip(r, c))
        nm = sum(int((a != b).sum()) for a, b in zip(r, w))
        nt = sum(int((a != host).sum()) for a in r)
        tag = f"B{B}_P{per}"
        res[tag] = "exact" if n == 0 else f"DIFF {n}"
        res[tag + "_mustfail"] = nm
        res[tag + "_torch"] = nt
        for t in (ref, cand, bad):
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return all(v == "exact" for k, v in res.items() if not (k.endswith("_mustfail") or k.endswith("_torch"))) and \
        all(v > 0 for k, v in res.items() if k.endswith("_mustfail"))


def selftest_cells(widths, pers):
    return sorted({(int(b), int(p)) for b in widths for p in pers if 1 <= int(b) <= MAX_ROWS})


def plain_widths(B_max):
    """The plain decode widths (mtp_batch_host.bucket_for / qwen36_vllm.decode_forward: the smallest power of two >= the
    active prefix, or the full width): every power of two below B_max, plus B_max."""
    B_max = int(B_max)
    out, w = {B_max}, 1
    while w < B_max:
        out.add(w)
        w *= 2
    return sorted(out)


def served_cells(B_max, vocab_per, draft_per=None, served=()):
    """Every (B, per) _greedy_argmax_device runs in a served boot, from the call sites:
      - model.ttnn_decode_forward (plain decode, QWEN36_DEVICE_ARGMAX, incl. the compile warmup's buckets): B in
        plain_widths(B_max), per = vocab/TP;
      - mtp_batch verify (_verify_ops, M = T*Bw rows): B = T*Bw for every served (T, Bw) (mtp_batch_host
        .served_verify_cells), per = vocab/TP;
      - the K draft steps (_commit_draft_ops -> _MTPHead.forward at Bw rows): B = Bw per served bucket, per = the draft
        vocab's N/TP (mtp_draft_vocab.draft_ids) when QWEN36_MTP_DRAFT_VOCAB is set, else vocab/TP.
    The 10-16 card boot refused because only the MTP widths were self-tested and the plain-decode warmup (B=1, 2, 4, 8
    at vocab/TP) ran first: this is the set the boot must cover, before any decode."""
    vocab_per = int(vocab_per)
    widths = set(plain_widths(B_max))
    cells = set()
    for T, Bw in served:
        widths.add(int(T) * int(Bw))
        cells.add((int(Bw), int(draft_per) if draft_per else vocab_per))
        cells.add((int(Bw), vocab_per))  # mtp_batch.py:232 (no draft vocab) / a plain step at that width
    cells |= {(b, vocab_per) for b in widths}
    return sorted(c for c in cells if 1 <= c[0] <= MAX_ROWS)


def ensure_selftest(mesh, tt_ccl, topology, widths=(), pers=(), cells=None):
    """Once per process per (B, per): fail closed, record the cells. cells = served_cells(...) (the served set) or
    widths x pers."""
    if not enabled():
        return
    ndev = _ndev(mesh)
    cells = selftest_cells(widths, pers) if cells is None else sorted(set((int(b), int(p)) for b, p in cells))
    todo = [c for c in cells if key(c[0], c[1], ndev) not in _tested]
    if not todo:
        return
    res = selftest(mesh, tt_ccl, topology, todo)
    if not selftest_ok(res):
        raise RuntimeError(f"QWEN36_ARGMAX_FOLD self-test REFUSED{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}")
    for B, per in todo:
        _tested[key(B, per, ndev)] = res[f"B{B}_P{per}"]
    logger.info(f"{SELFTEST_BANNER}: cells (B, per) {todo} on {ndev} chip(s), bitwise vs the served combine chain "
                f"(planted in-shard and cross-shard ties); wrong-input branch differs")
