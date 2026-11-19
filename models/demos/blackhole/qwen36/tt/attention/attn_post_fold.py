# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_ATTN_POST_FOLD=1: the decode attention post chain (sigmoid gate, multiply, concat heads) in ONE program.

Why (desk, [source validation record] + docs/OP-DATASHEETS.md D4): after the decode SDPA every
attention layer runs six small programs on the [1, B, NH, HD] output: ttnn.sigmoid(gate), ttnn.multiply,
to_memory_config (height reshard, one user per core), nlp_concat_heads_decode, sharded_to_interleaved and (B < 32) a
slice, before the wo matmul. Each is a launch plus an L1/DRAM round trip of the same 8 tiles per user. 16 attention
layers per verify and per draft step: 96 programs per forward.

What: one ttnn.generic_op on B cores (core b = batch row b). Reader: the row's 8 attn tiles (L1) and 8 gate tiles
(DRAM). Compute (kernels/attn_post_fold/attn_post_fold_compute.cpp): sigmoid_tile on the gate tile (the same LLK
ttnn.sigmoid runs: accurate mode, HiFi4, bf16 DEST), packed to bf16, then mul_tiles(attn, sig) (the FPU multiply
binary_ng runs for bf16 x bf16, HiFi4, bf16 DEST), packed to bf16. Writer: the 12 head rows of the gated tiles
scattered as 32-byte face-row segments straight into the [1, 1, B, NH*HD] bf16 TILE output (L1 interleaved):
chunk (tile t, face f, head h) of user b -> output tile h*8 + t, face (b >= 16) * 2 + f, row b % 16. That is the
byte move nlp_concat_heads_decode's reader does, without the reshard and the sharded output. No untilize.

Same bytes: the two math ops use the same LLK calls, fidelity and DEST mode as the ttnn ops; the rest is a copy.
Rows >= B of the output tiles are left as allocated (the served path's nlp_concat_heads_decode leaves them too:
its reader only writes q < batch); nothing downstream reduces across rows.

Applies to: _decode_grouped_post (MTP verify, width B = T*Bw) and the plain decode path's nlp branch, whenever
attn_out and gate are [1, B, NH, HD] bf16 TILE interleaved and the width was self-tested. Else today's ops.

Self-test (fail closed, once per process at set_paged_kv_cache, before any trace): for every width B the server can
run (1, 2, 4, 8, max batch, every served verify width T*Bw <= 32), random attn (L1) and gate (DRAM), the fold against
the served six-op chain, bitwise on rows < B on every chip. Wrong-input branch at every boot: the fold with the
must-fail head rotation (head h written to slot h+1) MUST differ, else the boot is refused.
QWEN36_ATTN_POST_FOLD_MUSTFAIL=1 serves the rotation: the comparison differs and the boot is REFUSED.

NO ttsim number is a performance number.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
from pathlib import Path

import ttnn
from loguru import logger

ENV = "QWEN36_ATTN_POST_FOLD"
MUSTFAIL_ENV = ENV + "_MUSTFAIL"
_MODE = _tt_public_getenv(ENV, "0")
if _MODE not in ("0", "1"):
    raise RuntimeError(f"{ENV} must be 0 or 1; got {_MODE!r}")
_MF = _tt_public_getenv(MUSTFAIL_ENV, "0")
if _MF not in ("0", "1"):
    raise RuntimeError(f"{MUSTFAIL_ENV} must be 0 or 1; got {_MF!r}")

SELFTEST_BANNER = "QWEN36_ATTN_POST_FOLD self-test exact"
MAX_ROWS = 32  # one tile row: the decode width cap (M = T*Bw <= 32)
TILE = 32
BF16_TILE = 2048

_KDIR = Path(__file__).resolve().parent.parent / "kernels" / "attn_post_fold"
_SRC = {}
CB_ATTN, CB_GATE, CB_SIG, CB_OUT = 0, 1, 2, 16

_tested = {}  # (B, NH, HD) -> "exact"
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


def key(B, NH, HD):
    return (int(B), int(NH), int(HD))


def _check(attn, gate, B, NH, HD):
    if not 1 <= B <= MAX_ROWS:
        raise RuntimeError(f"{ENV}: B={B} outside 1..{MAX_ROWS}")
    if HD % TILE or NH > TILE:
        raise RuntimeError(f"{ENV}: NH={NH} HD={HD}: heads must fit one tile row and HD be a tile multiple")
    for n, t in (("attn_out", attn), ("gate", gate)):
        if _shape(t) != [1, B, NH, HD] or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{ENV}: {n} {_shape(t)} {t.dtype} {t.layout}; expected [1, {B}, {NH}, {HD}] bf16 TILE")
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise RuntimeError(f"{ENV}: {n} must be interleaved, got {t.memory_config()}")


def usable(attn, gate, B, NH, HD):
    """True when the flag is on, the operands have the fold's shape and the boot self-test passed this width."""
    if not enabled():
        return False
    try:
        _check(attn, gate, B, NH, HD)
    except RuntimeError as e:
        k = ("shape", str(e))
        if k not in _logged:
            _logged.add(k)
            logger.warning(f"QWEN36_ATTN_POST_FOLD fallback: {e}; today's ops serve it")
        return False
    if key(B, NH, HD) in _tested:
        return True
    k = ("fallback", key(B, NH, HD))
    if k not in _logged:
        _logged.add(k)
        logger.warning(f"QWEN36_ATTN_POST_FOLD fallback: B={B} not self-tested this process; today's sigmoid/multiply/concat-heads serve it")
    return False


def post(attn, gate, B, NH, HD, mf=None, memory_config=None):
    """sigmoid(gate) * attn, heads concatenated: [1, B, NH, HD] x2 -> [1, 1, B, NH*HD] bf16 TILE (L1 interleaved).
    Does not deallocate its inputs. Refuses a width the self-test did not pass (mf=None, the served call)."""
    _check(attn, gate, B, NH, HD)
    if mf is None:
        k = key(B, NH, HD)
        if k not in _tested:
            raise RuntimeError(f"{ENV}: B={B} NH={NH} HD={HD} was not self-tested this process (tested: "
                               f"{sorted(_tested)}); refusing to serve it")
        mf = mustfail()
        if k not in _logged:
            _logged.add(k)
            logger.info(f"QWEN36_ATTN_POST_FOLD engaged: B={B}: sigmoid-gate, multiply and concat-heads in 1 program per attention "
                        f"layer (was 6 at B<32, 5 at B=32), NH={NH} HD={HD}{' MUSTFAIL head rotation' if mf else ''}")
    return _run(attn, gate, B, NH, HD, bool(mf), memory_config or ttnn.L1_MEMORY_CONFIG)


def _cores(dev, B):
    grid = dev.compute_with_storage_grid_size()
    if B > grid.x * grid.y:
        raise RuntimeError(f"{ENV}: B={B} > {grid.x * grid.y} cores")
    cores = [ttnn.CoreCoord(b % grid.x, b // grid.x) for b in range(B)]
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _run(attn, gate, B, NH, HD, mf, out_mc):
    dev = attn.device()
    WT = HD // TILE
    cores, crs = _cores(dev, B)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, NH * HD]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, out_mc)

    def cb(idx, n):
        return ttnn.CBDescriptor(total_size=n * BF16_TILE, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=BF16_TILE)])

    cbs = [cb(CB_ATTN, WT), cb(CB_GATE, WT), cb(CB_SIG, WT), cb(CB_OUT, WT)]
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
    r_ct = [CB_ATTN, CB_GATE, WT] + acc(attn) + acc(gate)
    w_ct = [CB_OUT, WT, NH, 1 if mf else 0] + acc(out)
    c_ct = [CB_ATTN, CB_GATE, CB_SIG, CB_OUT, WT]
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    aa, ga, oa = attn.buffer_address(), gate.buffer_address(), out.buffer_address()
    for b, c in enumerate(cores):
        r_rt[c.x][c.y] = [aa, ga, b]
        w_rt[c.x][c.y] = [oa, b]
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = ttnn.MathFidelity.HiFi4  # ttnn.sigmoid and binary_ng multiply: HiFi4, bf16 DEST
    ccfg.fp32_dest_acc_en = False
    ccfg.math_approx_mode = False

    def kern(name, ct, rt, cfg):
        return ttnn.KernelDescriptor(kernel_source=_src(name), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                     core_ranges=crs, compile_time_args=ct, runtime_args=rt, config=cfg)

    desc = ttnn.ProgramDescriptor(
        kernels=[kern("attn_post_fold_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
                 kern("attn_post_fold_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
                 kern("attn_post_fold_compute.cpp", c_ct, [], ccfg)],
        semaphores=[], cbs=cbs)
    ttnn.generic_op([attn, gate, out], desc)
    return out


# ---------------------------------------------------------------------------------------------- reference + self-test
def served_reference(attn, gate, B, NH, HD, grid):
    """Today's ops, op for op (attention/tp.py _decode_grouped_post + _concat_heads_decode): sigmoid, multiply, height
    reshard one user per core, nlp_concat_heads_decode, sharded_to_interleaved, slice to B rows. [1,1,B,NH*HD] L1."""
    L1 = ttnn.L1_MEMORY_CONFIG
    gated = ttnn.multiply(attn, ttnn.sigmoid(gate, memory_config=L1), memory_config=L1)
    gx = min(B, grid.x)
    if B >= gx and B % gx != 0:
        gx = max(x for x in range(gx, 0, -1) if B % x == 0 and B // x <= grid.y)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, B // gx - 1))})
    shard_cfg = ttnn.create_sharded_memory_config(shape=(TILE, HD), core_grid=core_grid,
                                                  strategy=ttnn.ShardStrategy.HEIGHT,
                                                  orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                  use_height_and_width_as_shard_shape=True)
    gated_sh = ttnn.to_memory_config(gated, shard_cfg)
    ttnn.deallocate(gated)
    out_sh = ttnn.experimental.nlp_concat_heads_decode(gated_sh, num_heads=NH)
    ttnn.deallocate(gated_sh)
    out = ttnn.sharded_to_interleaved(out_sh, L1)
    ttnn.deallocate(out_sh)
    if out.shape[-2] != B:
        out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=L1)
    return out


def _chips(dev, t):
    return ttnn.get_device_tensors(t) if hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice) else [t]


def _rows(dev, t, B):
    import torch

    return [ttnn.to_torch(c).reshape(-1, int(t.shape[-1]))[:B].contiguous().view(torch.int16) for c in _chips(dev, t)]


def _ndiff(a, b):
    return sum(int((x != y).sum()) for x, y in zip(a, b))


def selftest(dev, NH, HD, widths, seed=777):
    """Fold vs served_reference on dev (mesh: every chip), random attn (L1) and gate (DRAM), bitwise on rows < B.
    Per width: the served MUSTFAIL (head rotation) honoured as the candidate; the explicit must-fail MUST differ.
    Returns {f"B{B}": "exact" | "DIFF n", f"B{B}_mustfail": n_differing (must be > 0)}."""
    import torch

    g = torch.Generator().manual_seed(seed)
    grid = dev.compute_with_storage_grid_size()
    is_mesh = hasattr(ttnn, "MeshDevice") and isinstance(dev, ttnn.MeshDevice)
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if is_mesh else {}
    res = {}
    for B in sorted(set(int(b) for b in widths)):
        a = torch.randn(1, B, NH, HD, generator=g) * 2
        gt = torch.randn(1, B, NH, HD, generator=g) * 4
        attn = ttnn.from_torch(a.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.L1_MEMORY_CONFIG, **kw)
        gate = ttnn.from_torch(gt.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        ref = served_reference(attn, gate, B, NH, HD, grid)
        cand = post(attn, gate, B, NH, HD, mf=mustfail())
        bad = post(attn, gate, B, NH, HD, mf=True)
        r, c, w = _rows(dev, ref, B), _rows(dev, cand, B), _rows(dev, bad, B)
        n = _ndiff(r, c)
        res[f"B{B}"] = "exact" if n == 0 else f"DIFF {n}"
        res[f"B{B}_mustfail"] = _ndiff(r, w)
        for t in (attn, gate, ref, cand, bad):
            ttnn.deallocate(t)
    return res


def selftest_ok(res):
    return all(v == "exact" for k, v in res.items() if not k.endswith("_mustfail")) and \
        all(v > 0 for k, v in res.items() if k.endswith("_mustfail"))


def selftest_cells(B_max, served=()):
    """Widths the fold may serve: the plain/draft widths 1, 2, 4, 8, B_max, every served verify width T*Bw (<= 32)
    and the base verify cells T 1..4 x those widths."""
    base = sorted({1, 2, 4, 8, int(B_max)} | {T * Bw for T in range(1, 5) for Bw in (1, 2, 4, 8, int(B_max))}
                  | {int(T) * int(Bw) for T, Bw in served})
    return [b for b in base if 1 <= b <= MAX_ROWS]


def ensure_selftest(dev, NH, HD, B_max, served=()):
    """Once per process per width: fail closed, record the widths."""
    if not enabled():
        return
    todo = [b for b in selftest_cells(B_max, served) if key(b, NH, HD) not in _tested]
    if not todo:
        return
    res = selftest(dev, NH, HD, todo)
    if not selftest_ok(res):
        raise RuntimeError(f"QWEN36_ATTN_POST_FOLD self-test REFUSED (NH={NH} HD={HD})"
                           f"{' (' + MUSTFAIL_ENV + '=1: expected)' if mustfail() else ''}: {res}")
    for b in todo:
        _tested[key(b, NH, HD)] = res[f"B{b}"]
    logger.info(f"{SELFTEST_BANNER}: widths {todo} NH={NH} HD={HD}, bitwise vs the served sigmoid/multiply/"
                f"concat-heads chain on every chip; wrong-input branch differs")
