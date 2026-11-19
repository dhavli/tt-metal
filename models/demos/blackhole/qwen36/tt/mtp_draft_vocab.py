# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_DRAFT_VOCAB=<N>: a frequency-ranked reduced-vocabulary lm_head for the MTP DRAFT steps only
(FR-Spec style; docs/MTP-BATCH.md section 17).

Each MTP draft step ends in the full lm_head (248320 vocab, 124160 columns per chip) plus the device argmax.
A draft only has to be a good guess: the verify is exact greedy over the full head, so a draft taken from a
subset of the vocabulary can lower acceptance but never changes an emitted token.

  ids        the N token ids: the tokenizer's special / EOS ids first, then the most frequent ids of a general
             corpus (draft_vocab/vanilla-freq-rank.npy, provenance in the .json beside it), exactly N,
             sorted ASCENDING by token id. Sorted order makes the reduced argmax's tie rule (lowest sub-index)
             the full head's tie rule (lowest token id) restricted to the subset.
  weight     the N columns of the host lm_head [dim, V] cut at model load, in the served head's dtype
             (QWEN36_LMHEAD_DTYPE), vocab-sharded over the mesh exactly like the full head (ShardTensorToMesh
             dim=-1: chip d holds sub-indices [d*N/TP, (d+1)*N/TP)); with QWEN36_DECODE_MR_LMHEAD a
             multi-reader DRAM width-sharded decode copy (tp_common.DecodeMRLMHead at n = N/TP) replaces it.
  draft step logits = x @ W_sub (vocab-sharded), Qwen36Model._greedy_argmax_device(per=N/TP) -> sub-index,
             then the id map on device: ttnn.embedding of the sub-index into a replicated [N, 32] bf16 table
             holding the id's base-256 digits (each digit <= 255 is exact in bf16; ttnn.embedding takes bf16
             weights only), recombined in fp32 (exact below 2^24) and typecast to uint32. Nothing leaves the
             device; the table is a persistent tensor, so the draft trace replays it.
  verify / plain decode  untouched: they keep the full head.

QWEN36_MTP_DRAFT_VOCAB_MUSTFAIL=1 shifts the id table by one (sub-index s maps to ids[s+1]): the drafts become
wrong ids, so acceptance in the boot stats must collapse to ~0 while the output stays exact. The warmup
self-test must detect the shift (it then logs and lets the boot continue, so the served acceptance can show it);
without the must-fail flag any self-test miss refuses the boot.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ENV = "QWEN36_MTP_DRAFT_VOCAB"
ENV_MUSTFAIL = "QWEN36_MTP_DRAFT_VOCAB_MUSTFAIL"
ENV_CHUNKS = "QWEN36_MTP_DRAFT_VOCAB_CHUNKS"
ENV_FILE = "QWEN36_MTP_DRAFT_VOCAB_FILE"
DEFAULT_FILE = Path(__file__).resolve().parent / "draft_vocab" / "vanilla-freq-rank.npy"
TILE = 32
DIGITS = 3  # base-256 digits: 256**3 > every vocab id
DIGIT_BASE = 256
MARGIN = 0.05  # self-test: a planted row counts as clear when its subset top-1 leads top-2 by 5% of |top-1|


# ------------------------------------------------------------------ host side (no ttnn: unit-testable)
def requested(tp=2, vocab=248320):
    """N from QWEN36_MTP_DRAFT_VOCAB (0 = off). N must be a multiple of 32*TP (whole tiles per chip) and < vocab."""
    raw = _tt_public_getenv(ENV, "").strip()
    if raw in ("", "0"):
        return 0
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{ENV}={raw!r}: expected an integer token count") from None
    q = TILE * int(tp)
    if n < q or n % q:
        raise ValueError(f"{ENV}={n}: must be a positive multiple of {q} (32 x TP, whole tiles per chip)")
    if n >= vocab:
        raise ValueError(f"{ENV}={n}: must be below the vocab {vocab} (the full head already is the draft head)")
    return n


def mustfail():
    v = _tt_public_getenv(ENV_MUSTFAIL, "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{ENV_MUSTFAIL}={v!r}: 0|1")
    return v == "1"


def load_ranked(path=None):
    """(ranked ids int64, provenance dict). The .npy's sha256 must match the provenance (fail closed)."""
    p = Path(path or _tt_public_getenv(ENV_FILE) or DEFAULT_FILE)
    prov_p = p.with_suffix(".json")
    with open(prov_p) as f:
        prov = json.load(f)
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if prov.get("npy_sha256") != sha:
        raise RuntimeError(f"{ENV}: {p.name} sha256 {sha[:12]} != provenance {str(prov.get('npy_sha256'))[:12]}")
    ranked = np.load(p).astype(np.int64)
    return ranked, prov


def select_ids(ranked, specials, n, vocab):
    """Exactly n distinct ids: every special id first, then ranks in order. Sorted ascending (tie rule)."""
    specials = sorted({int(s) for s in specials})
    if any(s < 0 or s >= vocab for s in specials):
        raise ValueError(f"{ENV}: special id outside the vocab {vocab}")
    if len(specials) > n:
        raise ValueError(f"{ENV}={n}: fewer slots than the {len(specials)} special ids")
    chosen = set(specials)
    for i in ranked:
        if len(chosen) >= n:
            break
        i = int(i)
        if 0 <= i < vocab:
            chosen.add(i)
    if len(chosen) < n:
        raise ValueError(f"{ENV}={n}: the ranked list gives only {len(chosen)} ids")
    return np.array(sorted(chosen), dtype=np.int64)


def digit_table(ids, shift=0):
    """[N, 32] float table, row s = base-256 digits of ids[(s + shift) % N] in columns 0..2, zeros elsewhere.
    shift=1 is the must-fail map (off by one into the table)."""
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.int64)
    n = ids.numel()
    src = ids[(torch.arange(n) + int(shift)) % n]
    if int(src.max()) >= DIGIT_BASE**DIGITS:
        raise ValueError(f"{ENV}: id {int(src.max())} needs more than {DIGITS} base-{DIGIT_BASE} digits")
    t = torch.zeros(n, TILE, dtype=torch.float32)
    rem = src.clone()
    for d in range(DIGITS):
        t[:, d] = (rem % DIGIT_BASE).float()
        rem = rem // DIGIT_BASE
    return t


def digits_to_ids(rows):
    """Host mirror of the device recombine: rows [.., >=3] digit floats -> int64 ids."""
    r = torch.as_tensor(rows, dtype=torch.float32)
    out = r[..., 0].double()
    for d in range(1, DIGITS):
        out = out + r[..., d].double() * float(DIGIT_BASE**d)
    return out.round().long()


def host_ref_ids(sub_logits, ids):
    """Reference draft ids from reduced logits [B, N]: lowest-index argmax (vLLM greedy), mapped to token ids."""
    lg = torch.as_tensor(sub_logits, dtype=torch.float32)
    m = lg.max(dim=-1, keepdim=True).values
    first = (lg == m).float().argmax(dim=-1)  # first maximal index
    return torch.as_tensor(np.asarray(ids), dtype=torch.int64)[first]


def host_ref_from_full(full_logits, ids):
    """argmax(full_logits[:, subset]) mapped back to token ids (the definition the reduced head must meet)."""
    idt = torch.as_tensor(np.asarray(ids), dtype=torch.int64)
    return host_ref_ids(torch.as_tensor(full_logits, dtype=torch.float32)[:, idt], idt)


def sharded_argmax_ref(sub_logits, tp):
    """Host mirror of _greedy_argmax_device on a vocab-sharded [B, N] row: per-shard lowest-index argmax, the
    higher shard only when STRICTLY greater. Returns sub-indices [B]."""
    lg = torch.as_tensor(sub_logits, dtype=torch.float32)
    per = lg.shape[-1] // tp
    best_i = None
    best_v = None
    for d in range(tp):
        sh = lg[:, d * per:(d + 1) * per]
        v = sh.max(dim=-1).values
        i = (sh == v[:, None]).float().argmax(dim=-1) + d * per
        if best_i is None:
            best_i, best_v = i, v
        else:
            win = v > best_v
            best_i = torch.where(win, i, best_i)
            best_v = torch.where(win, v, best_v)
    return best_i


def cut_rows(lm_head_host, ids):
    """[dim, V] host head -> [dim, N] contiguous columns of `ids` (the reduced draft weight, pre-quantisation)."""
    idt = torch.as_tensor(np.asarray(ids), dtype=torch.int64)
    return lm_head_host.index_select(-1, idt).contiguous()


def plant_targets(n, tp):
    """The sub-indices the planted self-test rows must cover: both table edges, both sides of every shard seam,
    and an interior point. Sorted, distinct."""
    per = n // tp
    t = {0, n - 1, n // 3}
    for d in range(1, tp):
        t.update({d * per - 1, d * per})
    return sorted(t)


def plant_tiles(n, tp):
    """Local (per-chip) tile starts to read back so every plant target has its 32 neighbouring columns:
    {local_tile_start: [targets in that tile on any chip]}. One ttnn.slice [:, c0:c0+32] of the vocab-sharded
    weight returns that tile from EVERY chip (chip d's columns are sub-indices d*per + c0 + 0..31)."""
    per = n // tp
    out = {}
    for t in plant_targets(n, tp):
        c0 = ((t % per) // TILE) * TILE
        out.setdefault(c0, []).append(t)
    return out


def tile_sub_indices(c0, tp, per):
    """Sub-indices of the columns of a [dim, TILE*tp] read-back (ConcatMeshToTensor dim=-1) of local tile c0."""
    return [d * per + c0 + c for d in range(tp) for c in range(TILE)]


def check_plant_columns(cols, cand_j):
    """Fail closed on a reduced weight that cannot be the real head: a non-finite or all-zero column read back
    from the DEVICE copy means its tensor cache was written from a warm-cache placeholder (the host head is a
    dataless torch.empty on a warm build) or is corrupt."""
    cols = torch.as_tensor(cols, dtype=torch.float32)
    bad_nf = [int(j) for j, ok in zip(cand_j, torch.isfinite(cols).all(dim=0)) if not bool(ok)]
    zero = [int(j) for j, m in zip(cand_j, cols.abs().amax(dim=0)) if float(m) == 0.0]
    if bad_nf or len(zero) == len(cand_j):
        raise RuntimeError(
            f"{ENV}: the device reduced lm_head is not a real head (non-finite columns {bad_nf[:4]}, "
            f"{len(zero)}/{len(cand_j)} all-zero columns among the plant candidates); its tensor cache was built "
            f"from a placeholder. Delete output.weight.vshard*.dv* from the tensor cache and boot cold "
            f"(TT_TRANSFORMERS_FORCE_MODEL_LOAD=1)."
        )
    return zero


def plant_rows(cols, scale=40.0):
    """[dim, C] device-read-back columns -> [C, dim] planted hidden states x = W_j / |W_j| * scale."""
    c = torch.as_tensor(cols, dtype=torch.float32).T
    return c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6) * scale


def subset_margin(sub_logits):
    """Per row: (top1 - top2) / max(|top1|, 1) of [R, N] subset logits, and the lowest-index argmax."""
    lg = torch.as_tensor(sub_logits, dtype=torch.float32)
    top2 = lg.topk(2, dim=-1).values
    m = (top2[:, 0] - top2[:, 1]) / top2[:, 0].abs().clamp_min(1.0)
    first = (lg == top2[:, :1]).float().argmax(dim=-1)
    return m, first


def choose_plants(cand_j, full_sub_logits, targets, n, tp, margin=None):
    """Pick the planted columns from MEASURED full-head logits (the served full head on the device, at the served
    dtype and fidelity), not from a host-precision model of them. cand_j [C] sub-indices of the read-back
    candidate columns, full_sub_logits [C, N] = the full head's logits on plant_rows(cand cols), restricted to
    the subset. For each target, the candidates in the target's own tile on the target's chip are ranked:
    clear (margin >= MARGIN) and won (the planted j itself is the subset argmax) first, then by margin.
    Returns (plant_j sorted distinct, info {j: (margin, won)}). Fail closed unless every shard holds a clear
    plant: a self-test whose planted rows never clear decides nothing."""
    margin = MARGIN if margin is None else margin
    per = n // tp
    m, first = subset_margin(full_sub_logits)
    cand_j = [int(j) for j in cand_j]
    pos = {j: i for i, j in enumerate(cand_j)}
    chosen = {}
    for t in targets:
        d, c0 = t // per, ((t % per) // TILE) * TILE
        window = [j for j in range(d * per + c0, d * per + c0 + TILE) if j in pos]
        if not window:
            continue

        def key(j):
            i = pos[j]
            clear = float(m[i]) >= margin
            return (clear and int(first[i]) == j, clear, float(m[i]))

        best = max(window, key=key)
        i = pos[best]
        if float(m[i]) >= margin:
            chosen[best] = (round(float(m[i]), 4), int(first[i]) == best)
    shards = {j // per for j in chosen}
    if len(shards) < tp:
        best_m = sorted((round(float(x), 4) for x in m), reverse=True)[:6]
        raise RuntimeError(
            f"{ENV} self-test: no planted column with a clear device margin (>= {margin}) on every shard "
            f"(chosen {chosen}; best candidate margins {best_m}); the planted check cannot decide anything"
        )
    return sorted(chosen), chosen


def memory_bytes_per_chip(n, tp, dim, dtype_name, mr_copy_bytes=None):
    """Device DRAM per chip: the weight (interleaved, or the MR copy that replaces it) + the replicated table."""
    tile_b = {"bf4": 576, "bf8": 1088}[dtype_name]
    w = (dim // TILE) * (n // tp // TILE) * tile_b
    table = n * TILE * 2
    return (mr_copy_bytes if mr_copy_bytes is not None else w) + table


# ------------------------------------------------------------------ device side
class DraftVocabHead:
    """The reduced draft lm_head on the vocab-sharded TP mesh: weight, id table, the draft-id chain, self-test."""

    def __init__(self, model, lm_head_host, n, tensor_cache_path, lm_dtype, lm_dtype_name):
        import ttnn

        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        args, mesh = model.args, model.mesh_device
        self.model = model
        self.mesh = mesh
        self.tp = int(model.num_devices)
        self.vocab = int(args.vocab_size)
        self.n = int(n)
        self.per = self.n // self.tp
        self.mustfail = mustfail()
        if self.tp < 2 or not getattr(model, "_lmhead_vocab_sharded", False):
            raise RuntimeError(f"{ENV} needs the vocab-sharded TP lm_head (TP={self.tp})")
        ranked, prov = load_ranked()
        specials = list(prov.get("tokenizer", {}).get("special_ids", []))
        for e in getattr(args, "eos_token_ids", None) or []:
            specials.append(int(e))
        specials += list(prov.get("tokenizer", {}).get("eos_ids_generation_config", []))
        self.ids = select_ids(ranked, specials, self.n, self.vocab)
        self.ids_sha = hashlib.sha256(self.ids.astype(np.int64).tobytes()).hexdigest()[:10]
        self.prov = prov
        cov = prov.get("coverage_heldout", {}).get(str(self.n))
        # Weight: the host head's N columns in the served head dtype, vocab-sharded like the full head. The cache
        # name carries the dtype, N and the id-set hash (the tensor cache is keyed by file name only).
        sub = cut_rows(lm_head_host, self.ids)
        sfx = ".bf4" if lm_dtype_name == "bf4" else ""
        cache = (Path(tensor_cache_path) / f"output.weight.vshard{sfx}.dv{self.n}-{self.ids_sha}") if tensor_cache_path else None
        w = ttnn.as_tensor(
            sub,
            dtype=lm_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        # Planted-row candidates for the self-test, read back from the DEVICE weight (cache hit or fresh cut), never
        # from lm_head_host: on a warm-cache build the host head is a dataless placeholder (weight_cache.py
        # Qwen36CachedStateDict: tensors above 64M elements stay torch.empty, i.e. zeros), so a host-chosen plant
        # was x = 0 on every hardware boot (c160 steps 119/137/140: plant margins all 0.0, every planted row
        # drafted sub-index 0, planted_clear 0). The read-back columns are the dequantised values the device
        # multiplies; which of them get planted is decided in selftest() from the full head's DEVICE logits.
        if float(sub[:, :TILE].abs().amax()) == 0.0:
            logger.info(f"{ENV}: host lm_head is a warm-cache placeholder; the reduced weight comes from the tensor cache")
        self._plant_targets = plant_targets(self.n, self.tp)
        self._cand_j, self._cand_cols = self._read_plant_candidates(w)
        self._plant_j = None
        self._plant_info = None
        del sub
        self.mr = None
        self.mr_chunks = None
        if getattr(model, "_lm_head_mr", None) is not None:
            full = model._lm_head_mr
            env_c = _tt_public_getenv(ENV_CHUNKS, "").strip()
            tries = [int(env_c)] if env_c else [1, 2, 4, 8, 16]
            err = None
            for c in tries:
                try:
                    self.mr = tpc.DecodeMRLMHead(
                        int(args.dim), self.per, c, full.readers, full.in0_cores, mesh.compute_with_storage_grid_size()
                    )
                    break
                except ValueError as e:  # chunks that do not divide the tiles / overflow the L1 estimate
                    err = e
            if self.mr is None:
                raise RuntimeError(f"{ENV}: no multi-reader config for N/TP={self.per} ({err})")
            self.mr_chunks = tpc.decode_mr_lmhead_weights(w, self.mr)
            ttnn.deallocate(w)  # decode-only: the MR copy replaces the interleaved cut
            self.w = None
            layout = f"multi-reader {self.mr.describe()}"
            mr_bytes = self.mr.copy_bytes
        else:
            self.w = w
            layout = "DRAM interleaved (ttnn-auto matmul)"
            mr_bytes = None
        tab = digit_table(self.ids, shift=1 if self.mustfail else 0)
        self.table = ttnn.as_tensor(
            tab,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.mem_bytes = memory_bytes_per_chip(self.n, self.tp, int(args.dim), lm_dtype_name, mr_bytes)
        logger.info(
            f"QWEN36_MTP_DRAFT_VOCAB engaged: N={self.n} of {self.vocab} (draft steps only; verify + plain decode keep the full head), "
            f"{lm_dtype_name} vocab-sharded {self.per}/chip, {layout}; ids {self.ids_sha} from {DEFAULT_FILE.name} "
            f"(specials {len(set(specials))} kept), held-out coverage {cov}; +{self.mem_bytes / 2**20:.1f} MiB/chip"
        )
        if self.mustfail:
            logger.warning(
                f"QWEN36_MTP_DRAFT_VOCAB_MUSTFAIL engaged: id table shifted by one (sub-index s -> ids[s+1]); acceptance MUST collapse"
            )

    def _read_plant_candidates(self, w):
        """Read the self-test's candidate plant columns back from the vocab-sharded device weight `w`: one
        tile-aligned [dim, 32] slice per needed local tile (it returns that tile from every chip). Returns
        (cand_j sub-indices, cols [dim, C] float32). Small (C = 32 * TP * tiles); fails closed on a placeholder."""
        import ttnn

        cat = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
        shape = [int(d) for d in w.shape]  # per-chip shape [.., dim, N/TP]
        dim = shape[-2]
        if shape[-1] != self.per:
            raise RuntimeError(f"{ENV}: reduced weight is {shape} per chip, expected N/TP={self.per} columns")
        lead = [0] * (len(shape) - 2)
        cand_j, cols = [], []
        for c0 in sorted(plant_tiles(self.n, self.tp)):
            full_range = c0 == 0 and self.per == TILE
            s = w if full_range else ttnn.slice(w, lead + [0, c0], shape[:-2] + [dim, c0 + TILE])
            t = ttnn.to_torch(s, mesh_composer=cat).float().reshape(dim, TILE * self.tp)
            if not full_range:
                ttnn.deallocate(s)
            cand_j += tile_sub_indices(c0, self.tp, self.per)
            cols.append(t)
        cols = torch.cat(cols, dim=1)
        zero = check_plant_columns(cols, cand_j)
        keep = [i for i, j in enumerate(cand_j) if j not in set(zero)]
        return [cand_j[i] for i in keep], cols[:, keep].contiguous()

    def _choose_plants(self):
        """Run the FULL served head (device, served dtype + fidelity) on every candidate plant row and keep, per
        target, the candidate whose planted row has the clearest measured subset margin (choose_plants)."""
        import ttnn

        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        cat = ttnn.ConcatMeshToTensor(self.mesh, dim=3)
        dim = int(self.model.args.dim)
        idt = torch.as_tensor(self.ids, dtype=torch.int64)
        x_all = plant_rows(self._cand_cols)
        sub = []
        for r0 in range(0, x_all.shape[0], TILE):
            x = x_all[r0:r0 + TILE]
            R = x.shape[0]
            xt = ttnn.from_torch(
                x.reshape(1, 1, R, dim).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep,
            )
            full = self.model._lm_head_matmul(xt, decode=True)
            sub.append(ttnn.to_torch(full, mesh_composer=cat).float().reshape(-1, self.vocab)[:R][:, idt])
            ttnn.deallocate(full)
            ttnn.deallocate(xt)
        self._plant_j, self._plant_info = choose_plants(self._cand_j, torch.cat(sub), self._plant_targets, self.n, self.tp)
        pos = {j: i for i, j in enumerate(self._cand_j)}
        self._plant_cols = x_all[[pos[j] for j in self._plant_j]]
        logger.info(f"{ENV} plant columns (device-measured full-head margin, planted j won): {self._plant_info}")

    # -------------------------------------------------------------- forward
    def logits(self, x):
        """[1,1,B,dim] replicated -> vocab-sharded [1,1,B,N/TP] bf16 TILE DRAM (same contract as the full head)."""
        import ttnn

        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        if self.mr is not None:
            if x.shape[-2] > TILE:
                raise RuntimeError(f"{ENV}: draft lm_head input has {x.shape[-2]} rows > 32")
            return tpc.decode_mr_lmhead_matmul(x, self.mr_chunks, self.mr)
        return ttnn.linear(x, self.w)

    def map_ids(self, sub):
        """Replicated [1,1,B,1] uint32 sub-indices -> [1,1,B,1] uint32 token ids, on device."""
        import ttnn

        B = sub.shape[-2]
        s_rm = ttnn.reshape(ttnn.to_layout(sub, ttnn.ROW_MAJOR_LAYOUT), (1, B))
        e = ttnn.embedding(s_rm, self.table, layout=ttnn.TILE_LAYOUT)  # [1,B,32] bf16 digits
        e4 = ttnn.reshape(e, (1, 1, B, TILE))
        f = ttnn.typecast(e4, ttnn.float32)
        c = [ttnn.slice(f, [0, 0, 0, d], [1, 1, B, d + 1]) for d in range(DIGITS)]
        hi = ttnn.add(ttnn.multiply(c[1], float(DIGIT_BASE)), ttnn.multiply(c[2], float(DIGIT_BASE**2)))
        tok = ttnn.add(c[0], hi)
        out = ttnn.typecast(tok, ttnn.uint32)
        for t in [s_rm, e, f, hi, tok] + c:
            ttnn.deallocate(t)
        return out

    def draft_ids(self, x):
        """The draft step's greedy ids from the reduced head: [1,1,B,1] uint32 replicated token ids."""
        import ttnn

        sub = self.model._greedy_argmax_device(self.logits(x), per=self.per)
        ids = self.map_ids(sub)
        ttnn.deallocate(sub)
        return ids

    # -------------------------------------------------------------- warmup self-test
    def selftest(self, widths, seed=0):
        """Fail closed (unless MUSTFAIL, which it must catch) before any trace capture. Per width B, rows:
        planted rows first, then random normal hidden states. A planted row is x = W_sub[:, j] / |W_sub[:, j]| * 40
        with W_sub[:, j] read back from the DEVICE reduced weight (never the host head, a placeholder on warm
        builds) and j chosen once, per target (table edges, both sides of the shard seam, an interior point), by the
        full served head's own DEVICE logits on that row (served dtype + fidelity: the margin is measured where it
        is used, so bf4 / LoFi cannot erase a host-predicted one). Checks:
          chain    device draft ids == ids[sharded lowest-index argmax of the SAME device reduced logits]: the
                   argmax, the per-shard offset and the id map, bit-exact (the one check the must-fail breaks);
          subset   every draft id is in the id set;
          full     on planted rows whose subset margin in the full head is clear (MARGIN), the draft id ==
                   argmax(full_logits[subset]) mapped (the full served head on the same x): the cut columns are
                   the right ones. Every width must have at least one clear planted row, and the chosen plants
                   must cover every shard (choose_plants). Random rows only report the agreement (the bf4 blocks
                   of the cut regroup columns, so near-ties may legitimately differ)."""
        import ttnn

        if self._plant_j is None:
            self._choose_plants()
        g = torch.Generator().manual_seed(seed)
        dim = int(self.model.args.dim)
        idset = set(int(i) for i in self.ids)
        idt = torch.as_tensor(self.ids, dtype=torch.int64)
        res = {"widths": [], "mustfail": self.mustfail}
        chain_ok = subset_ok = plant_ok = clear_ok = True
        agree = total = n_clear = 0
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        cat = ttnn.ConcatMeshToTensor(self.mesh, dim=3)
        for B in sorted(set(int(w) for w in widths)):
            P = min(B, len(self._plant_j))
            x = torch.randn(B, dim, generator=g)
            plant = list(range(P))
            for r in plant:
                x[r] = self._plant_cols[r]
            xt = ttnn.from_torch(
                x.reshape(1, 1, B, dim).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep,
            )
            lg = self.logits(xt)
            sub_lg = ttnn.to_torch(lg, mesh_composer=cat).float().reshape(-1, self.n)[:B]
            ids_dev = self.draft_ids(xt)  # recomputes the logits (deterministic) and consumes them
            ttnn.deallocate(lg)
            dev = ttnn.to_torch(ttnn.get_device_tensors(ids_dev)[0]).reshape(-1)[:B].long()
            ttnn.deallocate(ids_dev)
            full = self.model._lm_head_matmul(xt, decode=True)
            full_lg = ttnn.to_torch(full, mesh_composer=cat).float().reshape(-1, self.vocab)[:B]
            ttnn.deallocate(full)
            ttnn.deallocate(xt)
            ref_chain = idt[sharded_argmax_ref(sub_lg, self.tp)]
            ref_full = host_ref_from_full(full_lg, self.ids)
            c_ok = bool(torch.equal(dev, ref_chain))
            s_ok = all(int(t) in idset for t in dev)
            # planted rows with a clear subset margin in the FULL head's logits must match it exactly
            fl_sub = full_lg[:P][:, idt]
            top2 = fl_sub.topk(2, dim=-1).values
            clear = (top2[:, 0] - top2[:, 1]) >= MARGIN * top2[:, 0].abs().clamp_min(1.0)
            p_ok = bool(torch.equal(dev[:P][clear], ref_full[:P][clear]))
            n_clear += int(clear.sum())
            planted_won = int((ref_full[:P] == idt[torch.as_tensor(self._plant_j[:P])]).sum())
            rnd = slice(P, B)
            agree += int((dev[rnd] == ref_full[rnd]).sum())
            total += max(0, B - P)
            chain_ok &= c_ok
            subset_ok &= s_ok
            plant_ok &= p_ok
            clear_ok &= int(clear.sum()) > 0
            res["widths"].append({"B": B, "chain_exact": c_ok, "in_subset": s_ok, "planted_exact": p_ok,
                                  "planted_clear": int(clear.sum()), "planted_j_won": planted_won,
                                  "dev": dev.tolist()[:8], "ref_chain": ref_chain.tolist()[:8]})
        res["random_rows_agree_full"] = f"{agree}/{total}"
        res["planted_clear_total"] = n_clear
        res["plants"] = {int(j): v for j, v in self._plant_info.items()}
        res["ok"] = chain_ok and subset_ok and plant_ok and clear_ok
        if self.mustfail:
            if chain_ok:
                raise RuntimeError(f"{ENV_MUSTFAIL}=1 but the self-test did NOT see the shifted id table (blind): {res}")
            logger.warning(f"{ENV} self-test caught the {ENV_MUSTFAIL} shifted table as expected; boot continues: {res}")
            return res
        if not res["ok"]:
            raise RuntimeError(f"{ENV} self-test FAILED (refusing the boot): {res}")
        logger.info(f"QWEN36_MTP_DRAFT_VOCAB self-test exact: {res}")
        return res


def build(model, lm_head_host, tensor_cache_path, lm_dtype, lm_dtype_name):
    """Model-init hook: DraftVocabHead when QWEN36_MTP_DRAFT_VOCAB is set, else None (inert)."""
    n = requested(getattr(model, "num_devices", 1), int(model.args.vocab_size))
    if not n:
        if mustfail():
            raise RuntimeError(f"{ENV_MUSTFAIL}=1 needs {ENV} set")
        return None
    if _tt_public_getenv("QWEN36_MTP_BATCH") != "1":
        raise RuntimeError(f"{ENV}={n} refused: it serves only the QWEN36_MTP_BATCH draft steps")
    return DraftVocabHead(model, lm_head_host, n, tensor_cache_path, lm_dtype, lm_dtype_name)
