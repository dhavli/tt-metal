# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_BATCH=1: batched MTP speculative decode at B=4/8 on the 1x2 mesh (docs/MTP-BATCH.md).

One speculative step for a decode bucket of width Bw with draft depth K (T = K+1, M = Bw*T <= 32):

  V trace   verify: the base model over M rows (row u = j*Bw + b: user b's token at depth j),
            GDN as T sequential sub-steps of the production decode kernels (per-sub-step state
            kept), full attention with a grouped KV update + grouped SDPA, device argmax.
  host      read the M verify ids, accept per row (greedy exact, mtp_batch_host.py).
  C trace   commit: per-row GDN recurrent + conv state after m_b accepted drafts (where-chain, or
            QWEN36_MTP_STATE_SLOT=1: one slot-index page read by two gather programs per layer);
            reseed: the MTP layer's KV at the T verify positions (KV-only pass over the verify
            hiddens); draft: K chained MTP steps at width Bw from the anchor row m_b*Bw + b.
  host      read K*Bw draft ids for the next step (QWEN36_MTP_OVERLAP=1: enqueued without blocking and
            consumed at the start of the next step, so the vLLM host work between steps runs while the
            device commits and drafts; docs/MTP-BATCH.md section 12).

QWEN36_MTP_DEVACCEPT=1 (section 25, tt/mtp_accept.py): the accept moves to the first program of the draft trace (cd
under SO1_CSPLIT; it writes gslot, anchor, t0n, the draft positions and, under the served exact rule, the chain noise
from the verify outputs on device), so V, R and the commit/draft traces are enqueued back to back with no host round
trip in between. Every host write of the step happens before V. The host accept still runs on the read-back verify
outputs for the emitted tokens and the slot state, and is checked against the device's m / t0n page at the next drain.

The emitted tokens of row b are the verify argmax ids[b, 0..m_b]; the first is the token plain
decode would produce, so correctness never rests on the drafter. Greedy only; any step the traces
cannot take (host sampling, an uncaptured bucket, a page-table width change) runs plain decode.

Requirements, checked at construction (fail closed): 1x2 mesh, QWEN36_ONESHOT_AR=1,
QWEN36_DEVICE_ARGMAX=1, QWEN36_DEVICE_RESIDENT off (the step is host-mediated, so the platform
must run the sync scheduler), the production GDN decode kernels, the decode-levers layer.py, no
prefix caching.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import contextlib
import json
import math
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import argmax_fold
from models.demos.blackhole.qwen36.tt import dev_sample
from models.demos.blackhole.qwen36.tt import fused_sampler
from models.demos.blackhole.qwen36.tt import spec_sample
from models.demos.blackhole.qwen36.tt import mtp_batch_host as mbh
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.tp import FoldGroupIO, spec_fold_enabled

ENV_WEIGHTS = "QWEN36_MTP_BATCH_WEIGHTS"
ENV_CACHE = "QWEN36_MTP_BATCH_CACHE"
ENV_COMMIT = "QWEN36_MTP_BATCH_COMMIT"
ENV_STATS_EVERY = "QWEN36_MTP_BATCH_STATS_EVERY"
_BF4_ENVS = ("QWEN36_ATTN_GDN_BF4", "QWEN36_MLP_DOWN_BF4")

# QWEN36_MTP_SPEC_SAMPLE_DUAL (docs/MTP-BATCH.md section 26): the trace family being compiled, captured or served.
# "s" = the sampled family (the served exact rule and the device sampler, -s6-ss's ops); "g" = the greedy family (the
# ops the flags-off tree runs: no sampler program, no rule output, no drafter block). Every device-op builder and host
# input builder of the step asks _serve() / _ds() instead of the module flags, so the greedy family is the flags-off
# code path, not a copy of it. Outside a _family("g") block the family is "s": with DUAL unset nothing changes.
_FAM = {"cur": "s"}


def _serve():
    return spec_sample.serve() and _FAM["cur"] != "g"


def _ds():
    return dev_sample.enabled() and _FAM["cur"] != "g"


def _ssp():
    """QWEN36_MTP_SPEC_SAMPLE_SSP on the sampled family (fused_sampler.py): fs_cut runs the exact rule, makes the
    noise and maps the draft ids; the host neither builds nor uploads the noise blocks."""
    return _serve() and fused_sampler.ssp()


def _sc(tag):
    """QWEN36_MTP_OPTRACE: an opmark scope tag on the ops this block launches (only while a window is open, i.e. during
    the MTP captures and the served windows of an OPTRACE boot). Without the flag: a null context."""
    if _tt_public_getenv("QWEN36_MTP_OPTRACE", "") != "1":
        return contextlib.nullcontext()
    from models.demos.blackhole.qwen36.tt import opmark

    return opmark.scope(tag)


class _family:
    """with _family("g"): the greedy family's ops and host inputs (QWEN36_MTP_SPEC_SAMPLE_DUAL)."""

    def __init__(self, fam):
        if fam not in ("s", "g"):
            raise ValueError(f"trace family {fam!r}")
        self.fam = fam

    def __enter__(self):
        self.prev, _FAM["cur"] = _FAM["cur"], self.fam
        return self

    def __exit__(self, *exc):
        _FAM["cur"] = self.prev


# the per-family state of a bucket: the trace handles and the traces' output tensors (QWEN36_MTP_SPEC_SAMPLE_DUAL)
_FAM_ATTRS = ("v_trace", "c_trace", "cc_trace", "cd_trace", "r_trace", "c2_trace", "v_out", "c_out")



def VERIFY_DS_BLOCK(K):
    """QWEN36_DEVICE_SAMPLE: (role name dev_sample.sample reads, v_ trace input, width) for the verify."""
    return (("noise", "v_gum", 2 * K), ("flag", "v_guf", 1), ("rank", "v_gkr", 2 * K),
            ("topp", "v_gp", 1), ("scale", "v_gs", 2 * K))

class _LazyMTPState:
    """The checkpoint's mtp.* tensors as a read-on-access mapping, keyed the way the model tree
    wants them: mtp.layers.0.<x> -> layers.<L>.<x> (so Qwen36DecoderLayer(L) loads it as a
    full-attention layer), the fc / pre-fc norms / head norm under their own mtp.* names. Block-FP8
    tensors (a weight_scale_inv sibling) are dequantised per tensor. Nothing is retained."""

    def __init__(self, ckpt_dir, host_layer):
        from safetensors import safe_open  # noqa: F401  (fail here, not mid-load)

        self._dir = Path(ckpt_dir)
        with open(self._dir / "model.safetensors.index.json") as f:
            wmap = json.load(f)["weight_map"]
        self._raw = {k: v for k, v in wmap.items() if k.startswith("mtp.")}
        if not self._raw:
            raise RuntimeError(
                f"QWEN36_MTP_BATCH: no mtp.* tensors in {self._dir} (a packed/quantised checkpoint can drop "
                f"the MTP head; point {ENV_WEIGHTS} at the source checkpoint that has it)"
            )
        self._map = {}
        pre = "mtp.layers.0."
        for k in self._raw:
            if k.endswith(".weight_scale_inv"):
                continue
            if k.startswith(pre):
                self._map[f"layers.{host_layer}." + k[len(pre):]] = k
            else:
                self._map[k] = k
        self._handles = {}

    def _read(self, raw):
        from safetensors import safe_open

        fname = self._raw[raw]
        h = self._handles.get(fname)
        if h is None:
            h = self._handles[fname] = safe_open(str(self._dir / fname), framework="pt")
        t = h.get_tensor(raw)
        scale_key = raw.replace(".weight", ".weight_scale_inv")
        if t.dtype == torch.float8_e4m3fn and scale_key in self._raw:
            t = tpc.dequant_fp8_block(t, self._read(scale_key))
        return t.to(torch.bfloat16)

    def __getitem__(self, k):
        return self._read(self._map[k])

    def __contains__(self, k):
        return k in self._map

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def keys(self):
        return self._map.keys()

    def items(self):
        for k in self._map:
            yield k, self[k]

    def substate(self, prefix):
        p = prefix + "."
        return {k[len(p):]: self[k] for k in self._map if k.startswith(p)}

    def close(self):
        self._handles.clear()


class _MTPHead:
    """The single-layer MTP drafter on the replicated decode seam (QWEN36_ONESHOT_AR):
    h' = fc([enorm(embed(tok)), hnorm(hidden)]) -> decoder layer -> mtp.norm -> shared lm_head.
    `hidden` is the base model's final-norm output (replicated, full width) for the first step and
    mtp.norm's output for the chained steps -- the post-norm feed contract of the upstream port."""

    def __init__(self, model, sd, host_layer, cache_root):
        from models.common.rmsnorm import RMSNorm
        from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
        from models.tt_transformers.tt.distributed_norm import DistributedNorm

        args, mesh = model.args, model.mesh_device
        self.model = model
        self.args = args
        self.mesh = mesh
        self.tt_ccl = model.tt_ccl

        def norm(key, sub):
            n = RMSNorm(
                device=mesh,
                dim=args.dim,
                state_dict=sd,
                weight_key=key,
                state_dict_prefix="mtp.",
                weight_cache_path=(cache_root / sub) if cache_root else None,
                weight_dtype=ttnn.bfloat16,
                add_unit_offset=True,
                eps=args.norm_eps,
                is_distributed=args.is_distributed_norm,
                ccl_topology=args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            )
            return DistributedNorm(n, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)

        self.enorm = norm("pre_fc_norm_embedding", "enorm")
        self.hnorm = norm("pre_fc_norm_hidden", "hnorm")
        self.head_norm = norm("norm", "norm")
        fc_w = sd["mtp.fc.weight"]
        assert tuple(fc_w.shape) == (args.dim, 2 * args.dim), f"unexpected mtp.fc shape {tuple(fc_w.shape)}"
        # Column-parallel [2*dim, dim/tp] per device; bf8 like the upstream port.
        self.fc = tpc.shard_w(
            fc_w,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=(cache_root / "fc") if cache_root else None,
            dtype=ttnn.bfloat8_b,
        )
        del fc_w
        # QWEN36_MTP_FC_BLOCK (tp_common, docs/MTP-BATCH.md section 30): the decode fc's in0_block_w + boot self-test.
        self.fc_pc = tpc.mtp_fc_block_build(mesh, self.fc, 2 * args.dim, args.dim // int(model.num_devices))
        self.decoder = Qwen36DecoderLayer(
            mesh, args, sd, layer_num=host_layer, tensor_cache_path=cache_root, tt_ccl=self.tt_ccl
        )
        assert self.decoder.is_full_attention, "MTP host layer must be a full-attention layer"
        self.attention = self.decoder.attention
        # QWEN36_MTP_SPEC_SAMPLE: the chain's per-depth block (the noise, the rank mask, the 1/T,
        # the top_p and the per-row rule flag) is set before each chain step.
        self._spec_blk = None
        self._nc = dict(args.get_norm_config("lm_head", self._mode_decode()))
        self._nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG

    @staticmethod
    def _mode_decode():
        from models.tt_transformers.tt.common import Mode

        return Mode.DECODE

    def _replicate(self, x):
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(
            x, self.mesh, self.tt_ccl, cluster_axis=None, dim=3, topology=self.args.ccl_topology()
        )

    def forward(self, hidden, tok_rm, cos, sin, position_tensor=None, page_table=None, group_io=None,
                kv_only=False):
        """hidden [1,1,N,dim] replicated; tok_rm [N,1] uint32 ROW_MAJOR (the token after hidden's
        position). kv_only -> (None, None) after the MTP layer's KV write. Otherwise returns
        (greedy ids [1,1,N,1] uint32 replicated, next hidden [1,1,N,dim] replicated)."""
        model = self.model
        N = hidden.shape[-2]
        e = model.embd(tok_rm)
        e = ttnn.reshape(e, (1, 1, e.shape[0] * e.shape[1], e.shape[-1]))
        e = self._replicate(e)
        e_n = tpc.norm_replicated_decode(self.enorm, e, self._nc)
        ttnn.deallocate(e)
        h_n = tpc.norm_replicated_decode(self.hnorm, hidden, self._nc)
        cat = ttnn.concat([e_n, h_n], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [embedding, hidden]
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        if self.fc_pc is not None and N <= tpc.TILE_SIZE:
            fused = tpc.mtp_fc_block_matmul(cat, self.fc, self.fc_pc)  # [1,1,N,dim/tp] DRAM, tier 2 (boot self-test)
        else:
            fused = ttnn.linear(
                cat, self.fc, compute_kernel_config=tpc.COMPUTE_HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # [1,1,N,dim/tp]
        ttnn.deallocate(cat)
        x = self._replicate(fused)
        out = self.decoder.forward(
            x,
            cos=cos,
            sin=sin,
            mode="decode",
            position_tensor=position_tensor,
            page_table=page_table,
            oneshot_ar=True,
            group_io=group_io,
            kv_only=kv_only,
        )
        ttnn.deallocate(x)
        if kv_only:
            assert out is None
            return None, None, None
        normed = tpc.norm_replicated_decode(self.head_norm, out, self._nc)
        ttnn.deallocate(out)
        ids, extra = self.draft_ids(normed)
        assert ids.shape[-2] == N
        return ids, normed, extra

    def draft_ids(self, normed):
        """The chain's token from the head's normed hidden [1,1,N,dim]: (ids [1,1,N,1] uint32 full-vocab,
        extra). extra is None unless QWEN36_MTP_SPEC_SAMPLE serves, then {"probs", "ids"}: the drafter's
        post-cut candidate block (draft-vocab SUB-ids under QWEN36_MTP_DRAFT_VOCAB; the host maps them)."""
        model = self.model
        dv = getattr(model, "_draft_vocab", None)
        if _serve():
            return self._spec_candidates(normed, dv)
        if dv is not None:
            return dv.draft_ids(normed), None  # QWEN36_MTP_DRAFT_VOCAB: reduced head, ids mapped back on device
        logits = model._lm_head_matmul(normed, decode=True)
        return model._greedy_argmax_device(logits), None

    def _spec_candidates(self, normed, dv):
        """QWEN36_MTP_SPEC_SAMPLE: the drafter's candidate block (the request's own T/top_k/top_p, from
        self._spec_blk) and its token: a sample x ~ q on an exact-rule row (flag 1), else the argmax.

        The argmax is the SAME op on the SAME logits as the flag-off draft (_greedy_argmax_device, then
        the draft-vocab map), so a T = 0 or compare row drafts byte-identically to the flag off."""
        model = self.model
        blk = self._spec_blk
        per = int(dv.per) if dv is not None else int(model.args.vocab_size) // int(model.num_devices)
        logits = dv.logits(normed) if dv is not None else model._lm_head_matmul(normed, decode=True)
        if fused_sampler.enabled() and "_ssp" in blk:
            # QWEN36_MTP_SPEC_SAMPLE_SSP: the same 3 programs with the noise made in fs_cut and the draft-vocab id map
            # folded in (no map_ids chain); the candidate ids stay draft-vocab sub-ids (the host maps them)
            c = fused_sampler.serve_candidates(logits, per, self.mesh, model.tt_ccl, model.args.ccl_topology(), blk,
                                               spec_sample.k(), noise=blk["_ssp"],
                                               table=getattr(self, "_ssp_table", None) if dv is not None else None)
            ttnn.deallocate(logits)
            return c["sel"], {"probs": c["probs"], "ids": c["ids"]}
        if fused_sampler.enabled():
            # QWEN36_MTP_SPEC_SAMPLE_FUSED: block + cut + pick + greedy + select in 3 programs (fused_sampler.py);
            # sel = flag ? pick : greedy, the same ids as the chain below (boot self-test)
            c = fused_sampler.serve_candidates(logits, per, self.mesh, model.tt_ccl, model.args.ccl_topology(), blk,
                                               spec_sample.k())
            ttnn.deallocate(logits)
            sel = c["sel"]
            if dv is not None:
                sub = sel
                sel = dv.map_ids(sub)
                ttnn.deallocate(sub)
            return sel, {"probs": c["probs"], "ids": c["ids"]}
        c = spec_sample.candidates(logits, per, self.mesh, model.tt_ccl, model.args.ccl_topology(), blk,
                                   K=spec_sample.k())
        if spec_sample.draft_pick():
            # QWEN36_MTP_SPEC_SAMPLE_DRAFTPICK: the host sends ZERO noise on the non-exact rows, so the pick IS the
            # block's argmax there; the second full-vocab argmax and the select are not run.
            ttnn.deallocate(logits)
            sel = c["pick"]
        else:
            greedy = model._greedy_argmax_device(logits, per=per if dv is not None else None)  # frees the logits
            sel = dev_sample.select_ids(greedy, c["pick"], blk["flag"])
            ttnn.deallocate(greedy)
            ttnn.deallocate(c["pick"])
        if dv is not None:
            sub = sel
            sel = dv.map_ids(sub)
            ttnn.deallocate(sub)
        return sel, {"probs": c["probs"], "ids": c["ids"]}


class _Bucket:
    """Persistent host->device inputs (allocated before any trace capture) and the trace handles
    and outputs of one speculating bucket width."""

    def __init__(self, bw, K, Bmax, nblk, rec_mask_dtype=torch.bfloat16, slot=False, da=False):
        self.bw, self.K, self.T, self.M = bw, K, K + 1, bw * (K + 1)
        # QWEN36_MTP_DEVACCEPT: more persistent inputs (da_plan, the plan-only page written before V; under the served
        # rule d_sgtab{k}, the chain noise for every possible m) and an output page (da_m: the device's m_b and t0n_b,
        # read back with the drafts and checked at the drain).
        self.da = bool(da)
        self.da_m = None
        self.Bmax = Bmax
        # QWEN36_MTP_STATE_SLOT: the commit reads one uint32 slot-index page (gslot) instead of the
        # 2*T one-hot mask tensors (gmask*/cmask*), which are then neither allocated nor written.
        self.slot = bool(slot)
        self.nblk = nblk
        # dtype of the GDN recurrent-state commit masks: the self-test picks it (an fp32 state
        # selected through a bf16 mask can come back bf16, i.e. silently rounded every step).
        self.rec_mask_dtype = rec_mask_dtype
        self.v_trace = None
        self.c_trace = None
        # QWEN36_DEVICE_SAMPLE: the step's sampling params (the per-row seed and temperature), stashed
        # by maybe_step so the Gumbel block can be keyed by the request's seed.
        self.sp = None
        self.cc_trace = None  # TIMING, or OVERLAP+so1 C-split: commit (cc) + reseed/draft (cd)
        self.cd_trace = None
        self.r_trace = None  # QWEN36_MTP_EARLY_RESEED: reseed (r) enqueued after verify, then commit + draft steps (c2)
        self.c2_trace = None
        self.v_out = None  # (ids_rm [M,1], hid [1,1,M,dim] TILE, hid_rm [M,dim] RM)
        # QWEN36_MTP_SPEC_SAMPLE: the step's per-row flags (set by _spec_block): exact = the exact rule
        # serves the row (rule 1 and T > 0), greedy = T == 0.
        self.spec_exact = None
        self.spec_greedy = None
        self.wcache = {}  # QWEN36_MTP_SPEC_SAMPLE_WSKIP: name -> the host block its device buffer holds
        self.c_out = None  # [K] draft ids [Bw,1] RM
        # QWEN36_MTP_SPEC_SAMPLE_DUAL: family -> {the _FAM_ATTRS of that family's traces, "keep": the GDN layers'
        # verify leftovers (_mtp_verify[Bw]) its commit trace reads}; fam = the family bound to the attributes now.
        self.fams = {}
        self.fam = "s"
        # QWEN36_PREFIX_CACHING: the MTP head reads its own page table v_mpt (the verify table shifted per row past the
        # request's prefix-hit blocks, mbh.shifted_page_table); pt_shift is this step's per-row shift. Off: no v_mpt
        # buffer exists and the head reads v_pt, as before.
        self.mpt = _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") == "1"
        self.pt_shift = None

    def save_fam(self, fam, keep=None):
        """Record the traces just captured (the attributes) as family fam. keep holds the GDN layers' verify leftovers
        that fam's commit trace reads: the next family's verify capture re-binds each layer's _mtp_verify[Bw], and the
        tensors must outlive it (they are the commit trace's inputs for the life of the trace)."""
        rec = {a: getattr(self, a) for a in _FAM_ATTRS}
        rec["keep"] = keep
        self.fams[fam] = rec
        self.fam = fam

    def use_fam(self, fam):
        """Bind family fam's trace handles and outputs to the attributes the step functions read."""
        if self.fam == fam:
            return
        rec = self.fams[fam]
        for a in _FAM_ATTRS:
            setattr(self, a, rec[a])
        self.fam = fam

    def _ds_block(self, plan, M):
        """QWEN36_DEVICE_SAMPLE: the verify's per-step HOST block: the SCALED Gumbel noise (T*g),
        the select flag, the top_k rank mask, the per-row 1/T (the temperature, for vLLM's order)
        and the top_p threshold.

        Row r = j*bw + b (the SAME order as the verify ids): the noise is keyed by (the request's
        seed, the position of the token that row predicts), so it is deterministic per request and
        per position.  A row with temperature == 0 gets the flag 0.0 and keeps the greedy ids."""
        K = dev_sample.k()
        if plan is None:
            if _ssp():
                self._np_ds = ([0] * M, [0] * M, [0.0] * M)
            z = torch.zeros((M, 2 * K), dtype=torch.float32)
            return {"noise": z.reshape(1, 1, M, 2 * K),
                    "flag": torch.zeros((M, 1), dtype=torch.float32).reshape(1, 1, M, 1),
                    "rank": torch.ones((M, 2 * K), dtype=torch.float32).reshape(1, 1, M, 2 * K),
                    "scale": torch.ones((M, 2 * K), dtype=torch.float32).reshape(1, 1, M, 2 * K),
                    "topp": torch.ones((M, 1), dtype=torch.float32).reshape(1, 1, M, 1)}
        sp = getattr(self, "sp", None)
        bw, T = plan.bw, plan.T
        sl = list(getattr(sp, "seed", None) or []) if sp is not None else []
        tl = list(getattr(sp, "temperature", None) or []) if sp is not None else []
        kl = list(getattr(sp, "top_k", None) or []) if sp is not None else []
        pl = list(getattr(sp, "top_p", None) or []) if sp is not None else []
        seeds, temps, topks, topps = [], [], [], []
        for j in range(T):
            for b in range(bw):
                s = sl[b] if b < len(sl) else None
                t = tl[b] if b < len(tl) else None
                # QWEN36_DEVICE_SAMPLE (the s-sample3 review's blocker 1): a row with NO seed must
                # not share its noise with another row.  The old code mapped None to 0, so every
                # unseeded MTP row drew the SAME noise (two identical prompts, the same tokens).
                # The plugin writes a per-request seed; this is the fallback for a row it did not
                # seed (a fresh random 31-bit seed per row).
                if s is None:
                    seeds.append(dev_sample.random_seed31())
                else:
                    si = int(s)
                    # the same fold as the plain-decode path (model.py _ds_block_host), so the
                    # two paths give the same noise for the same (seed, position)
                    seeds.append(si if 0 <= si <= 0x7FFFFFFF else dev_sample.fold_seed31(si))
                temps.append(float(t) if t is not None else 0.0)
                topks.append(int(kl[b]) if b < len(kl) and kl[b] is not None else 0)
                topps.append(float(pl[b]) if b < len(pl) and pl[b] is not None else 1.0)
        pos = [int(plan.verify_pos[j][b]) + 1 for j in range(T) for b in range(bw)]
        if _ssp():
            # QWEN36_MTP_SPEC_SAMPLE_SSP: fs_cut makes the pick noise from these keys (the noise plan page, built in
            # _spec_block); the host builds the parameter blocks only
            self._np_ds = (seeds, pos, temps)
            return dev_sample.row_block_fast(seeds, pos, temps, topks=topks, topps=topps, K=K, M=M, noise=False)
        if spec_sample.host_split():
            # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: the same block, numpy hash + memoised parameter blocks
            # (the boot self-test compares it bitwise against row_block through the full host_inputs)
            return dev_sample.row_block_fast(seeds, pos, temps, topks=topks, topps=topps, K=K, M=M)
        return dev_sample.row_block(seeds, pos, temps, topks=topks, topps=topps, K=K, M=M)

    def _spec_params(self):
        """QWEN36_MTP_SPEC_SAMPLE: per batch row (seeds folded like _ds_block, temps, top_ks, top_ps, rules)."""
        sp = getattr(self, "sp", None)
        bw = self.bw

        def col(name, default):
            v = getattr(sp, name, None) if sp is not None else None
            v = list(v.tolist() if hasattr(v, "tolist") else (v or []))
            return [v[b] if b < len(v) and v[b] is not None else default for b in range(bw)]

        seeds = []
        for s in col("seed", None):
            if s is None:
                seeds.append(dev_sample.random_seed31())
            else:
                si = int(s)
                seeds.append(si if 0 <= si <= 0x7FFFFFFF else dev_sample.fold_seed31(si))
        temps = [float(t) for t in col("temperature", 0.0)]
        topks = [int(t) for t in col("top_k", 0)]
        topps = [float(t) for t in col("top_p", 1.0)]
        return seeds, temps, topks, topps, spec_sample.row_rules(sp, bw)

    def _spec_block(self, plan, acc, M, parts=("v", "d")):
        """QWEN36_MTP_SPEC_SAMPLE: the host blocks of one step.

        Verify (v_, written before the verify; rows j*bw+b): v_sqp/v_sqi the carried q block of draft j
        (zeros on rows without a draft), v_sx the draft id, v_sxf 1.0 where the row is EXACT and j < n_draft
        (the verify then returns the draft as its id there, so the MTP reseed sees the token that is emitted
        if it is accepted), v_su the accept uniform, v_snr the residual's Exp(1) noise.
        Chain (d_, written after the accept): d_sgum{k} the depth-k Gumbel noise pre-scaled by T, the cut
        params of the request (d_sscale / d_srank / d_stopp) and d_sflag (1.0 = sample: an exact row)."""
        K, bw = self.K, self.bw
        C2 = 2 * spec_sample.k()
        seeds, temps, topks, topps, rules = self._spec_params()
        greedy = [1 if t <= 0.0 else 0 for t in temps]
        exact = [1 if (rules[b] == spec_sample.RULE_EXACT and not greedy[b]) else 0 for b in range(bw)]
        self.spec_exact, self.spec_greedy = exact, greedy
        out = {}
        # QWEN36_MTP_SPEC_SAMPLE_DRAFTPICK: the chain's noise is zero on the non-exact rows (their draft is then the
        # block's argmax); exact rows keep their T*g.  Flag off: T*g on every T > 0 row, as served.
        ctemps = ([t if (exact[b] and t > 0.0) else 0.0 for b, t in enumerate(temps)] if spec_sample.draft_pick()
                  else [t if t > 0.0 else 0.0 for t in temps])
        if "v" not in parts:
            # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: the post-accept write sends no v_ block; skip building them
            return self._spec_d_block(plan, acc, out, seeds, temps, topks, topps, exact, ctemps, C2)
        qp, qi, xs, xf = mbh.spec_v_blocks(plan, bw, K, M, C2, exact)  # (QWEN36_MTP_DEVFEED's reference too)
        vpos = ([int(plan.verify_pos[j][b]) + 1 for j in range(self.T) for b in range(bw)] if plan is not None
                else [j for j in range(self.T) for _ in range(bw)])
        vseeds = [seeds[b] for _ in range(self.T) for b in range(bw)]
        out["v_sqp"], out["v_sqi"] = qp.reshape(1, 1, M, C2), qi.reshape(1, 1, M, C2)
        out["v_sx"], out["v_sxf"] = xs.reshape(1, 1, M, 1), xf.reshape(1, 1, M, 1)
        if _ssp():
            # QWEN36_MTP_SPEC_SAMPLE_SSP: no noise block; the one noise plan page keys fs_cut's made noise (the verify
            # pick's dev_sample key from _ds_block, the accept uniform / residual noise, and the chain's base position
            # p_b + 2: the device adds m_b from the device accept's m page and the depth).  HOSTSPLIT_MUSTFAIL moves
            # the chain base by one, as it moves the host chain noise without the flag.
            nds = getattr(self, "_np_ds", None)
            if nds is None or len(nds[0]) != M:
                raise RuntimeError(f"{fused_sampler.SSP_ENV}: the verify's dev_sample keys were not built this step")
            cb = [(int(plan.p[b]) + 2 if plan is not None else 2) + (1 if spec_sample.host_split_mustfail() else 0)
                  for b in range(bw)]
            out["v_snp"] = fused_sampler.noise_plan_page(nds[0], vpos, nds[2], vseeds, seeds, cb, ctemps)
            self._np_ds = None
            if "d" not in parts:
                return out
            return self._spec_d_block(plan, acc, out, seeds, temps, topks, topps, exact, ctemps, C2)
        if "d" not in parts:
            # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: the pre-verify write sends no d_ block; build the verify noise only
            if spec_sample.fast_noise_enabled():
                gv = spec_sample.noise_group([
                    {"seeds": vseeds, "positions": vpos, "stream": spec_sample.STREAM_U, "width": 1, "kind": "u01"},
                    {"seeds": vseeds, "positions": vpos, "stream": spec_sample.STREAM_RES, "width": C2,
                     "kind": "expo"},
                ])
                out["v_su"], out["v_snr"] = gv[0].reshape(1, 1, M, 1), gv[1].reshape(1, 1, M, C2)
            else:
                out["v_su"] = spec_sample.uniform(vseeds, vpos, spec_sample.STREAM_U).reshape(1, 1, M, 1)
                out["v_snr"] = spec_sample.expo(vseeds, vpos, C2, spec_sample.STREAM_RES).reshape(1, 1, M, C2)
            return out
        m = list(getattr(acc, "m", None) or [0] * bw)
        base = [int(plan.p[b]) + int(m[b]) + 2 if plan is not None else 0 for b in range(bw)]
        cpos = [[base[b] + k for b in range(bw)] for k in range(K)]
        if spec_sample.fast_noise_enabled():
            # QWEN36_MTP_SPEC_SAMPLE_HOSTNOISE: ONE hash per group of streams (the verify's two and
            # the chain's K), the values bit-identical to the per-stream path (spec_sample.noise_group).
            gv = spec_sample.noise_group([
                {"seeds": vseeds, "positions": vpos, "stream": spec_sample.STREAM_U, "width": 1, "kind": "u01"},
                {"seeds": vseeds, "positions": vpos, "stream": spec_sample.STREAM_RES, "width": C2, "kind": "expo"},
            ])
            out["v_su"] = gv[0].reshape(1, 1, M, 1)
            out["v_snr"] = gv[1].reshape(1, 1, M, C2)
            gc = spec_sample.noise_group([
                {"seeds": seeds, "positions": cpos[k], "stream": spec_sample.STREAM_CHAIN + k, "width": C2,
                 "kind": "gumbel", "temps": ctemps} for k in range(K)
            ])
            for k in range(K):
                out[f"d_sgum{k}"] = gc[k].reshape(1, 1, bw, C2)
            prm = spec_sample.param_rows_fast(temps, topks, topps, C2)
        else:
            out["v_su"] = spec_sample.uniform(vseeds, vpos, spec_sample.STREAM_U).reshape(1, 1, M, 1)
            out["v_snr"] = spec_sample.expo(vseeds, vpos, C2, spec_sample.STREAM_RES).reshape(1, 1, M, C2)
            for k in range(K):
                g = spec_sample.gumbel(seeds, cpos[k], C2, spec_sample.STREAM_CHAIN + k, temps=ctemps)
                out[f"d_sgum{k}"] = g.reshape(1, 1, bw, C2)
            prm = spec_sample.param_rows(temps, topks, topps, C2)
        out["d_sscale"] = prm["scale"].reshape(1, 1, bw, C2)
        out["d_srank"] = prm["rank"].reshape(1, 1, bw, C2)
        out["d_stopp"] = prm["topp"].reshape(1, 1, bw, 1)
        out["d_sflag"] = torch.tensor(exact, dtype=torch.float32).reshape(1, 1, bw, 1)
        return out

    def _spec_d_block(self, plan, acc, out, seeds, temps, topks, topps, exact, ctemps, C2):
        """QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: the chain's blocks only (the post-accept write), the same values as the
        d_ part of the full _spec_block.  HOSTSPLIT_MUSTFAIL shifts the chain positions by one: the boot
        self-test's comparison against the full builder must then refuse."""
        K, bw = self.K, self.bw
        m = list(getattr(acc, "m", None) or [0] * bw)
        base = [int(plan.p[b]) + int(m[b]) + 2 if plan is not None else 0 for b in range(bw)]
        if spec_sample.host_split_mustfail():
            base = [x + 1 for x in base]
        cpos = [[base[b] + k for b in range(bw)] for k in range(K)]
        if _ssp():
            # QWEN36_MTP_SPEC_SAMPLE_SSP: the chain's fs_cut makes d_sgum{k} (the noise plan page + the m page)
            prm = spec_sample.param_rows_fast(temps, topks, topps, C2)
        elif spec_sample.fast_noise_enabled():
            gc = spec_sample.noise_group([
                {"seeds": seeds, "positions": cpos[k], "stream": spec_sample.STREAM_CHAIN + k, "width": C2,
                 "kind": "gumbel", "temps": ctemps} for k in range(K)
            ])
            for k in range(K):
                out[f"d_sgum{k}"] = gc[k].reshape(1, 1, bw, C2)
            prm = spec_sample.param_rows_fast(temps, topks, topps, C2)
        else:
            for k in range(K):
                g = spec_sample.gumbel(seeds, cpos[k], C2, spec_sample.STREAM_CHAIN + k, temps=ctemps)
                out[f"d_sgum{k}"] = g.reshape(1, 1, bw, C2)
            prm = spec_sample.param_rows(temps, topks, topps, C2)
        out["d_sscale"] = prm["scale"].reshape(1, 1, bw, C2)
        out["d_srank"] = prm["rank"].reshape(1, 1, bw, C2)
        out["d_stopp"] = prm["topp"].reshape(1, 1, bw, 1)
        out["d_sflag"] = torch.tensor(exact, dtype=torch.float32).reshape(1, 1, bw, 1)
        return out

    def host_inputs(self, plan=None, acc=None, pt=None, want=None):
        """Host (torch) values for every persistent input, in _NAMES order. Without a plan: warmup
        values (every row active at position j, page table all zeros = vLLM's null block)."""
        bw, K, T, M = self.bw, self.K, self.T, self.M
        if plan is None:
            v_tok = [0] * M
            v_pos = [[j] * bw for j in range(T)]
            v_rpos = [j for j in range(T) for _ in range(bw)]
            onehot = [[1.0 if j == 0 else 0.0] * bw for j in range(T)]
            anchor = list(range(bw))
            t0n = [0] * bw
            rs_pos = [[j] * bw for j in range(T)]
            rs_rpos = list(v_rpos)
            d_pos = [[k] * bw for k in range(K)]
            d_rpos = [[k] * bw for k in range(K)]
            pt = torch.zeros(bw, self.nblk, dtype=torch.int32)
        else:
            v_tok, v_pos, v_rpos = plan.verify_tokens, plan.verify_pos, plan.verify_rope_pos
            onehot, anchor, t0n = acc.gdn_onehot, acc.anchor_idx, acc.t0_next
            rs_pos, rs_rpos = acc.reseed_pos, acc.reseed_rope_pos
            d_pos, d_rpos = acc.draft_pos, acc.draft_rope_pos
        i32 = torch.int32
        vals = {
            "v_tok": torch.tensor(v_tok, dtype=i32).reshape(M, 1),
            "v_rpos": torch.tensor(v_rpos, dtype=i32),
            "v_pt": pt.to(i32).contiguous(),
        }
        if self.mpt:
            sh = self.pt_shift if (plan is not None and self.pt_shift is not None) else [0] * bw
            vals["v_mpt"] = mbh.shifted_page_table(pt, sh).to(i32).contiguous()
        if _ds() and (want is None or want("v_gum")):
            # QWEN36_DEVICE_SAMPLE: the verify's per-step HOST block (the scaled noise T*g keyed by
            # the request's seed and the row's position, the select flag, the top_k rank mask and
            # the top_p threshold).  Without a plan (the warmup) the block is zeros/ones: only the
            # SHAPE matters at the capture.
            #
            # The names are in the "v_" namespace ON PURPOSE: the pre-verify write is
            # {n: v for n, v in vals.items() if n.startswith("v_")} (mtp_batch.py:2002/2126), so a
            # name outside it would be written AFTER the verify and the verify would read the
            # PREVIOUS step's noise and select flag (the s-sample2 review's blocker 2: one Gumbel
            # vector driving two emitted tokens, and a greedy row that reuses a slot freed by a
            # sampled row sampling instead of argmaxing).
            blk = self._ds_block(plan, M)
            if blk["noise"] is not None:  # None under QWEN36_MTP_SPEC_SAMPLE_SSP (fs_cut makes it)
                vals["v_gum"] = blk["noise"]
            vals["v_guf"] = blk["flag"]
            vals["v_gkr"] = blk["rank"]
            vals["v_gp"] = blk["topp"]
            # the per-row 1/T: vLLM's order is temperature -> top_k -> softmax -> top_p, so the
            # device tempers the candidate values BEFORE the softmax (the s-sample3 review's
            # blocker 2).
            vals["v_gs"] = blk["scale"]
        if _serve():
            # QWEN36_MTP_SPEC_SAMPLE (see _spec_block): the v_ blocks are written before the verify, the d_
            # blocks after the accept (the chain's noise is keyed by the accepted depth).
            # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT (want given): only the part this write sends is built.
            parts = ("v", "d") if want is None else tuple(p_ for p_, n_ in (("v", "v_sqp"), ("d", "d_sgum0")) if want(n_))
            if parts:
                vals.update(self._spec_block(plan, acc, M, parts=parts))
        if spec_fold_enabled():
            # QWEN36_SDPA_SPEC_FOLD: the folded verify SDPA's cur_pos, candidate-major (entry j*bw + b)
            vals["v_posf"] = torch.tensor([p for j in range(T) for p in v_pos[j]], dtype=i32)
        for j in range(T):
            vals[f"v_pos{j}"] = torch.tensor(v_pos[j], dtype=i32)
            vals[f"rs_pos{j}"] = torch.tensor(rs_pos[j], dtype=i32)
            if self.slot:
                continue
            vals[f"gmask{j}"] = torch.tensor(onehot[j], dtype=self.rec_mask_dtype).reshape(bw, 1, 1, 1)
            cm = torch.zeros(1, self.Bmax, 1, dtype=torch.bfloat16)
            cm[0, :bw, 0] = torch.tensor(onehot[j], dtype=torch.bfloat16)
            if j == T - 1:
                cm[0, bw:, 0] = 1.0  # rows outside the bucket keep the live conv taps
            vals[f"cmask{j}"] = cm
        if self.slot:
            vals["gslot"] = torch.tensor(mbh.slot_index(onehot, bw, T), dtype=torch.int32).reshape(1, mbh.SLOT_LEN)
        vals["rs_rpos"] = torch.tensor(rs_rpos, dtype=i32)
        vals["anchor"] = torch.tensor(anchor, dtype=i32).reshape(1, bw)
        vals["t0n"] = torch.tensor(t0n, dtype=i32).reshape(bw, 1)
        for k in range(K):
            vals[f"d_pos{k}"] = torch.tensor(d_pos[k], dtype=i32)
            vals[f"d_rpos{k}"] = torch.tensor(d_rpos[k], dtype=i32)
        if self.da:
            # QWEN36_MTP_DEVACCEPT: the warmup page (every row active, no usable draft) and zero noise tables; the step
            # writes the real ones through da_inputs()
            vals["da_plan"] = torch.tensor(mbh.da_warmup_page(bw), dtype=i32).reshape(1, mbh.DA_PAGE_LEN)
            if _serve() and not _ssp():  # QWEN36_MTP_SPEC_SAMPLE_SSP: no noise tables (fs_cut makes the chain noise)
                C2 = 2 * spec_sample.k()
                for k in range(K):
                    vals[f"d_sgtab{k}"] = torch.zeros((1, 1, (K + 1) * bw, C2), dtype=torch.float32)
        if want is not None:
            vals = {n: v for n, v in vals.items() if want(n)}
        return vals

    def da_inputs(self, plan, page):
        """QWEN36_MTP_DEVACCEPT: the device accept's own pre-verify inputs: the plan page and, under the served rule,
        the m-independent chain parameters (d_sscale / d_srank / d_stopp / d_sflag, the same values _spec_d_block
        writes) and the chain noise for EVERY possible accepted length (d_sgtab{k}: row i*bw + b = the d_sgum{k} row b
        that _spec_d_block writes for m_b = i; the device copies row m_b*bw + b). bk.spec_exact must be this step's
        (host_inputs built the v_ part first)."""
        out = {"da_plan": torch.tensor(page, dtype=torch.int32).reshape(1, mbh.DA_PAGE_LEN)}
        if _serve():
            out.update(self._spec_d_tables(plan))
        return out

    def _spec_d_tables(self, plan, shift=0):
        """QWEN36_MTP_DEVACCEPT + QWEN36_MTP_SPEC_SAMPLE_SERVE: see da_inputs. shift moves every table position (the
        tests' wrong-row mutant; 0 when served)."""
        K, bw = self.K, self.bw
        C2 = 2 * spec_sample.k()
        seeds, temps, topks, topps, rules = self._spec_params()
        greedy = [1 if t <= 0.0 else 0 for t in temps]
        exact = [1 if (rules[b] == spec_sample.RULE_EXACT and not greedy[b]) else 0 for b in range(bw)]
        ctemps = ([t if (exact[b] and t > 0.0) else 0.0 for b, t in enumerate(temps)] if spec_sample.draft_pick()
                  else [t if t > 0.0 else 0.0 for t in temps])
        R = (K + 1) * bw
        out = {}
        if _ssp():
            # QWEN36_MTP_SPEC_SAMPLE_SSP: the m-independent chain parameters only (the noise is made on device)
            prm = spec_sample.param_rows_fast(temps, topks, topps, C2)
            out["d_sscale"] = prm["scale"].reshape(1, 1, bw, C2)
            out["d_srank"] = prm["rank"].reshape(1, 1, bw, C2)
            out["d_stopp"] = prm["topp"].reshape(1, 1, bw, 1)
            out["d_sflag"] = torch.tensor(exact, dtype=torch.float32).reshape(1, 1, bw, 1)
            return out
        pos = mbh.da_noise_positions(plan, K)
        if shift:
            pos = [[x + shift for x in row] for row in pos]
        st = [seeds[b] for _ in range(K + 1) for b in range(bw)]
        ct = [ctemps[b] for _ in range(K + 1) for b in range(bw)]
        if spec_sample.fast_noise_enabled():
            gc = spec_sample.noise_group([
                {"seeds": st, "positions": pos[k], "stream": spec_sample.STREAM_CHAIN + k, "width": C2,
                 "kind": "gumbel", "temps": ct} for k in range(K)
            ])
            for k in range(K):
                out[f"d_sgtab{k}"] = gc[k].reshape(1, 1, R, C2)
            prm = spec_sample.param_rows_fast(temps, topks, topps, C2)
        else:
            for k in range(K):
                out[f"d_sgtab{k}"] = spec_sample.gumbel(st, pos[k], C2, spec_sample.STREAM_CHAIN + k,
                                                        temps=ct).reshape(1, 1, R, C2)
            prm = spec_sample.param_rows(temps, topks, topps, C2)
        out["d_sscale"] = prm["scale"].reshape(1, 1, bw, C2)
        out["d_srank"] = prm["rank"].reshape(1, 1, bw, C2)
        out["d_stopp"] = prm["topp"].reshape(1, 1, bw, 1)
        out["d_sflag"] = torch.tensor(exact, dtype=torch.float32).reshape(1, 1, bw, 1)
        return out

    def hostsplit_selftest(self, mf=False):
        """QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: the split builders against the full one, BITWISE, on a synthetic
        sampled plan (every row a different rule / T / top_k / top_p, chain blocks carried), for both the
        pre-verify part (v_, and rs_ as under EARLY_RESEED) and the post-accept part.  mf: the chain positions
        shifted by one (HOSTSPLIT_MUSTFAIL's wrong branch), which must differ.  Returns (diffs, names checked)."""
        import types as _types

        bw, K, T, M = self.bw, self.K, self.T, self.M
        C2 = 2 * spec_sample.k()
        g = torch.Generator().manual_seed(1021 + bw)
        rules = [spec_sample.RULE_EXACT, spec_sample.RULE_COMPARE, spec_sample.RULE_EXACT, spec_sample.RULE_EXACT]
        temps = [1.0, 1.0, 0.0, 0.7]
        sp_old = self.sp
        self.sp = _types.SimpleNamespace(
            seed=[11 + 7 * b for b in range(bw)], temperature=[temps[b % 4] for b in range(bw)],
            top_k=[[20, 20, 0, 10][b % 4] for b in range(bw)], top_p=[[0.95, 0.95, 1.0, 0.8][b % 4] for b in range(bw)],
            spec_rule=[rules[b % 4] for b in range(bw)])
        p0 = [100 + 13 * b for b in range(bw)]
        plan = mbh.Plan(bw=bw, K=K, T=T, active=[True] * bw, p=p0, t0=[5] * bw, n_draft=[K] * bw,
                        drafts=[[7 + b + k for k in range(K)] for b in range(bw)], origin=[0] * bw,
                        verify_tokens=[1] * M, verify_pos=[[p0[b] + j for b in range(bw)] for j in range(T)],
                        verify_rope_pos=[p0[b] + j for j in range(T) for b in range(bw)], kv_capacity=[1 << 20] * bw,
                        qprobs=torch.rand(K * bw, C2, generator=g), qids=torch.randint(0, 1 << 17, (K * bw, C2),
                                                                                       generator=g).float())
        m_ = [b % (K + 1) for b in range(bw)]
        acc = _types.SimpleNamespace(m=m_, gdn_onehot=[[1.0 if m_[b] == j else 0.0 for b in range(bw)] for j in range(T)],
                                     anchor_idx=list(range(bw)), t0_next=[3] * bw,
                                     reseed_pos=[[p0[b] + j for b in range(bw)] for j in range(T)],
                                     reseed_rope_pos=[p0[b] + j for j in range(T) for b in range(bw)],
                                     draft_pos=[[k] * bw for k in range(K)], draft_rope_pos=[[k] * bw for k in range(K)])
        pt = torch.arange(bw * self.nblk, dtype=torch.int32).reshape(bw, self.nblk)
        n = 0
        names = 0
        try:
            for first in (("v_",), ("v_", "rs_")):
                hs, spec_sample._HOSTSPLIT = spec_sample._HOSTSPLIT, False  # the reference: the served builders
                mf0, spec_sample._HOSTSPLIT_MF = spec_sample._HOSTSPLIT_MF, False
                try:
                    full = self.host_inputs(plan=plan, acc=acc, pt=pt)
                finally:
                    spec_sample._HOSTSPLIT = hs
                    spec_sample._HOSTSPLIT_MF = mf0
                # QWEN36_MTP_SPEC_SAMPLE_SSP: the chain's base position is in the pre-verify noise plan page (v_snp),
                # so the shifted-chain branch reaches the v_ part too
                mf_a, spec_sample._HOSTSPLIT_MF = spec_sample._HOSTSPLIT_MF, (bool(mf) if _ssp() else False)
                try:
                    a = self.host_inputs(plan=plan, acc=acc, pt=pt, want=lambda n_, f=first: n_.startswith(f))
                finally:
                    spec_sample._HOSTSPLIT_MF = mf_a
                mf_env, spec_sample._HOSTSPLIT_MF = spec_sample._HOSTSPLIT_MF, bool(mf)
                try:
                    b_ = self.host_inputs(plan=plan, acc=acc, pt=pt, want=lambda n_, f=first: not n_.startswith(f))
                finally:
                    spec_sample._HOSTSPLIT_MF = mf_env
                got = dict(a)
                got.update(b_)
                if set(got) != set(full):
                    n += len(set(got) ^ set(full))
                for k_, v in full.items():
                    if k_ in got and (got[k_].dtype != v.dtype or got[k_].shape != v.shape or not torch.equal(got[k_], v)):
                        n += 1
                names += len(full)
        finally:
            self.sp = sp_old
        return n, names


_MIB = float(1 << 20)


def _trace_region(mesh):
    """(allocated, free) bytes of the device TRACE region, or None when the build lacks the view."""
    try:
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        nb = int(mv.num_banks)
        return int(mv.total_bytes_allocated_per_bank) * nb, int(mv.total_bytes_free_per_bank) * nb
    except Exception:  # accounting only; never blocks the capture
        return None


_CLASS_ABLATE_KNOWN = ("mlp", "attn", "lmhead")
_CLASS_ABLATE_LOGGED = set()


def verify_class_ablate():
    """QWEN36_MTP_VERIFY_CLASS_ABLATE: comma set of {mlp, attn, lmhead}. A timing diagnostic that drops one op
    class across the WHOLE verify (all rows, all candidates): mlp = the 64 base-layer MLPs return a copy of their
    input; attn = the 16 full-attention mixers return a copy of their input (no KV write); lmhead = no lm_head
    matmul and no argmax (the verified ids are a copy of the input tokens). The outputs are WRONG, so it is never a
    speed or correctness number; the per-step verify p50 delta against the unablated arm is that class's cost."""
    v = _tt_public_getenv("QWEN36_MTP_VERIFY_CLASS_ABLATE", "")
    s = {x.strip() for x in v.split(",") if x.strip() and x.strip() != "0"}
    bad = s - set(_CLASS_ABLATE_KNOWN)
    if bad:
        raise RuntimeError(f"QWEN36_MTP_VERIFY_CLASS_ABLATE: unknown {sorted(bad)} (known: {', '.join(_CLASS_ABLATE_KNOWN)})")
    return s


class _ClassAblate:
    """Instance-level patches of the base layers for the duration of one _verify_ops call (eager or capture)."""

    def __init__(self, layers, abl):
        self.layers, self.abl, self.saved = layers, abl, []

    def __enter__(self):
        def _copy_in(x, *a, **k):
            return ttnn.clone(x)

        for layer in self.layers:
            if "mlp" in self.abl:
                self.saved.append((layer.feed_forward, "forward", layer.feed_forward.__dict__.get("forward")))
                layer.feed_forward.forward = _copy_in
            if "attn" in self.abl and layer.is_full_attention:
                self.saved.append((layer.attention, "forward_decode", layer.attention.__dict__.get("forward_decode")))
                layer.attention.forward_decode = _copy_in
        return self

    def __exit__(self, *exc):
        for obj, name, prev in reversed(self.saved):
            if prev is None:
                del obj.__dict__[name]
            else:
                setattr(obj, name, prev)
        self.saved = []
        return False


def _to_host_tt(name, t):
    if name == "v_snp":  # QWEN36_MTP_SPEC_SAMPLE_SSP: the noise plan page (int32 words)
        return ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    if name.startswith(("gmask", "cmask")):
        dt = ttnn.float32 if t.dtype == torch.float32 else ttnn.bfloat16
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT)
    if name in ("v_gum", "v_guf", "v_gkr", "v_gp", "v_gs") or name.startswith(("v_s", "d_s")):
        # QWEN36_DEVICE_SAMPLE: the Gumbel noise, the select flag, the top_k rank mask, the
        # top_p threshold and the per-row 1/T (fp32, row-major)
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    if name in ("v_tok", "t0n", "anchor", "gslot", "da_m"):
        return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _read_ids(t, n):
    return [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)[:n].tolist()]


def _spec_key(t):
    """What a program compiled for tensor t is keyed on (per-device shape, dtype, layout, memory config)."""
    return (tuple(int(x) for x in t.shape), str(t.dtype), str(t.layout), str(t.memory_config()))


class _MTPWarm:
    """QWEN36_MTP_WARM_TAIL=N (docs/MTP-BATCH.md section 13): a KV-only MTP pass over the last n = min(N, L) prompt
    positions at prefill, so the drafter's first steps attend to prompt context instead of an empty cache (R3).

    Per request (model.prefill_paged_slots): begin(L) -> segment() for every prefill segment whose hidden is still
    on the device (the last full chunk, from the chunk trace's output, right after the last replay and BEFORE the
    tail's replay; the tail / short-prompt masked bucket, traced or eager) -> finish() once the host logits are read.

      gather   per segment: hidden [1,1,S,D] (D = dim/TP, the residual leaving the last layer, fractured) ->
               row-major [S, D] + 32 zero rows -> ttnn.embedding with a [1, Wn] row index (window row i ->
               segment row of position w0+i, else a zero row) -> ADDED into the persistent window buffer. A copy
               and an add of exact zeros: the window equals the hidden rows bit for bit (ttsim record).
      warm     h = final norm of the window (distributed, no gather: fractured, as the B=1 port's spec_norm),
               e = embed(token of position s+1, the argmax of the prefill logits for the last row), then the
               decode head's own chain in PREFILL mode: fc([enorm(e), hnorm(h)]) -> the MTP decoder layer with
               kv_only (attention norm, QKV, k-norm, RoPE at absolute w0+i, cast to the cache dtypes,
               paged_fill_cache at relative index 0 through a fixed-width fill table whose pad blocks map to a
               scratch block). No SDPA / wo / MLP: only the KV write survives, as in the decode reseed.

    Every shape is fixed per boot (Wn = warm_bucket(N); one gather program per source shape), so no program
    compiles at request time: compile_from() runs once per source spec during the prefill warmup (before any
    trace is parked) and a source whose spec was not compiled is skipped (counted, logged), never compiled.
    All device inputs are persistent buffers allocated with the MTP KV (before every trace capture), written by
    host->device copies. No trace is added, so the TRACE region is unchanged. A request whose window the
    segments do not cover (an eager chunked path, a lost segment) is skipped: its slot starts cold, as before."""

    def __init__(self, mtp, N, mustfail_mode, span=0, span_mustfail=False, early=False, early_mustfail=False):
        from models.tt_transformers.tt.common import Mode
        from models.tt_transformers.tt.distributed_norm import DistributedNorm

        self.mtp, self.model = mtp, mtp.model
        self.mesh = self.model.mesh_device
        self.N, self.Wn, self.mustfail = int(N), mbh.warm_bucket(N), mustfail_mode
        args = self.model.args
        self._Mode = Mode
        if not args.is_distributed_norm(Mode.PREFILL):
            raise RuntimeError("QWEN36_MTP_WARM_TAIL refused: needs the distributed PREFILL norm (fractured final norm)")
        # The base model's final norm with its trailing all-gather off: the fractured final-norm output, the hidden the
        # MTP pre-fc norm takes in PREFILL mode (same weights as model.norm, so the same values the logits use).
        self.base_norm_frac = DistributedNorm(
            self.model.norm.norm, args, tt_ccl=self.model.tt_ccl, TG=args.is_galaxy, enable_all_gather=False
        )
        self.D = int(args.dim) // int(self.model.num_devices)
        self.block_size = 64  # the paged-KV block; MTPBatch.allocate_kv replaces it from the KV shape (2048: prefix arm)
        self.pad_block = None
        self.bufs = None
        self.compiled = {}  # source spec key -> rows S
        self.ops_compiled = False
        self.req = None
        self.stats = {"warmed": 0, "skipped": {}}
        self._prefix_pending = {}  # QWEN36_PREFIX_CACHING: request id -> (w0, total, passes issued by its head steps)
        # QWEN36_PD_EXPORT_MTP (vllm_tt_plugin/pd_export.py): pd_hook(req_id) -> None (not an MTP export) or the token
        # the last warmed row pairs with (>= 0: the prompt token after TT's prompt, which the P/D router knows and TT
        # was not given; -1: keep the prefill argmax). pd_spans[req_id] = (origin, hit, total, last token) of a warmed
        # export request, popped by the exporter at the request's final step.
        self.pd_hook = None
        self.pd_spans = {}
        logger.info(
            f"QWEN36_MTP_WARM_TAIL engaged N={self.N}: KV-only MTP pass over the last min(N, L) prompt positions at "
            f"prefill (device rows {self.Wn}, origin = first warmed position, RoPE absolute, (h_s, tok_s+1) pairing, "
            f"last row paired with the prefill argmax)"
        )
        if self.mustfail:
            logger.info(
                f"QWEN36_MTP_WARM_TAIL_MUSTFAIL engaged: {self.mustfail} (warm rows pair h_s with tok_s, one position "
                "early: acceptance must DROP vs the warm arm; the stream stays greedy-exact)"
            )
        # QWEN36_MTP_WARM_TAIL_SPAN=k: up to k passes of N rows (the same Wn programs), k window buffers.
        self.span, self.span_mustfail = int(span), bool(span_mustfail)
        self.wins = None
        self.span_selftested = False
        self._anchor = []  # host tensors of in-flight span writes, held until finish() (after the logits readback)
        if self.span:
            mbh.warm_span_check(self.N, self.span, self.block_size)
            self.stats["span_rebased"] = 0
            logger.info(
                f"QWEN36_MTP_WARM_TAIL_SPAN engaged: k={self.span} passes of N={self.N} rows (device rows {self.Wn} "
                f"each, the N-window programs), window up to {self.span * self.N} prompt positions; every chunk replay "
                f"meeting the window is gathered right after its replay into one of {self.span} [Wn, D] buffers"
            )
            if self.span_mustfail:
                logger.info(
                    f"{mbh.ENV_WARM_SPAN_MUSTFAIL} engaged: span gathers read the hidden row one position late "
                    "(the warmup self-test must REFUSE the boot)"
                )
        # QWEN36_MTP_WARM_TAIL_SPAN_EARLY: every pass but the last runs its KV pass right after its window is complete,
        # under the chunk loop (same CQ as the non-blocking replays), so its host dispatch and host-side input prep
        # overlap the device's replay of the next chunk instead of running after the logits readback with the CQ
        # drained. The last pass (its last row pairs with the prefill argmax) still runs in finish().
        self.early, self.early_mustfail = bool(early), bool(early_mustfail)
        self.early_selftested = False
        if self.early:
            self.stats["span_early"] = 0
            self.stats["span_early_wasted"] = 0
            logger.info(
                f"QWEN36_MTP_WARM_TAIL_SPAN_EARLY engaged: span passes 0..{self.span - 2} run under the chunk loop as "
                f"soon as their window is complete (same CQ, between chunk replays); only pass {self.span - 1} (last "
                f"row = prefill argmax) waits for the logits. Same programs, same inputs, same per-pass order."
            )
            if self.early_mustfail:
                logger.info(
                    f"{mbh.ENV_WARM_SPAN_EARLY_MUSTFAIL} engaged: early passes pair h_s with tok_s (one position early) "
                    "(the warmup self-test must REFUSE the boot)"
                )

    # ---------------------------------------------------------------- buffers
    def allocate(self, pad_block):
        """Persistent inputs, allocated with the MTP KV (before any trace capture, so no trace intermediate can
        alias them). pad_block: the MTP cache's scratch block (one block past the scheduler's)."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        Wn, D = self.Wn, self.D
        self.pad_block = int(pad_block)

        def up(t, dt, layout, mapper=rep):
            return ttnn.from_torch(t, dtype=dt, layout=layout, device=self.mesh, mesh_mapper=mapper,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        cos_t, sin_t = self._rope_tables(0, Wn)
        self.bufs = {
            "win": up(torch.zeros(1, 1, Wn, D * int(self.model.num_devices), dtype=torch.bfloat16), ttnn.bfloat16,
                      ttnn.TILE_LAYOUT, ttnn.ShardTensorToMesh(self.mesh, dim=3)),
            "idx": up(torch.zeros(1, Wn, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "tok": up(torch.zeros(1, Wn, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "cos": up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "sin": up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "fill": up(torch.full((1, -(-Wn // self.block_size)), self.pad_block, dtype=torch.int32), ttnn.int32,
                       ttnn.ROW_MAJOR_LAYOUT),
            "zero": up(torch.zeros(32, D, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        }
        if self.span:
            # QWEN36_MTP_WARM_TAIL_SPAN: window buffer j holds pass j's rows; pass 0 reuses "win".
            self.wins = [self.bufs["win"]] + [
                up(torch.zeros(1, 1, Wn, D * int(self.model.num_devices), dtype=torch.bfloat16), ttnn.bfloat16,
                   ttnn.TILE_LAYOUT, ttnn.ShardTensorToMesh(self.mesh, dim=3))
                for _ in range(self.span - 1)
            ]

    def _rope_tables(self, start, n):
        """[1,1,Wn,rd] prefill-format cos/sin for absolute positions start..start+n-1, pad rows repeating the last."""
        cos_t, sin_t = self.model._rope_tp_cos_sin_torch(int(start), int(n))
        if n < self.Wn:
            rep = self.Wn - n
            cos_t = torch.cat([cos_t, cos_t[:, :, -1:, :].expand(-1, -1, rep, -1)], dim=2)
            sin_t = torch.cat([sin_t, sin_t[:, :, -1:, :].expand(-1, -1, rep, -1)], dim=2)
        return cos_t.contiguous(), sin_t.contiguous()

    def _write(self, vals):
        from models.tt_transformers.tt.common import copy_host_to_device

        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        spec = {"idx": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), "tok": (ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "cos": (ttnn.bfloat16, ttnn.TILE_LAYOUT), "sin": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
                "fill": (ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)}
        names = list(vals)
        host = [ttnn.from_torch(vals[n], dtype=spec[n][0], layout=spec[n][1], mesh_mapper=rep) for n in names]
        copy_host_to_device(host, device_tensors=[self.bufs[n] for n in names])
        return host

    # ---------------------------------------------------------------- device ops
    def _gather_add(self, hidden, S, first):
        """window (+)= embedding(idx, [rows(hidden); 32 zero rows]). The idx buffer must hold this segment's index."""
        h = hidden if hidden.dtype == ttnn.bfloat16 else ttnn.typecast(hidden, ttnn.bfloat16)
        rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        if h is not hidden:
            ttnn.deallocate(h)
        rm = ttnn.reshape(rm, (int(S), self.D))
        w = ttnn.concat([rm, self.bufs["zero"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rm)
        g = ttnn.embedding(self.bufs["idx"], w, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(w)
        g = ttnn.reshape(g, (1, 1, self.Wn, self.D))
        if first:
            ttnn.copy(g, self.bufs["win"])
        else:
            s = ttnn.add(self.bufs["win"], g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(s, self.bufs["win"])
            ttnn.deallocate(s)
        ttnn.deallocate(g)

    def _warm_ops(self, win=None):
        """The KV-only MTP pass over the window buffer (fixed shapes: Wn rows). win: a span window buffer (default
        the single window)."""
        Mode = self._Mode
        head, model, b = self.mtp.head, self.model, self.bufs
        e = model.embd(b["tok"])
        e = ttnn.reshape(e, (1, 1, self.Wn, e.shape[-1]))
        e = ttnn.to_memory_config(e, ttnn.DRAM_MEMORY_CONFIG)
        e_n = head.enorm(e, mode=Mode.PREFILL)  # fractured in -> full width
        ttnn.deallocate(e)
        hf = self.base_norm_frac(b["win"] if win is None else win, mode=Mode.PREFILL)  # fractured final-norm output
        h_n = head.hnorm(hf, mode=Mode.PREFILL)
        ttnn.deallocate(hf)
        cat = ttnn.concat([e_n, h_n], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [embedding, hidden]
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        fused = ttnn.linear(cat, head.fc, compute_kernel_config=tpc.COMPUTE_HIFI2,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,Wn,dim/tp] fractured
        ttnn.deallocate(cat)
        out = head.decoder.forward(
            fused, cos=b["cos"], sin=b["sin"], mode="prefill", page_table=b["fill"], chunk_page_table=b["fill"],
            chunk_start_idx=0, kv_only=True,
        )
        ttnn.deallocate(fused)
        assert out is None, "QWEN36_MTP_WARM_TAIL: the kv_only prefill returned an output"

    # ---------------------------------------------------------------- warmup
    def compile_from(self, hidden, S):
        """Prefill warmup hook (model.py, eager, before any trace is parked): compile the gather for this source
        spec (both the copy and the add form) and, once, the warm pass. Writes only the MTP scratch block."""
        if self.bufs is None:
            return
        key = _spec_key(hidden)
        if key in self.compiled:
            return
        S = int(S)
        idx = torch.full((1, self.Wn), S, dtype=torch.int32)
        n = min(S, self.Wn)
        idx[0, :n] = torch.arange(n, dtype=torch.int32)
        self._write({"idx": idx})
        self._gather_add(hidden, S, first=True)
        self._gather_add(hidden, S, first=False)
        if not self.ops_compiled:
            self._write({"fill": torch.full((1, -(-self.Wn // self.block_size)), self.pad_block, dtype=torch.int32)})
            self._warm_ops()
            self.ops_compiled = True
        ttnn.synchronize_device(self.mesh)
        self.compiled[key] = S
        logger.info(f"QWEN36_MTP_WARM_TAIL compiled: source {list(key[0])} {key[1]} rows {S} -> window {self.Wn}")
        if self.span and not self.span_selftested:
            self._span_selftest(hidden, S)
        if self.early and not self.early_selftested:
            self._span_early_selftest()

    # ---------------------------------------------------------------- per request
    def _skip(self, why):
        n = self.stats["skipped"][why] = self.stats["skipped"].get(why, 0) + 1
        if n == 1 or n % 256 == 0:
            logger.info(f"QWEN36_MTP_WARM_TAIL skipped ({why}): count={n} (the slot starts cold)")

    def begin(self, L, prompt_ids=None, page_table_row=None):
        """prompt_ids / page_table_row: the request's tokens and page-table row, needed before the logits only by
        QWEN36_MTP_WARM_TAIL_SPAN_EARLY (finish() still receives them)."""
        self.req = None
        if self.bufs is None or not self.ops_compiled:
            self._skip("not compiled")
            return
        if self.span:
            # NOT cleared here: the previous user's finish() anchors (its span writes, queued non-blocking) are still
            # in flight when the next user of the SAME prefill step begins; only the next finish(), which runs after
            # this user's blocking logits readback (same CQ, so those writes have drained), replaces them.
            # (pd-r3 G9 2026-11-02: a 4-user step at span 16 dropped them, then the next prefill hung the device.)
            if not self.span_selftested:
                self._skip("span self-test not run")
                return
            sp = mbh.warm_span_window(int(L), self.N, self.span)
            self.req = {"span": sp, "covered": [0] * sp.passes, "first": [True] * sp.passes, "bad": None,
                        "issued": [False] * sp.passes, "prompt": None, "pt": None}
            if self.early and self.early_selftested:
                if prompt_ids is None or page_table_row is None:
                    self._skip("early: begin() without prompt/page table")
                    self.req = None
                    return
                self.req["prompt"] = [int(t) for t in prompt_ids]
                self.req["pt"] = page_table_row
            return
        self.req = {"win": mbh.warm_window(int(L), self.N), "covered": 0, "first": True, "bad": None}

    def abort(self):
        self.req = None

    # ---------------------------------------------------------------- QWEN36_PREFIX_CACHING (S12 prefix arm)
    def begin_prefix(self, req_id, total, hit, final):
        """The warm span of one PREFILL STEP of a prefix-cache request: vLLM's align mode prefills a long prompt as a
        head step ending on a block boundary, then the tail, and a cache hit prefills only [hit, total). The passes are
        anchored on the N grid (mbh.prefix_warm_window), so each one is gathered inside ONE step; a head step issues
        the passes it completed (their (h_s, tok_s+1) pairs are all prompt tokens), the final step issues the rest and
        seeds the slot. Every pass is written at relative index j*N from the window origin w0 >= hit, through the
        slot's shifted MTP page table (no write lands in a shared cached block). Needs the span pass
        (QWEN36_MTP_WARM_TAIL_SPAN) and not its early variant."""
        self.req = None
        if self.bufs is None or not self.ops_compiled:
            self._skip("not compiled")
            return
        if not self.span or self.early:
            self._skip("prefix: needs QWEN36_MTP_WARM_TAIL_SPAN without _EARLY")
            return
        if not self.span_selftested:
            self._skip("span self-test not run")
            return
        if int(hit) % self.N:
            self._skip(f"prefix: hit {hit} not on the N={self.N} grid")
            return
        w0, passes = mbh.prefix_warm_window(int(total), int(hit), self.N, self.span, self.block_size)
        sp = mbh.WarmSpan(L=int(total), n=int(total) - w0, w0=w0, N=self.N, Wn=self.Wn, passes=len(passes))
        prev = self._prefix_pending.pop(req_id, None)
        issued = list(prev[2]) if (prev is not None and prev[:2] == (w0, int(total))) else [False] * sp.passes
        # self._anchor is NOT cleared here (see begin): a head step's finish() of the previous user in this prefill
        # step queued its span writes non-blocking and nothing has drained them yet.
        self.req = {"span": sp, "covered": [0] * sp.passes, "first": [True] * sp.passes, "bad": None,
                    "issued": [False] * sp.passes, "prompt": None, "pt": None,
                    "prefix": {"req": req_id, "final": bool(final), "prev": issued, "hit": int(hit)}}

    def _prefix_finish(self, r, prompt_ids, logits, slot, page_table_row):
        """See begin_prefix. prompt_ids: [total] (a head step passes its prompt up to the step end, the next prompt
        token, then zeros: no issued row reads past the step end + 1). page_table_row: the slot's SHIFTED row."""
        sp, pre = r["span"], r["prefix"]
        if r["bad"]:
            self._skip(r["bad"])
            return False
        done = [r["covered"][j] == sp.sub(j)[1] for j in range(sp.passes)]
        prompt = [int(t) for t in prompt_ids]
        if not pre["final"]:
            anchor = []
            issued = list(pre["prev"])
            for j in range(sp.passes):
                if done[j] and not issued[j]:
                    anchor += self._span_issue(sp, j, 0, prompt, None, page_table_row)
                    issued[j] = True
            self._anchor = anchor
            self._prefix_pending[pre["req"]] = (sp.w0, sp.L, issued)
            while len(self._prefix_pending) > 64:  # a request aborted between its head and tail steps never pops
                self._prefix_pending.pop(next(iter(self._prefix_pending)))
            self.stats["prefix_head"] = self.stats.get("prefix_head", 0) + 1
            return True
        prev = pre["prev"]
        if all(prev[j] or done[j] for j in range(sp.passes)):
            j0 = 0
            todo = [j for j in range(sp.passes) if not prev[j]]
        else:
            # a head pass was lost (preempted, skipped): the passes covered NOW are re-issued from the first of their
            # contiguous suffix, which becomes the origin (block-aligned), as _span_finish rebases
            j0 = mbh.warm_span_covered_from(sp, r["covered"])
            todo = list(range(j0, sp.passes))
            if j0:
                self.stats["span_rebased"] = self.stats.get("span_rebased", 0) + 1
        if j0 >= sp.passes:
            self._skip(f"prefix span window covered {r['covered']} of {[sp.sub(j)[1] for j in range(sp.passes)]}")
            return False
        first = int(torch.argmax(logits.reshape(-1)[: int(self.model.args.vocab_size)]).item())
        pd = None
        if self.pd_hook is not None:
            try:
                pd = self.pd_hook(pre["req"])
            except Exception as e:  # noqa: BLE001 - an export hook never fails the prefill
                logger.warning(f"QWEN36_PD_EXPORT_MTP hook failed for {pre['req']}: {type(e).__name__}: {e}")
                pd = None
            if pd is not None and int(pd) >= 0:
                first = int(pd)  # the export's real next prompt token (TT was given the prompt minus it)
        anchor = []
        for j in todo:
            anchor += self._span_issue(sp, j, j0, prompt, first, page_table_row)
        self._anchor = anchor
        origin = sp.sub(j0)[0]
        self.mtp.host.warm_slot(int(slot), origin, sp.L, first)
        if pd is not None:
            self.pd_spans[pre["req"]] = (int(origin), int(pre.get("hit", 0)), int(sp.L), int(first))
            while len(self.pd_spans) > 256:  # a request the exporter never collected (aborted) never pops
                self.pd_spans.pop(next(iter(self.pd_spans)))
        self.stats["warmed"] += 1
        self.stats["prefix_warmed"] = self.stats.get("prefix_warmed", 0) + 1
        if self.stats["prefix_warmed"] == 1 or self.stats["prefix_warmed"] % 256 == 0:
            logger.info(
                f"QWEN36_MTP_WARM_TAIL prefix warmed: L={sp.L} origin={origin} slot={int(slot)} first={first} "
                f"passes={sp.passes - j0}/{sp.passes} (from earlier steps {sum(prev[j0:])}) "
                f"count={self.stats['prefix_warmed']} heads={self.stats.get('prefix_head', 0)} "
                f"skipped={self.stats['skipped']}"
            )
        return True

    def segment(self, hidden, seg_start, valid, S):
        """A prefill segment's hidden rows [0, valid) = absolute positions seg_start.. are on the device NOW
        (the caller replays the next trace only after this returns)."""
        r = self.req
        if r is None or r["bad"]:
            return
        if self.span:
            self._span_segment(r, hidden, seg_start, valid, S)
            return
        idx, cov = mbh.warm_gather_index(r["win"], seg_start, valid, zero_row=S)
        if idx is None:
            return
        if self.compiled.get(_spec_key(hidden)) != int(S):
            r["bad"] = f"source {list(hidden.shape)} {hidden.dtype} not compiled"
            return
        self._write({"idx": torch.tensor(idx, dtype=torch.int32).reshape(1, -1)})
        self._gather_add(hidden, S, first=r["first"])
        r["first"] = False
        r["covered"] += cov

    def finish(self, prompt_ids, logits, slot, page_table_row):
        """Run the warm pass for the request begun last and seed its MTP slot. logits: host [1,1,vocab] (the
        prefill's); its argmax is the greedy first token, which is the first decode step's t0.

        Anchors: a warm that runs replaces self._anchor with its own writes' host tensors. A skipped or failed one
        (False) clears it at the same point, so the per-segment idx tensors _span_gather appended for a request whose
        warm never ran do not accumulate (begin() no longer clears them, see there). Same drain argument as the
        success path: finish() runs after this request's blocking logits readback on the same CQ, so every write
        anchored so far has completed."""
        ok = self._finish(prompt_ids, logits, slot, page_table_row)
        if not ok:
            self._anchor = []
        return ok

    def _finish(self, prompt_ids, logits, slot, page_table_row):
        r, self.req = self.req, None
        if r is None:
            return False
        if r.get("prefix") is not None:
            return self._prefix_finish(r, prompt_ids, logits, slot, page_table_row)
        if self.span:
            return self._span_finish(r, prompt_ids, logits, slot, page_table_row)
        win = r["win"]
        if r["bad"]:
            self._skip(r["bad"])
            return False
        if r["covered"] != win.n:
            self._skip(f"window covered {r['covered']}/{win.n}")
            return False
        first = int(torch.argmax(logits.reshape(-1)[: int(self.model.args.vocab_size)]).item())
        prompt = [int(t) for t in prompt_ids]
        toks = mbh.warm_tokens(win, prompt, first, self.mustfail)
        cos_t, sin_t = self._rope_tables(win.w0, win.n)
        from models.demos.blackhole.qwen36.tt import masked_bucket_trace as mbt

        fill = mbt.fill_pt_row(page_table_row, 0, win.n, win.Wn, self.pad_block, self.block_size, mapped_pad=False)
        self._write({"tok": torch.tensor(toks, dtype=torch.int32).reshape(1, -1), "cos": cos_t, "sin": sin_t,
                     "fill": fill})
        self._warm_ops()
        self.mtp.host.warm_slot(int(slot), win.w0, win.L, first)
        self.stats["warmed"] += 1
        if self.stats["warmed"] == 1 or self.stats["warmed"] % 256 == 0:
            logger.info(
                f"QWEN36_MTP_WARM_TAIL warmed: L={win.L} n={win.n} origin={win.w0} slot={int(slot)} first={first} "
                f"count={self.stats['warmed']} skipped={self.stats['skipped']}"
            )
        return True

    # ---------------------------------------------------------------- QWEN36_MTP_WARM_TAIL_SPAN
    def _span_targets(self, sp, seg_start, valid, S, shift=None):
        """[(pass j, idx, covered)] for every span pass this segment meets. shift: the must-fail (default: the env)."""
        shift = self.span_mustfail if shift is None else bool(shift)
        out = []
        for j in range(sp.passes):
            idx, cov = mbh.warm_span_gather_index(sp, j, seg_start, valid, zero_row=int(S), shift=shift)
            if idx is not None:
                out.append((j, idx, cov))
        return out

    def _span_gather(self, hidden, S, r, targets):
        """window[j] (+)= embedding(idx_j, [rows(hidden); 32 zero rows]) for every target pass j: the same op sequence
        and shapes as _gather_add (so no program is new), with the row-major source built once per segment."""
        h = hidden if hidden.dtype == ttnn.bfloat16 else ttnn.typecast(hidden, ttnn.bfloat16)
        rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        if h is not hidden:
            ttnn.deallocate(h)
        rm = ttnn.reshape(rm, (int(S), self.D))
        w = ttnn.concat([rm, self.bufs["zero"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rm)
        for j, idx, cov in targets:
            # Host tensors stay referenced until finish() (after the logits readback): these writes are queued beside
            # non-blocking chunk replays.
            self._anchor += self._write({"idx": torch.tensor(idx, dtype=torch.int32).reshape(1, -1)})
            g = ttnn.embedding(self.bufs["idx"], w, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            g = ttnn.reshape(g, (1, 1, self.Wn, self.D))
            win = self.wins[j]
            if r["first"][j]:
                ttnn.copy(g, win)
            else:
                s = ttnn.add(win, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(s, win)
                ttnn.deallocate(s)
            ttnn.deallocate(g)
            r["first"][j] = False
            r["covered"][j] += cov
        ttnn.deallocate(w)

    def _span_segment(self, r, hidden, seg_start, valid, S):
        targets = self._span_targets(r["span"], seg_start, valid, S)
        if not targets:
            return
        if self.compiled.get(_spec_key(hidden)) != int(S):
            r["bad"] = f"source {list(hidden.shape)} {hidden.dtype} not compiled"
            return
        self._span_gather(hidden, S, r, targets)
        if self.early and r["prompt"] is not None:
            # QWEN36_MTP_WARM_TAIL_SPAN_EARLY: a pass this segment completed (and every pass before it) runs now,
            # queued behind its gather on the same CQ, ahead of the next chunk replay the caller enqueues after this.
            for j, _idx, _cov in targets:
                if not r["issued"][j] and mbh.warm_span_early_ready(r["span"], r["covered"], j):
                    self._anchor += self._span_issue(r["span"], j, 0, r["prompt"], None, r["pt"],
                                                     shift=self.early_mustfail)
                    r["issued"][j] = True
                    self.stats["span_early"] += 1

    def _span_issue(self, sp, j, j0, prompt, first, page_table_row, shift=False):
        """Write pass j's inputs (tokens, RoPE tables at its absolute positions, the fill table at relative index
        (j - j0) * N) and run the KV pass over window buffer j. ONE builder for the early and the late path, so both
        write the same KV. shift: the early must-fail pairing (h_s with tok_s). Returns the host tensors to anchor."""
        from models.demos.blackhole.qwen36.tt import masked_bucket_trace as mbt

        sw = sp.sub_window(j)
        toks = mbh.warm_tokens(sw, prompt, first, "pair_minus1" if shift else self.mustfail)
        cos_t, sin_t = self._rope_tables(sw.w0, sw.n)
        fill = mbt.fill_pt_row(page_table_row, (j - j0) * sp.N, sw.n, sw.Wn, self.pad_block, self.block_size,
                               mapped_pad=False)
        host = self._write({"tok": torch.tensor(toks, dtype=torch.int32).reshape(1, -1), "cos": cos_t, "sin": sin_t,
                            "fill": fill})
        self._warm_ops(self.wins[j])
        return host

    def _span_finish(self, r, prompt_ids, logits, slot, page_table_row):
        """Up to k KV-only passes (the N-window programs), pass j at relative index j*N (origin = first warmed
        position). If the earliest passes were not covered (resumed prefix, eager chunk path), the covered suffix of
        passes is warmed with the origin moved to its first pass (block-aligned); none covered = skipped (cold)."""
        sp = r["span"]
        if r["bad"]:
            self._skip(r["bad"])
            return False
        j0 = mbh.warm_span_covered_from(sp, r["covered"])
        if j0 >= sp.passes:
            self._skip(f"span window covered {r['covered']} of {[sp.sub(j)[1] for j in range(sp.passes)]}")
            return False
        if j0:
            self.stats["span_rebased"] += 1
        first = int(torch.argmax(logits.reshape(-1)[: int(self.model.args.vocab_size)]).item())
        prompt = [int(t) for t in prompt_ids]
        origin = sp.sub(j0)[0]
        anchor = []
        issued = r.get("issued") or [False] * sp.passes
        if j0 and any(issued):
            # QWEN36_MTP_WARM_TAIL_SPAN_EARLY: early passes wrote at relative j*N from w0, but the origin moved to
            # pass j0 (a lost chunk after them). Every pass from j0 is re-run below at its final index: the early
            # rows are overwritten where they overlap and sit past the warmed window otherwise, where decode writes
            # its own KV (relative n, n+1, ...) before anything reads it.
            self.stats["span_early_wasted"] += sum(issued)
            issued = [False] * sp.passes
        for j in range(j0, sp.passes):
            if issued[j]:
                continue  # QWEN36_MTP_WARM_TAIL_SPAN_EARLY: already queued under the chunk loop (origin w0, j0 == 0)
            anchor += self._span_issue(sp, j, j0, prompt, first, page_table_row)
        self._anchor = anchor
        self.mtp.host.warm_slot(int(slot), origin, sp.L, first)
        self.stats["warmed"] += 1
        if self.stats["warmed"] == 1 or self.stats["warmed"] % 256 == 0:
            logger.info(
                f"QWEN36_MTP_WARM_TAIL warmed: L={sp.L} n={sp.L - origin} origin={origin} slot={int(slot)} "
                f"first={first} span passes={sp.passes - j0}/{sp.passes} count={self.stats['warmed']} "
                f"skipped={self.stats['skipped']} span_rebased={self.stats['span_rebased']}"
                + (f" early={self.stats['span_early']} early_wasted={self.stats['span_early_wasted']}"
                   if self.early else "")
            )
        return True

    def _span_selftest(self, hidden, S):
        """QWEN36_MTP_WARM_TAIL_SPAN warmup self-test, fail closed, both chips: a synthetic prompt of L = k*N + N/2
        positions, reported as consecutive S-row segments that all read one source (this spec's own warmup output
        plus a random offset, so every row is distinct), so every pass straddles segment boundaries. Each window
        buffer read back over the whole mesh must equal the source rows of its positions bit for bit (pad rows zero).
        The wrong-input branch (gathers one row late) must DIFFER at every boot, else the test is blind and the boot
        is refused. Under QWEN36_MTP_WARM_TAIL_SPAN_MUSTFAIL=1 the served branch is the late one: REFUSED."""
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=3)
        S = int(S)
        g = torch.Generator().manual_seed(1234)
        rnd = ttnn.from_torch(torch.randn(1, 1, S, self.D * int(self.model.num_devices), generator=g),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                              mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3),
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        src = ttnn.add(hidden, rnd, dtype=hidden.dtype, memory_config=hidden.memory_config())
        ttnn.deallocate(rnd)
        ref_src = ttnn.to_torch(src, mesh_composer=comp).reshape(S, -1).to(torch.bfloat16)
        L = self.span * self.N + self.N // 2
        segs = [(st, min(S, L - st)) for st in range(0, L, S)]

        def build(shift):
            sp = mbh.warm_span_window(L, self.N, self.span)
            r = {"span": sp, "covered": [0] * sp.passes, "first": [True] * sp.passes, "bad": None}
            for st, valid in segs:
                t = self._span_targets(sp, st, valid, S, shift=shift)
                if t:
                    self._span_gather(src, S, r, t)
            ttnn.synchronize_device(self.mesh)
            bad = []
            for j in range(sp.passes):
                s0, rows = sp.sub(j)
                got = ttnn.to_torch(self.wins[j], mesh_composer=comp).reshape(self.Wn, -1).to(torch.bfloat16)
                exp = torch.zeros_like(got)
                pos = torch.arange(s0, s0 + rows)
                exp[:rows] = ref_src[pos % S]
                if r["covered"][j] != rows or not torch.equal(got, exp):
                    bad.append((j, int(r["covered"][j]), rows, int((got != exp).any(dim=-1).sum())))
            return sp, bad

        _, wrong = build(True)  # the wrong-input branch: must differ in every pass
        sp, bad = build(None)  # the served branch (late under the must-fail)
        ttnn.deallocate(src)
        self._anchor = []
        if len(wrong) != sp.passes:
            raise RuntimeError(
                f"QWEN36_MTP_WARM_TAIL_SPAN self-test REFUSED: the wrong-input branch matched in "
                f"{sp.passes - len(wrong)}/{sp.passes} passes (the test is blind)")
        if bad:
            raise RuntimeError(
                f"QWEN36_MTP_WARM_TAIL_SPAN self-test REFUSED: (pass, covered, rows, wrong rows) {bad} "
                f"(must-fail {'on' if self.span_mustfail else 'off'})")
        self.span_selftested = True
        logger.info(
            f"QWEN36_MTP_WARM_TAIL_SPAN self-test exact: L={L} S={S} {len(segs)} segments -> {sp.passes} passes x "
            f"{self.Wn} rows, both chips bitwise; wrong-input branch differs in {len(wrong)}/{sp.passes} passes"
        )

    # ---------------------------------------------------------------- QWEN36_MTP_WARM_TAIL_SPAN_EARLY
    def _scratch_block_readback(self):
        """The MTP KV scratch block (K, V) over the whole mesh, as torch: what a fill table of pad_block entries wrote
        last. The early self-test compares this between the early and the late path."""
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=1)
        out = []
        for c in self.mtp.kv:
            nb, nh, bs, hd = (int(x) for x in c.shape)
            blk = ttnn.slice(c, (self.pad_block, 0, 0, 0), (self.pad_block + 1, nh, bs, hd))
            out.append(ttnn.to_torch(blk, mesh_composer=comp).to(torch.float32).clone())
            ttnn.deallocate(blk)
        return out

    def _span_early_selftest(self):
        """QWEN36_MTP_WARM_TAIL_SPAN_EARLY warmup self-test, fail closed, both chips: for every early-eligible pass j of
        a synthetic k-pass prompt (random tokens, the window buffers as the span self-test left them, a page-table
        row of scratch blocks so only the MTP scratch block is written), the KV block the EARLY path writes
        (_span_issue with the early pairing) must equal, bit for bit over the whole mesh, the KV block the LATE path
        writes (the same builder with the served pairing); the wrong-input branch (tokens one position early) must
        DIFFER in every pass, else the test is blind and the boot is refused. Under
        QWEN36_MTP_WARM_TAIL_SPAN_EARLY_MUSTFAIL=1 the early path IS the wrong-input branch: REFUSED."""
        sp = mbh.warm_span_window(self.span * self.N + self.N // 2, self.N, self.span)
        g = torch.Generator().manual_seed(4321)
        vocab = int(self.model.args.vocab_size)
        prompt = [int(t) for t in torch.randint(1, min(vocab, 30000), (sp.L,), generator=g).tolist()]
        nblk = -(-sp.n // self.block_size) + 1
        pt_row = torch.full((1, nblk), int(self.pad_block), dtype=torch.int32)
        passes = [j for j in range(sp.passes) if mbh.warm_span_early_ready(sp, [sp.sub(i)[1] for i in range(sp.passes)], j)]
        bad, same_wrong = [], []
        for j in passes:
            self._span_issue(sp, j, 0, prompt, None, pt_row, shift=False)  # the late path's pairing
            ttnn.synchronize_device(self.mesh)
            late = self._scratch_block_readback()
            self._span_issue(sp, j, 0, prompt, None, pt_row, shift=True)  # the wrong-input branch
            ttnn.synchronize_device(self.mesh)
            wrong = self._scratch_block_readback()
            if all(torch.equal(a, b) for a, b in zip(late, wrong)):
                same_wrong.append(j)
            self._span_issue(sp, j, 0, prompt, None, pt_row, shift=self.early_mustfail)  # the served early path
            ttnn.synchronize_device(self.mesh)
            early = self._scratch_block_readback()
            diff = [int((a != b).sum()) for a, b in zip(late, early)]
            if any(diff):
                bad.append((j, diff))
        ttnn.synchronize_device(self.mesh)
        self._anchor = []
        if not passes:
            raise RuntimeError("QWEN36_MTP_WARM_TAIL_SPAN_EARLY self-test REFUSED: no early-eligible pass (k < 2)")
        if same_wrong:
            raise RuntimeError(
                f"QWEN36_MTP_WARM_TAIL_SPAN_EARLY self-test REFUSED: the wrong-input branch matched the late path in "
                f"passes {same_wrong} (the test is blind)")
        if bad:
            raise RuntimeError(
                f"QWEN36_MTP_WARM_TAIL_SPAN_EARLY self-test REFUSED: (pass, [K, V] differing elements) {bad} "
                f"(must-fail {'on' if self.early_mustfail else 'off'})")
        self.early_selftested = True
        logger.info(
            f"QWEN36_MTP_WARM_TAIL_SPAN_EARLY self-test exact: L={sp.L} passes {passes} of {sp.passes}: early KV block "
            f"== late KV block bitwise over the mesh (K and V); wrong-input branch differs in {len(passes)}/{len(passes)}"
        )


class Qwen36MTPBatch:
    """Owns the MTP head, its paged KV, the per-bucket traces and the host bookkeeping."""

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh_device
        args = model.args
        self._check_requirements()
        self.kcap = mbh.k_cap()
        # QWEN36_MTP_BATCH_K_BY_BUCKET: per-bucket K (each <= kcap and Bw*(K+1) <= 32; validated, fails closed).
        # Every bucket already has its own V/C(/R) traces, so a shallower K only shrinks that bucket's traces.
        self.kmap = mbh.k_by_bucket(self.kcap)
        self.Bmax = int(args.max_batch_size)
        self.widths = mbh.buckets(self.Bmax)
        if not self.widths:
            raise RuntimeError(f"QWEN36_MTP_BATCH: no speculating bucket for max_batch_size={self.Bmax}")
        # The paged-KV block: 64 on the shipped arms; allocate_kv() replaces it from the KV shape (2048 on the prefix
        # arm, where vLLM's hybrid prefix caching makes the attention block equal the GDN block). Every capacity and
        # fill-table computation below reads self.block_size / host.block_size, never a literal.
        self.block_size = 64
        self.host = mbh.MTPBatchHost(self.Bmax, self.kcap, self.block_size)
        self.method = _tt_public_getenv(ENV_COMMIT, "auto")
        if self.method not in ("auto", "where", "blend"):
            raise ValueError(f"{ENV_COMMIT}={self.method!r}: auto|where|blend")
        self.rec_mask_dtype = torch.bfloat16  # set by the commit self-test
        self.buckets = {}
        self.compiled = False
        self.captured = False
        self._stats_every = int(_tt_public_getenv(ENV_STATS_EVERY, "500"))
        self._steps = 0
        self._last_step_t = None  # wall clock of the last speculative step (stats / TIMING lines: t=)
        self._fallbacks = {}
        if getattr(model, "_resident_cos_tab", None) is None:
            model._build_resident_rope_tables()  # device RoPE from positions (same bf16 values as the host path)

        host_layer = next((i for i, t in enumerate(args.attention_type_list) if t == "full_attention"), None)
        assert host_layer is not None and args.is_full_attention_layer(host_layer)
        wdir = _tt_public_getenv(ENV_WEIGHTS) or str(args.CKPT_DIR)
        cache = _tt_public_getenv(ENV_CACHE)
        cache_root = Path(cache) if cache else None
        sd = _LazyMTPState(wdir, host_layer)
        saved = {k: os.environ.pop(k) for k in _BF4_ENVS if k in os.environ}
        try:
            from models.demos.blackhole.qwen36.public_defaults import without_defaults
            with without_defaults(_BF4_ENVS):
                self.head = _MTPHead(model, sd, host_layer, cache_root)
        finally:
            os.environ.update(saved)
            sd.close()
        self._bucketing = _tt_public_getenv("TT_DECODE_BUCKETING", "1") == "1"
        if not self._bucketing:
            # Fixed decode width: every step speculates at Bw = max_batch, padded rows included, so a
            # batch whose rows finish at different times never drops to a narrower (plain) bucket.
            logger.info(
                f"QWEN36_MTP_BATCH fixed width engaged: TT_DECODE_BUCKETING=0, every step at Bw={self.Bmax}"
            )
        if self.host.reject_all:
            logger.info(
                "QWEN36_MTP_BATCH_FORCE_REJECT engaged: drafts are verified at full width but none is accepted "
                "(output must equal plain decode unless the wider verify changes numerics)"
            )
        self._plain_only = mbh.plain_only()
        if self._plain_only:
            logger.info(
                "QWEN36_MTP_BATCH_PLAIN_ONLY engaged: traces built and captured, every step runs plain decode "
                "(control for the MTP arm's sync, non-resident serving mode)"
            )
        if self.host.mustfail:
            logger.info(
                f"QWEN36_MTP_BATCH_MUSTFAIL engaged: {self.host.mustfail} (GDN state committed at min(m+1, K) "
                "while the emitted tokens stay m+1: equivalence vs plain decode MUST fail)"
            )
        self._slot = mbh.state_slot()
        if self._slot and not all(hasattr(ttnn, a) for a in ("generic_op", "ProgramDescriptor", "TensorAccessorArgs")):
            raise RuntimeError("QWEN36_MTP_STATE_SLOT=1 refused: this ttnn has no generic_op / ProgramDescriptor")
        # QWEN36_MTP_CONV_MULTI (gdn/conv_multi.py, docs/MTP-BATCH.md section 14): the verify's GDN conv as one
        # batch-M conv_gates; its windows are committed by the slot programs only.
        from models.demos.blackhole.qwen36.tt.gdn import conv_multi as _cm

        self._conv_multi = _cm.enabled()
        if self._conv_multi and not self._slot:
            raise RuntimeError("QWEN36_MTP_CONV_MULTI=1 refused: it needs QWEN36_MTP_STATE_SLOT=1 (slot commit)")
        # QWEN36_MTP_CONV_FOLD (gdn/conv_fold.py): CONV_MULTI's window build folded into the conv_gates reader (one
        # program per layer instead of two); check_env refuses without CONV_MULTI / STATE_SLOT / QWEN_GDN_CONV_GATES.
        from models.demos.blackhole.qwen36.tt.gdn import conv_fold as _cf

        self._conv_fold = _cf.check_env()
        if self._conv_fold and not self._conv_multi:
            raise RuntimeError(f"{_cf.ENV}=1 refused: it needs QWEN36_MTP_CONV_MULTI engaged")
        if self._conv_fold and _cf.mustfail():
            logger.info(f"{_cf.MF_BANNER}: the last candidate's conv windows read one token too far (the warmup "
                        "self-test must REFUSE the boot)")
        # QWEN36_MTP_GDN_STATE_ONCE (gdn/state_once.py, docs/MTP-BATCH.md section 16): the verify reads each row's GDN
        # state once and writes none; the slot commit recomputes the accepted state. Refuses without STATE_SLOT and
        # CONV_MULTI (check_env raises).
        from models.demos.blackhole.qwen36.tt.gdn import state_once as _so

        self._state_once = _so.check_env()
        if self._state_once and not self._conv_multi:
            raise RuntimeError("QWEN36_MTP_GDN_STATE_ONCE=1 refused: it needs QWEN36_MTP_CONV_MULTI engaged")
        if self._state_once and _so.mustfail():
            logger.info(
                "QWEN36_MTP_GDN_STATE_ONCE_MUSTFAIL engaged: the commit gives the last row of each bucket one token too "
                "many (the warmup self-test must REFUSE the boot)"
            )
        # QWEN36_MTP_GDN_SO_T0 (docs/MTP-BATCH.md section 17): the state-once verify writes each in-bucket row's state
        # after token 0 (always committed: slot_index is one-hot for b < Bw) and the commit recomputes tokens 1..idx.
        # t0_enabled raises without QWEN36_MTP_GDN_STATE_ONCE=1.
        self._so_t0 = _so.t0_enabled()
        if self._so_t0:
            logger.info(
                f"{_so.T0_BANNER}: the verify writes rec_state rows < Bw after token 0, the slot commit recomputes "
                f"tokens 1..idx[b] from them (c156a: commit 6.12 ms/step at Bw=8 was the idx+1-token recompute)"
            )
            if _so.t0_mustfail():
                logger.info(
                    f"{_so.T0_MF_BANNER}: the verify writes the last row of each bucket after token 1 instead of 0 "
                    "(the warmup self-test must REFUSE the boot)"
                )
        # QWEN36_MTP_GDN_SO_SFPU_ROW0 (gdn/state_once.py, section 20): read per call by the state-once verify and both
        # commits; checked here so a bad value or a missing prerequisite refuses the boot. row0_enabled raises without
        # QWEN36_MTP_GDN_STATE_ONCE=1.
        if _so.row0_enabled():
            logger.info(
                f"{_so.ROW0_BANNER} (verify + commit; outputs bitwise today's, checked by the warmup self-test against "
                f"the sequential op; op-anatomy 20261006T045611Z: 4.5 of ~15.5 us per token were full-tile SFPU)"
            )
            if _so.row0_mustfail():
                logger.info(f"{_so.ROW0_MF_BANNER}: the decay skips exp (the warmup self-test must REFUSE the boot)")
        # QWEN36_MTP_GDN_SO_FUSE / QWEN36_MTP_GDN_SO_SPLITREAD (gdn/state_once.py, sections 21/22): read per call by the
        # state-once kernels; checked here so a bad value or a missing prerequisite refuses the boot.
        if getattr(_so, "fuse_parts", None) is not None and _so.fuse_parts():
            logger.info(f"{_so.FUSE_BANNER} (parts {','.join(_so.fuse_parts())}; verify + commit; outputs bitwise "
                        "today's, checked by the warmup self-test against the sequential op)")
            if _so.fuse_mustfail():
                logger.info(f"{_so.FUSE_MF_BANNER}: delta skips v - kn@h (the warmup self-test must REFUSE the boot)")
        if getattr(_so, "split_enabled", None) is not None and _so.split_enabled():
            logger.info(f"{_so.SPLIT_BANNER} (verify only; data movement only, outputs bitwise today's)")
            if _so.split_mustfail():
                logger.info(f"{_so.SPLIT_MF_BANNER}: the writer reads the neighbour head's state (the warmup "
                            "self-test must REFUSE the boot)")
        # QWEN36_MTP_GDN_SO_IOSRC (gdn/state_once.py, section 23): the recurrence's io operands (the recurrent state
        # and g) are read by the SFPU directly, so their fp32 mirror copies (a 16-tile copy per token for the state)
        # are not made; the unpacker converts io -> fp32 exactly, so the outputs are bitwise today's.
        if getattr(_so, "iosrc_enabled", None) is not None and _so.iosrc_enabled():
            logger.info(
                f"{_so.IOSRC_BANNER} (verify + commit; outputs bitwise today's, checked by the warmup self-test "
                f"against the sequential op)"
            )
            if _so.iosrc_mustfail():
                logger.info(f"{_so.IOSRC_MF_BANNER}: the decay rotates the state's tiles (the warmup self-test must "
                            "REFUSE the boot)")
        # QWEN36_MTP_GDN_SO_QKH (gdn/state_once.py): the verify's q/k L2 norms once per instance (every token's row at
        # once) instead of once per token; o and cin bitwise the per-token chain's, checked by the warmup self-test.
        if getattr(_so, "qkh_enabled", None) is not None and _so.qkh_enabled():
            logger.info(
                f"{_so.QKH_BANNER} (verify only; o and cin bitwise today's, checked by the warmup self-test against "
                f"the sequential op)"
            )
            if _so.qkh_mustfail():
                logger.info(f"{_so.QKH_MF_BANNER} (the warmup self-test must REFUSE the boot)")
        # QWEN36_MTP_GDN_SO_VSPLIT (gdn/state_once.py): read per verify call; checked here so a bad value or a missing
        # prerequisite refuses the boot. The per-Bw banner (P, cores) is logged by the verify's first call.
        if getattr(_so, "vsplit_cap", None) is not None and _so.vsplit_cap():
            if _so.vsplit_mustfail():
                logger.info(f"{_so.VSPLIT_MF_BANNER} (the warmup self-test must REFUSE the boot)")
        # QWEN36_MTP_COMMIT_PACK (gdn/commit_pack.py, section 19): read per call; checked here so a bad value or a missing
        # prerequisite refuses the boot instead of silently running the per-layer commit.
        from models.demos.blackhole.qwen36.tt.gdn import commit_pack as _cp

        if _cp.mode() and not self._state_once:
            raise RuntimeError(f"{_cp.ENV} refused: it needs QWEN36_MTP_GDN_STATE_ONCE engaged")
        if _cp.mustfail():
            logger.info(
                f"{_cp.MF_BANNER}: the packed commit hands the last GDN layer the previous layer's cin / windows "
                "(the warmup self-test must REFUSE the boot)"
            )
        self._timing = mbh.timing()
        self._timer = mbh.PhaseTimer() if self._timing else None
        if self._timing:
            logger.info(
                f"QWEN36_MTP_BATCH_TIMING engaged: host timers + ttnn.synchronize_device around verify, verify "
                f"readback, accept, commit (GDN {'slot-index gather' if self._slot else 'where-chain'}), reseed+draft (MTP head), draft readback; the C trace "
                f"is captured as two traces; one line per bucket every {mbh.TIMING_EVERY} steps and at each drain (diagnostic, the "
                f"syncs serialise host and device: not a speed number)"
            )
        if self.host.trace:
            logger.info(f"QWEN36_MTP_BATCH_TRACE engaged: per-step trace of rows matching {self.host.trace!r}")
        # QWEN36_MTP_OVERLAP (section 12): non-blocking commit/draft, deferred draft readback.
        self._overlap = mbh.overlap()
        if self._overlap and self._timing:
            raise RuntimeError(
                "QWEN36_MTP_OVERLAP=1 with QWEN36_MTP_BATCH_TIMING=1 refused: the sync phase split serialises the "
                "overlap it would measure (use QWEN36_MTP_OVERLAP_TIMING=1)"
            )
        self._pending = None  # _Pending of the step whose drafts are still on the device
        self._ov_timer = (mbh.PhaseTimer(phases=mbh.OVERLAP_PHASES + mbh.OVERLAP_SUB_PHASES
                                         + (mbh.FEED_OV_PHASES if mbh.devfeed() else ()), mode="overlap")
                          if mbh.overlap_timing() else None)
        self._ov_sub = {}  # OVERLAP_TIMING sub-phases of the drain since the last close (ms)
        self._ov_part = None  # the last step's host-clock phases, closed at the next steady step
        self._ov_wait_ms = 0.0
        self._alloc_cap = mbh.async_sched()
        self._early_reseed = mbh.early_reseed()
        if self._early_reseed:
            logger.info(
                "QWEN36_MTP_EARLY_RESEED engaged: MTP KV reseed captured as its own trace and enqueued right after "
                "verify (plan-only inputs), overlapping the verify-id readback + host accept; commit + draft follow"
            )
        # QWEN36_MTP_SO1_CSPLIT (Option A, section 16f): capture/enqueue GDN commit (cc) and reseed+drafts (cd)
        # as separate traces. cd is enqueued first so the draft event (wait_c) fires without the so1 commit
        # recompute on its critical path; commit runs after on the CQ and drain/next-V waits for it before the
        # next verify (host prep between steps can hide it). Same ops as the one C trace, disjoint buffers (GDN
        # state vs MTP head), intended bit-exact. Opt-in: unset keeps the one C trace (flag off = today's path).
        # Requires OVERLAP + STATE_ONCE; TIMING refuses it (mbh.so1_csplit). Under EARLY_RESEED the C2 trace (commit +
        # draft steps) is the one split, R stays where it is. The banner is logged
        # at trace capture (_csplit_logged is an arm-switch guard), so a served switch re-earns it.
        self._c_split = mbh.so1_csplit()
        if self._c_split and not self._state_once:
            raise RuntimeError(f"{mbh.ENV_SO1_CSPLIT}=1 refused: it needs QWEN36_MTP_GDN_STATE_ONCE engaged")
        self._commit_event = None  # deferred cc event when C-split drain left commit in flight
        self._csplit_logged = False
        # QWEN36_MTP_DEVACCEPT (tt/mtp_accept.py, docs/MTP-BATCH.md section 25): the accept as the first program of the
        # draft trace. It writes the slot page (needs the slot commit) and relies on the deferred draft readback.
        self._da = mbh.devaccept()
        self._da_mf = mbh.devaccept_mustfail()
        if self._da:
            _bad = mbh.devaccept_check(self._slot, self._overlap, self._timing, mbh.lookup_cfg()[2])
            if _bad:
                raise RuntimeError(f"{mbh.ENV_DEVACCEPT}=1 refused; it needs: " + "; ".join(_bad))
        self._da_stats = {"checked": 0, "mismatch": 0}
        self._da_logged = False
        # QWEN36_MTP_DEVFEED (mtp_accept.feed, docs/MTP-BATCH.md section 31): the sampled family's draft trace ends with
        # the next verify's draft-dependent inputs built on device; a fast step (mbh.MTPBatchHost.feed_predict) enqueues
        # its verify before the previous step's drafts are read back. Every other step is the S10 step.
        self._feed = mbh.devfeed()
        self._feed_mf = mbh.devfeed_mustfail()
        self._feed_check = mbh.devfeed_check_steps()
        # QWEN36_MTP_DEVFEED_PAD (docs/MTP-BATCH.md section 32): steps with padded rows may be fast steps too; their own
        # step self-test (the first QWEN36_MTP_DEVFEED_CHECK padded fast steps) compares the active rows only.
        self._feed_pad = mbh.devfeed_pad()
        self._feed_pad_mf = mbh.devfeed_pad_mustfail()
        if self._feed:
            _bad = []
            if not self._da:
                _bad.append(f"{mbh.ENV_DEVACCEPT}=1 (row 0 of the fed v_tok is the device accept's t0n)")
            if not self._c_split or not self._early_reseed:
                _bad.append(f"{mbh.ENV_SO1_CSPLIT}=1 and QWEN36_MTP_EARLY_RESEED=1 (the feed is the last program of cd)")
            if not spec_sample.serve():
                _bad.append(f"{spec_sample.SERVE_ENV}=1 (the fed q blocks are the served rule's)")
            if self._timing:
                _bad.append(f"{mbh.ENV_TIMING} off")
            if set(mbh.FEED_NAMES) & set(spec_sample.WSKIP_NAMES):
                _bad.append("no fed input among the WSKIP buffers (the device writes them behind the cache's back)")
            if _bad:
                raise RuntimeError(f"{mbh.ENV_DEVFEED}=1 refused; it needs: " + "; ".join(_bad))
            if self._feed_mf:
                logger.info(f"{mbh.ENV_DEVFEED_MUSTFAIL} engaged: the captured feed writes draft 0 of row 0 + 1; the step "
                            f"self-test MUST refuse")
            if self._feed_pad:
                logger.info(f"QWEN36_MTP_DEVFEED_PAD engaged: steps with padded rows (position -1) may be fast steps; the "
                            f"plan equality and the padded step self-test (first {self._feed_check} padded fast steps) "
                            f"compare the active rows"
                            + (" (MUSTFAIL: every row; the padded step self-test MUST refuse)" if self._feed_pad_mf else ""))
            if self._feed_pad_mf:
                logger.info(f"{mbh.ENV_DEVFEED_PAD_MUSTFAIL} engaged: the padded step self-test compares every row and MUST "
                            f"refuse")
        self._feed_stats = {"fast": 0, "slow": 0, "miss": {}, "checked": 0, "by_bw": {}, "pad_fast": 0, "pad_checked": 0}
        self._feed_tab = None
        self._feed_n_ids = 0
        self._feed_logged = False
        # QWEN36_MTP_SPEC_SAMPLE_DUAL (section 26): a greedy trace family per bucket beside the sampled one; maybe_step
        # picks the family per step (mbh.dual_family). Refused without the served rule (there is no sampled family to
        # pair) and the must-fail without the flag.
        self._dual = spec_sample.dual()
        self._dual_mf = spec_sample.dual_mustfail()
        if self._dual and not spec_sample.serve():
            raise RuntimeError(f"{spec_sample.DUAL_ENV}=1 refused: it needs {spec_sample.SERVE_ENV}=1 (the sampled "
                               f"family; without it every step is already the greedy path)")
        if self._dual_mf and not self._dual:
            raise RuntimeError(f"{spec_sample.DUAL_MUSTFAIL_ENV}=1 needs {spec_sample.DUAL_ENV}=1")
        if self._dual_mf:
            logger.info(
                f"{spec_sample.DUAL_MUSTFAIL_ENV} engaged: a step with exactly one sampled row runs the greedy trace "
                f"family (that row emits argmax tokens): the sampling-distribution check MUST fail"
            )
        # steps per family, family switches, steps the greedy family could not take (reason -> count)
        self._dual_stats = {"g": 0, "s": 0, "switch": 0, "last": None, "fallback": {}}
        self._dual_logged = False
        if self._da_mf:
            logger.info(
                f"{mbh.ENV_DEVACCEPT_MUSTFAIL} engaged: row 0's device m is off by one (m+1, or m-1 at m=K): its GDN "
                f"commit, anchor, t0n, draft positions and chain noise follow the wrong m while the host emits the right "
                f"tokens; the drain check logs instead of raising, so the served equivalence MUST fail"
            )
        if self._overlap:
            logger.info(
                "QWEN36_MTP_OVERLAP engaged: commit/draft trace enqueued non-blocking, draft ids read back through "
                "an event and consumed at the next step, prefill, release or remap (host work overlaps the device "
                "commit + draft)"
            )
        if self._ov_timer is not None:
            logger.info(
                f"QWEN36_MTP_OVERLAP_TIMING engaged: host wall clocks per phase ({', '.join(mbh.OVERLAP_PHASES)}), "
                f"no device syncs; one line per bucket every {mbh.TIMING_EVERY} steps and at each drain"
            )
        # QWEN36_MTP_OPTRACE (section 29): named device-time windows of the served step (diagnostic, host-side only)
        self._optrace = _OpTrace(self.mesh) if optrace_enabled() else None
        if self._alloc_cap:
            logger.info(
                "QWEN36_MTP_ASYNC engaged in MTP verify: KV capacity per row = pages vLLM allocated (the async "
                "scheduler sizes them from a count that lags by <= K)"
            )
        self._idle_logged_at = 0
        if self.kmap:
            logger.info(
                f"QWEN36_MTP_BATCH_K_BY_BUCKET engaged: {mbh.k_by_bucket_banner(self.kmap, self.kcap)}; "
                f"this boot: {[(w, mbh.k_for_bucket(w, self.kcap, self.kmap)) for w in self.widths]}"
            )
        # QWEN36_MTP_WARM_TAIL (section 13): built after the head (it runs the head's layers in PREFILL mode).
        _wn = mbh.warm_tail()
        _wmf = mbh.warm_mustfail()
        if _wmf and not _wn:
            raise RuntimeError(f"{mbh.ENV_WARM_MUSTFAIL}={_wmf} needs {mbh.ENV_WARM} set")
        _wspan, _wsmf = mbh.warm_span(), mbh.warm_span_mustfail()  # QWEN36_MTP_WARM_TAIL_SPAN
        if (_wspan or _wsmf) and not _wn:
            raise RuntimeError(f"{mbh.ENV_WARM_SPAN} / {mbh.ENV_WARM_SPAN_MUSTFAIL} need {mbh.ENV_WARM} set")
        if _wsmf and not _wspan:
            raise RuntimeError(f"{mbh.ENV_WARM_SPAN_MUSTFAIL}=1 needs {mbh.ENV_WARM_SPAN} >= 2")
        _wearly, _wemf = mbh.warm_span_early(), mbh.warm_span_early_mustfail()  # QWEN36_MTP_WARM_TAIL_SPAN_EARLY
        if (_wearly or _wemf) and not _wspan:
            raise RuntimeError(f"{mbh.ENV_WARM_SPAN_EARLY} / {mbh.ENV_WARM_SPAN_EARLY_MUSTFAIL} need {mbh.ENV_WARM_SPAN} >= 2")
        if _wemf and not _wearly:
            raise RuntimeError(f"{mbh.ENV_WARM_SPAN_EARLY_MUSTFAIL}=1 needs {mbh.ENV_WARM_SPAN_EARLY}=1")
        if not _wspan:
            self.warm = _MTPWarm(self, _wn, _wmf) if _wn else None
        else:
            self.warm = _MTPWarm(self, _wn, _wmf, span=_wspan, span_mustfail=_wsmf, early=_wearly,
                                 early_mustfail=_wemf)
        logger.info(
            f"QWEN36_MTP_BATCH engaged: batched MTP speculative decode, buckets {self.widths} "
            f"K={[mbh.k_for_bucket(w, self.kcap, self.kmap) for w in self.widths]} (cap {self.kcap}), greedy exact acceptance, "
            f"MTP head from {wdir} ({len(sd)} tensors, host layer {host_layer}, bf8 weights; bf4 envs exempt "
            f"{sorted(saved)}), cache {cache_root or 'none'}"
        )
        # QWEN36_MTP_LOOKUP (step 1): host-only draft source swap, read per plan() call; validated here so a bad
        # value refuses the boot instead of the first decode step.
        _lk = mbh.lookup_banner(mbh.lookup_cfg())
        if _lk:
            logger.info(_lk)

    # ------------------------------------------------------------------ requirements
    def _check_requirements(self):
        m = self.model
        bad = []
        if m.num_devices != 2 or tuple(int(d) for d in m.mesh_device.shape) != (1, 2):
            bad.append("a 1x2 mesh")
        if not getattr(m, "_oneshot_ar", False):
            bad.append("QWEN36_ONESHOT_AR=1 (replicated decode seam)")
        if not getattr(m, "_device_argmax", False):
            bad.append("QWEN36_DEVICE_ARGMAX=1")
        if getattr(m, "_device_resident", False):
            bad.append("QWEN36_DEVICE_RESIDENT off (the speculative step is host-mediated)")
        # QWEN36_PREFIX_CACHING (the S12 prefix arm, docs/SERVING.md section 5c): admitted. The prefix cache keeps GDN
        # checkpoints at PREFILL chunk ends only (the plugin's qwen36_prefix_cache.py), the scheduler caches GDN
        # blocks over the prompt only (platform.py, MambaManager prompt-only patch), so multi-token commits never
        # create a block the cache would have to resume from. The MTP head's own KV is addressed through a per-slot
        # shifted page table (pt_shift: the request's prefix-hit blocks are skipped), so siblings sharing a cached
        # prefix never write each other's drafter KV. _check_prefix() below holds the warm-tail requirements.
        if _tt_public_getenv("QWEN_GDN_CONV_GATES") != "1" or _tt_public_getenv("QWEN_GDN_PACKED_QKV") != "1":
            bad.append("QWEN_GDN_CONV_GATES=1 and QWEN_GDN_PACKED_QKV=1")
        tr = getattr(ttnn, "transformer", None)
        if not (hasattr(tr, "gdn_decode_conv_gates") and hasattr(tr, "decode_gated_delta_rule_packed")):
            bad.append("an image with gdn_decode_conv_gates + decode_gated_delta_rule_packed")
        if not all(getattr(type(layer), "MTP_BATCH_VERIFY", 0) for layer in m.layers):
            bad.append("the decode-levers layer.py overlay (MTP_BATCH_VERIFY marker)")
        if getattr(m.rope, "rope_delta", 0) != 0:
            bad.append("text-only RoPE (rope_delta 0)")
        if bad:
            raise RuntimeError("QWEN36_MTP_BATCH=1 refused; it needs: " + "; ".join(bad))

    # ------------------------------------------------------------------ KV
    def allocate_kv(self, kv_cache_shape):
        # The MTP head's own paged KV follows the same resolver as the main model's
        # allocate_kv_caches, so an asymmetric QWEN36_KV_DTYPE arm (the k8v4 base: K bf8, V bf4)
        # keeps one KV policy across the whole process. The head is decode-only: its KV is written
        # by paged_update_cache (grouped reseed + draft), which casts the bf16 K/V rows to the
        # cache dtype, so no paged_fill_cache / typecast path exists for it. Q stays bf16.
        # No pad block: the +1 block of QWEN36_PREFILL_BUCKET_TRACE serves the main caches' prefill
        # fill only; this cache is indexed by the scheduler's page tables (0..num_blocks-1).
        k_dtype, v_dtype, _src = tpc.resolve_kv_dtypes(ttnn.bfloat16)
        main = (getattr(self.model, "_kv_k_dtype", None), getattr(self.model, "_kv_v_dtype", None))
        if main != (None, None) and main != (k_dtype, v_dtype):
            raise RuntimeError(
                f"QWEN36_MTP_BATCH: MTP KV K={k_dtype} V={v_dtype} would differ from the main caches "
                f"K={main[0]} V={main[1]} ({_src}); refusing a split KV policy"
            )

        def mk_shape(dt, shp):
            return ttnn.as_tensor(
                torch.zeros(shp, dtype=torch.bfloat16),
                device=self.mesh,
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        shape = list(kv_cache_shape)
        warm = getattr(self, "warm", None)
        # The real paged-KV block (kv_cache_shape = [num_blocks, kv_heads, block_size, head_dim]).
        bs = int(shape[2])
        if bs != self.block_size:
            logger.info(f"QWEN36_MTP_BATCH block size {bs} (from the KV shape; was {self.block_size})")
        self.block_size = bs
        self.host.block_size = bs
        if warm is not None:
            warm.block_size = bs
            if warm.span:
                mbh.warm_span_check(warm.N, warm.span, bs)
        if warm is not None:
            # QWEN36_MTP_WARM_TAIL: one scratch block past the scheduler's (never in a page table) takes the warm pass's
            # pad rows, so the fixed-width fill never writes outside the request's own blocks.
            shape[0] = int(shape[0]) + 1
        from models.demos.blackhole.qwen36.tt import kv_devinit as kvd

        if kvd.enabled():
            # QWEN36_KV_DEVICE_INIT: same allocations, same order, zero-filled on device (kv_devinit.py), self-tested.
            kvd.log_engaged(k_dtype, v_dtype, shape, 2, "MTP head")
            self.kv = tuple(kvd.alloc_filled(shape, dt, self.mesh, ttnn.DRAM_MEMORY_CONFIG) for dt in (k_dtype, v_dtype))
            summary = kvd.selftest([("mtp.K", self.kv[0]), ("mtp.V", self.kv[1])])
            logger.info(f"QWEN36_KV_DEVICE_INIT self-test exact (MTP head): {summary}")
        else:
            self.kv = (mk_shape(k_dtype, shape), mk_shape(v_dtype, shape))
        self.head.attention.set_paged_kv_cache(*self.kv)
        self.head.attention.B = self.Bmax
        if warm is not None:
            warm.allocate(pad_block=shape[0] - 1)
        logger.info(f"QWEN36_MTP_BATCH: MTP paged KV {shape} K={k_dtype} V={v_dtype} (own cache, same page tables)")

    # ------------------------------------------------------------------ device passes
    def _gdn_layers(self):
        return [layer.attention for layer in self.model.layers if not layer.is_full_attention]

    def _rope(self, pos_tt):
        return self.model._rope_from_positions_device(pos_tt)

    def _verify_ops(self, bk, d):
        """V: the base model over M = Bw*T rows. Returns (ids_rm [M,1] u32, hid TILE, hid_rm RM)."""
        from models.tt_transformers.tt.ccl import tt_all_gather

        model = self.model
        with _sc("embed"):
            x = model.embd(d["v_tok"])
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
            x = tt_all_gather(x, self.mesh, model.tt_ccl, cluster_axis=None, dim=3, topology=model.args.ccl_topology())
            cos, sin = self._rope(d["v_rpos"])
        group_io = FoldGroupIO((d[f"v_pos{j}"], d["v_pt"]) for j in range(bk.T))
        group_io.fold_pos = d.get("v_posf")
        abl = verify_class_ablate()
        key = (tuple(sorted(abl)), bk.bw, bk.T)
        if abl and key not in _CLASS_ABLATE_LOGGED:
            _CLASS_ABLATE_LOGGED.add(key)
            logger.warning(
                f"QWEN36_MTP_VERIFY_CLASS_ABLATE engaged: {sorted(abl)} in the verify Bw={bk.bw} T={bk.T} "
                f"(timing diagnostic: outputs are WRONG, never a speed or correctness number)"
            )
        with _ClassAblate(model.layers, abl):
            for layer in model.layers:
                if layer.is_full_attention:
                    x = layer.forward(x, cos, sin, mode="decode", oneshot_ar=True, group_io=group_io)
                else:
                    x = layer.forward(x, mode="decode", oneshot_ar=True, verify_T=bk.T)
        with _sc("head"):
            return self._verify_head(bk, d, model, x, abl)

    def _verify_head(self, bk, d, model, x, abl):
        """V's tail: final norm, LM head, the verify sample (QWEN36_MTP_OPTRACE scopes it as "head")."""
        hid = model._final_norm_decode(x, replicated=True)  # [1,1,M,dim] replicated, DRAM
        if "lmhead" in abl:
            ids_rm = ttnn.clone(d["v_tok"])  # QWEN36_MTP_VERIFY_CLASS_ABLATE=lmhead: WRONG ids, timing only
            spec_out = None
        else:
            logits = model._lm_head_matmul(hid, decode=True)
            ids, spec_out = self._verify_sample(bk, d, logits)
            ids_rm = ttnn.reshape(ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT), (bk.M, 1))
            if ids.layout != ttnn.ROW_MAJOR_LAYOUT:
                # QWEN36_MTP_SPEC_SAMPLE_FUSED hands back ROW_MAJOR ids: to_layout + reshape are then VIEWS of the same
                # buffer, and freeing ids would free ids_rm (the reseed's and the host's tokens)
                ttnn.deallocate(ids)
        hid_rm = ttnn.reshape(ttnn.to_layout(hid, ttnn.ROW_MAJOR_LAYOUT), (bk.M, hid.shape[-1]))
        if _serve():
            if spec_out is None:
                raise RuntimeError("QWEN36_MTP_SPEC_SAMPLE_SERVE=1 with QWEN36_MTP_VERIFY_CLASS_ABLATE=lmhead: no rule outputs")
            # the rule's outputs, read by _spec_data: the accept flags and the residual ids
            return (ids_rm, hid, hid_rm, ttnn.reshape(spec_out[0], (bk.M, 1)), ttnn.reshape(spec_out[1], (bk.M, 1)))
        return ids_rm, hid, hid_rm

    def _verify_sample(self, bk, d, logits):
        """The verify's ids from its logits (freed): (ids [1,1,M,1] uint32, spec_out). spec_out is None unless
        QWEN36_MTP_SPEC_SAMPLE serves, then (accept [1,1,M,1] fp32, resid [1,1,M,1] uint32).

        Greedy rows always take _greedy_argmax_device (byte-identical to the flag off). QWEN36_DEVICE_SAMPLE:
        a T > 0 row takes the device sample (dev_sample.sample; under the served rule, the same Gumbel-max
        pick over spec_sample's exact cut of the same block). The block is keyed by ROLE: the trace inputs
        carry the v_ names (VERIFY_DS_BLOCK maps one to the other; keying it by the v_ names refused every
        card boot in s-sample4-validate 2026-10-21)."""
        model = self.model
        topo = model.args.ccl_topology()
        per = int(logits.shape[3])
        if not _ds():
            return model._greedy_argmax_device(logits), None
        K = dev_sample.k()
        # (QWEN36_MTP_SPEC_SAMPLE_SSP: no v_gum; fs_cut makes the pick noise)
        blk = {role: ttnn.reshape(d[src], (1, 1, bk.M, w)) for role, src, w in VERIFY_DS_BLOCK(K) if src in d}
        if not _serve():
            samp = dev_sample.sample(logits, per, self.mesh, model.tt_ccl, topo, blk, K=K, deallocate=False)
            ids = model._greedy_argmax_device(logits)
            return dev_sample.select_ids(ids, samp, blk["flag"]), None
        # QWEN36_MTP_SPEC_SAMPLE: the target's block p (the same cut as the drafter's) and its sample, then
        # the exact rule against the carried q.
        M, C2 = bk.M, 2 * K
        x = ttnn.reshape(d["v_sx"], (1, 1, M, 1))
        if _ssp():
            # QWEN36_MTP_SPEC_SAMPLE_SSP: fs_cut also makes the noise (v_snp) and runs the exact rule against the
            # carried q (spec_sample.rule's ~46 programs): 3 programs for the whole verify sample
            c = fused_sampler.serve_candidates(
                logits, per, self.mesh, model.tt_ccl, topo, blk, K, override=(x, ttnn.reshape(d["v_sxf"], (1, 1, M, 1))),
                rule={"qp": ttnn.reshape(d["v_sqp"], (1, 1, M, C2)), "qi": ttnn.reshape(d["v_sqi"], (1, 1, M, C2))},
                noise={"np": d["v_snp"], "mode": 1})
            ttnn.deallocate(logits)
            ttnn.deallocate(c["probs"])
            ttnn.deallocate(c["ids"])
            return c["sel"], (c["accept"], c["resid"])
        rule_in = (ttnn.reshape(d["v_sqp"], (1, 1, M, C2)), ttnn.reshape(d["v_sqi"], (1, 1, M, C2)), x,
                   ttnn.reshape(d["v_su"], (1, 1, M, 1)), ttnn.reshape(d["v_snr"], (1, 1, M, C2)))
        if fused_sampler.enabled():
            # QWEN36_MTP_SPEC_SAMPLE_FUSED: the target's block, its cut, the pick, the greedy id, the flag select and
            # the draft override (v_sxf ? v_sx : ...) in 3 programs straight from the TILE logits (fused_sampler.py)
            c = fused_sampler.serve_candidates(logits, per, self.mesh, model.tt_ccl, topo, blk, K,
                                               override=(x, ttnn.reshape(d["v_sxf"], (1, 1, M, 1))))
            ttnn.deallocate(logits)
            rr = spec_sample.rule(c["probs"], c["ids"], *rule_in)
            ttnn.deallocate(c["probs"])
            ttnn.deallocate(c["ids"])
            return c["sel"], (rr["accept"], rr["resid"])
        if spec_sample.rm_share():
            # QWEN36_MTP_SPEC_SAMPLE_RMSHARE: one untilize feeds the wide top-K and the greedy argmax (which
            # untilized the same logits again); the same row-major values, so the same ids.
            rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(logits)
            c = spec_sample.candidates(rm, per, self.mesh, model.tt_ccl, topo, blk, K=K)
            greedy = argmax_fold.greedy_rm(rm, per, self.mesh, model.tt_ccl, topo)
            ttnn.deallocate(rm)
        else:
            c = spec_sample.candidates(logits, per, self.mesh, model.tt_ccl, topo, blk, K=K)
            greedy = model._greedy_argmax_device(logits)  # frees the logits
        ids = dev_sample.select_ids(greedy, c["pick"], blk["flag"])
        # an exact row with a draft at depth j returns the DRAFT as its id (what the MTP reseed writes at
        # that position, and what is emitted if accepted); the host reads its tokens from the rule outputs.
        xu = ttnn.typecast(x, ttnn.uint32)
        ids2 = dev_sample.select_ids(ids, xu, ttnn.reshape(d["v_sxf"], (1, 1, M, 1)))
        rr = spec_sample.rule(c["probs"], c["ids"], *rule_in)
        for t in (greedy, ids, xu, c["pick"], c["probs"], c["ids"]):
            ttnn.deallocate(t)
        return ids2, (rr["accept"], rr["resid"])

    def _commit_draft_ops(self, bk, d, v_out, release=False):
        """C: GDN per-row commit, MTP KV reseed over the verify rows, K chained draft steps (QWEN36_MTP_DEVACCEPT: the
        device accept first)."""
        self._accept_ops(bk, d, v_out)
        self._commit_ops(bk, d, release=release)
        return self._draft_ops(bk, d, v_out)

    def _accept_ops(self, bk, d, v_out):
        """QWEN36_MTP_DEVACCEPT: the accept on device, the first program of the draft trace. Writes gslot, anchor, t0n,
        d_pos*, d_rpos*, (served exact rule) d_sgum* and da_m from the verify outputs and the pre-verify inputs. A no-op
        without the flag."""
        if not getattr(bk, "da", False):
            return
        with _sc("accept"):
            self._accept_run(bk, d, v_out)

    def _accept_run(self, bk, d, v_out):
        from models.demos.blackhole.qwen36.tt import mtp_accept

        K = bk.K
        kw = {}
        if _ssp():
            kw = {"accept": v_out[3], "resid": v_out[4]}  # QWEN36_MTP_SPEC_SAMPLE_SSP: no chain-noise copy
        elif _serve():
            kw = {"accept": v_out[3], "resid": v_out[4], "tabs": [d[f"d_sgtab{k}"] for k in range(K)],
                  "nz": [d[f"d_sgum{k}"] for k in range(K)]}
        mtp_accept.run(v_out[0], d["v_tok"], d["da_plan"], d["gslot"], d["anchor"], d["t0n"],
                       [d[f"d_pos{k}"] for k in range(K)], [d[f"d_rpos{k}"] for k in range(K)], bk.da_m, bk.bw, bk.T,
                       **kw)

    def _commit_ops(self, bk, d, release=False):
        """C, first half: the GDN per-row recurrent + conv state commit (the where-chain, or the
        QWEN36_MTP_STATE_SLOT slot-index gather)."""
        if bk.slot:
            # QWEN36_MTP_COMMIT_PACK (gdn/commit_pack.py, docs/MTP-BATCH.md section 19): read per call (runtime class),
            # the commit of every GDN layer in one program per state kind; unset: the per-layer loop below.
            from models.demos.blackhole.qwen36.tt.gdn import commit_pack as _cp

            kinds = _cp.mode() if self._state_once else frozenset()
            if kinds:
                _cp.commit_all(self._gdn_layers(), bk.bw, d["gslot"], kinds, release=release, logger=logger)
                return
            for dn in self._gdn_layers():
                dn.mtp_commit_slot(bk.bw, d["gslot"], release=release)
            return
        gm = [d[f"gmask{j}"] for j in range(bk.T)]
        cm = [d[f"cmask{j}"] for j in range(bk.T)]
        for dn in self._gdn_layers():
            dn.mtp_commit(bk.bw, gm, cm, self.method, release=release)

    def _draft_ops(self, bk, d, v_out):
        """C, second half: the MTP KV reseed over the verify rows, then K chained draft steps."""
        self._reseed_ops(bk, d, v_out)
        return self._draft_steps(bk, d, v_out)

    def _reseed_ops(self, bk, d, v_out):
        """The MTP KV reseed over the T verify rows (KV-only). Reads the plan-only rs_* inputs and V's outputs; the
        accepted length never enters, so QWEN36_MTP_EARLY_RESEED runs it before the host accept."""
        ids_rm, hid, hid_rm = v_out[0], v_out[1], v_out[2]
        with _sc("reseed"):
            cos, sin = self._rope(d["rs_rpos"])
            self.head.forward(
                hid, ids_rm, cos, sin, group_io=[(d[f"rs_pos{j}"], d.get("v_mpt", d["v_pt"])) for j in range(bk.T)],
                kv_only=True
            )

    def _draft_steps(self, bk, d, v_out):
        """K chained MTP draft steps from the anchor row m_b*Bw + b. Returns the K [Bw,1] draft-id
        tensors, plus (QWEN36_MTP_SPEC_SAMPLE) the K candidate PROBS blocks and the K candidate ID
        blocks: the drafter's candidate set, carried to the next verify (the rule needs q at the
        target's ids).  The order is [ids..., probs..., qids...]."""
        ids_rm, hid, hid_rm = v_out[0], v_out[1], v_out[2]
        h = ttnn.embedding(d["anchor"], hid_rm, layout=ttnn.TILE_LAYOUT)  # [1,Bw,dim]
        h = ttnn.reshape(h, (1, 1, bk.bw, h.shape[-1]))
        tok = d["t0n"]
        drafts = []
        probs, qids = [], []
        for k in range(bk.K):
            with _sc(f"d{k}"):
                h = self._draft_depth(bk, d, k, h, tok, drafts, probs, qids)
            tok = drafts[-1]
        ttnn.deallocate(h)
        return drafts + probs + qids

    def _draft_depth(self, bk, d, k, h, tok, drafts, probs, qids):
        """Draft depth k of _draft_steps (QWEN36_MTP_OPTRACE scopes it as d<k>): appends the depth's ids (and the served
        rule's candidate blocks) and returns the next hidden state."""
        cos, sin = self._rope(d[f"d_rpos{k}"])
        if _serve():
            # QWEN36_MTP_SPEC_SAMPLE: depth k's noise and the request's own cut params (T, top_k, top_p)
            self.head._spec_blk = {"rank": d["d_srank"], "scale": d["d_sscale"], "topp": d["d_stopp"],
                                   "flag": d["d_sflag"]}
            if _ssp():
                # QWEN36_MTP_SPEC_SAMPLE_SSP: fs_cut makes depth k's noise from the plan page and the device
                # accept's m page (the first program of this trace wrote it)
                self.head._spec_blk["_ssp"] = {"np": d["v_snp"], "mode": 2, "depth": k, "m": bk.da_m}
            else:
                self.head._spec_blk["noise"] = d[f"d_sgum{k}"]
        ids, h_next, extra = self.head.forward(h, tok, cos, sin, position_tensor=d[f"d_pos{k}"],
                                               page_table=d.get("v_mpt", d["v_pt"]))
        ttnn.deallocate(h)
        tok = ttnn.reshape(ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT), (bk.bw, 1))
        if ids.layout != ttnn.ROW_MAJOR_LAYOUT:  # a ROW_MAJOR ids (FUSED, no draft vocab) IS tok's buffer
            ttnn.deallocate(ids)
        drafts.append(tok)
        if extra is not None:
            probs.append(ttnn.reshape(ttnn.to_layout(extra["probs"], ttnn.ROW_MAJOR_LAYOUT), (bk.bw, -1)))
            qids.append(ttnn.reshape(ttnn.to_layout(extra["ids"], ttnn.ROW_MAJOR_LAYOUT), (bk.bw, -1)))
        return h_next

    # ------------------------------------------------------------------ warmup
    def _alloc_inputs(self, bk):
        from models.tt_transformers.tt.common import copy_host_to_device

        vals = bk.host_inputs()
        names = list(vals)
        dev = copy_host_to_device([_to_host_tt(n, vals[n]) for n in names], mesh_device=self.mesh)
        bk.names = names
        bk.dev = dict(zip(names, dev))
        if bk.da:
            # QWEN36_MTP_DEVACCEPT: the m page, an output only (never host-written per step), same placement as the inputs
            (bk.da_m,) = copy_host_to_device([_to_host_tt("da_m", torch.zeros(1, mbh.DA_M_LEN, dtype=torch.int32))],
                                             mesh_device=self.mesh)
        # QWEN36_MTP_SPEC_SAMPLE_WSKIP: what each skippable buffer holds now
        bk.wcache = {n: vals[n].clone() for n in names if n in spec_sample.WSKIP_NAMES}

    def _write_inputs(self, bk, vals):
        from models.tt_transformers.tt.common import copy_host_to_device

        names = [n for n in bk.names if n in vals]
        if spec_sample.wskip():
            # QWEN36_MTP_SPEC_SAMPLE_WSKIP: a parameter block equal (dtype, shape, bits) to the last one written is
            # already in its device buffer (nothing else writes these buffers; the traces only read them)
            cache = bk.wcache
            keep = []
            for n in names:
                if n in spec_sample.WSKIP_NAMES:
                    old = cache.get(n)
                    v = vals[n]
                    if old is not None and old.dtype == v.dtype and old.shape == v.shape and torch.equal(old, v):
                        continue
                    cache[n] = v.clone()
                keep.append(n)
            names = keep
            if not names:
                return
        copy_host_to_device([_to_host_tt(n, vals[n]) for n in names], device_tensors=[bk.dev[n] for n in names])

    def _selftest_commit(self, bk):
        """The commit select must be exact on THIS ttnn AT THE REAL DTYPES: the recurrent state is
        fp32 by default (QWEN35_GDN_STATE_BF16 unset) and the conv taps bf16. Random candidates of
        those dtypes, random one-hot masks, bitwise compare including the output dtype: a select
        that returns bf16 for fp32 candidates rounds the committed state every step, which plain
        decode never does. auto tries where then blend, each with a mask of the candidate dtype
        first and bf16 second; a combination that raises or differs is refused, and no working
        combination refuses the lever. (The first version tested bf16 candidates only, so an fp32
        truncation passed it.)"""
        from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

        dn = self._gdn_layers()[0]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        g = torch.Generator().manual_seed(1234 + bk.bw)
        to_torch_dt = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
        rec_dt = getattr(getattr(dn, "rec_state", None), "dtype", ttnn.float32)
        conv_dt = getattr((getattr(dn, "conv_states", None) or [None])[0], "dtype", ttnn.bfloat16)
        rec_t, conv_t = to_torch_dt.get(rec_dt, torch.float32), to_torch_dt.get(conv_dt, torch.bfloat16)
        methods = ["where", "blend"] if self.method == "auto" else [self.method]
        mask_dts = [rec_t] + ([torch.bfloat16] if rec_t != torch.bfloat16 else [])
        if getattr(self, "_select_fixed", False):  # a later bucket re-checks the pair the first one chose
            methods, mask_dts = [self.method], [self.rec_mask_dtype]
        # (shape, row dim, candidate dtype, mask dtype) for the rec state and one conv tap
        results = {}
        for method in methods:
            for mdt in mask_dts:
                key = f"{method}/mask={str(mdt).replace('torch.', '')}"
                ok = True
                try:
                    for shape, row_dim, cdt, kdt in (
                        ((bk.bw, dn.Nv, dn.Dk, dn.Dv), 0, rec_t, mdt),
                        ((1, self.Bmax, dn.qkv_dim_tp), 1, conv_t, torch.bfloat16),
                    ):
                        rows = shape[row_dim]
                        cands_t = [torch.randn(*shape, generator=g).to(cdt) for _ in range(bk.T)]
                        pick = torch.randint(0, bk.T, (rows,), generator=g)
                        mshape = [1] * len(shape)
                        mshape[row_dim] = rows
                        masks_t = [(pick == j).to(kdt).reshape(mshape) for j in range(bk.T)]
                        idx = [slice(None)] * len(shape)
                        exp = torch.empty(*shape, dtype=cdt)
                        for r in range(rows):
                            idx[row_dim] = r
                            exp[tuple(idx)] = cands_t[int(pick[r])][tuple(idx)]

                        def up(t):
                            dt = ttnn.float32 if t.dtype == torch.float32 else ttnn.bfloat16
                            return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                                   mesh_mapper=rep)

                        cands = [up(t) for t in cands_t]
                        masks = [up(t) for t in masks_t]
                        out = TPGatedDeltaNet.mtp_select(cands, masks, method)
                        out_dt = out.dtype
                        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(shape)
                        for t in cands + masks + [out]:
                            ttnn.deallocate(t)
                        if got.dtype != cdt or not torch.equal(got, exp):
                            ok = False
                            err = float((got.float() - exp.float()).abs().max())
                            results[key] = f"mismatch cand={cdt} out={out_dt} max_abs={err:.4g}"
                            break
                except Exception as e:  # an unsupported broadcast / dtype mix raises at compile time
                    ok = False
                    results[key] = f"raised {type(e).__name__}: {str(e)[:160]}"
                if ok:
                    results[key] = "exact"
                    self.method = method
                    self.rec_mask_dtype = mdt
                    self._select_fixed = True
                    bk.rec_mask_dtype = mdt
                    logger.info(
                        f"QWEN36_MTP_BATCH commit select: {method} rec-mask {mdt} (self-test exact at rec state "
                        f"{rec_dt} / conv {conv_dt}; tried {results})"
                    )
                    return
        raise RuntimeError(f"QWEN36_MTP_BATCH: no exact commit select on this ttnn ({results})")

    def _selftest_slot(self, bk):
        """QWEN36_MTP_STATE_SLOT warmup self-test, fail closed: the two slot-index programs on THIS
        image at the real state shapes and dtypes, against a torch gather, bitwise. Random candidates
        and a random persistent state; idx covers depth 0, depth K (T-1), and a SKIP row when the bucket
        is narrower than the state; rows >= Bw must come back untouched. Two rounds on the same
        buffers with different idx (slot reuse through one trace-stable page). Then a must-fail: the
        same program with every entry shifted by one depth must NOT match the reference, which proves
        the kernels read idx rather than a fixed slot."""
        from models.demos.blackhole.qwen36.tt.gdn import state_slot

        dn = self._gdn_layers()[0]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        g = torch.Generator().manual_seed(4321 + bk.bw)
        to_t = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
        rec_t = to_t.get(dn.rec_state.dtype, torch.float32)
        conv_t = to_t.get(dn.conv_states[0].dtype, torch.bfloat16)
        T, bw, Bmax = bk.T, bk.bw, self.Bmax
        rshape, cshape = (bw, dn.Nv, dn.Dk, dn.Dv), (1, Bmax, dn.qkv_dim_tp)

        def up(t, layout=ttnn.TILE_LAYOUT, dt=None):
            dt = dt or (ttnn.float32 if t.dtype == torch.float32 else ttnn.bfloat16)
            return ttnn.from_torch(t, dtype=dt, layout=layout, device=self.mesh, mesh_mapper=rep,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def back(t, shape):
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(shape)

        hist_t = [torch.randn(*rshape, generator=g).to(rec_t) for _ in range(T)]
        snap_t = [[torch.randn(*cshape, generator=g).to(conv_t) for _ in range(dn.K)] for _ in range(T - 1)]
        rec0 = torch.randn(Bmax, dn.Nv, dn.Dk, dn.Dv, generator=g).to(rec_t)
        live0 = [torch.randn(*cshape, generator=g).to(conv_t) for _ in range(dn.K)]
        hists = [up(t) for t in hist_t]
        snaps = [[up(t) for t in sn] for sn in snap_t]
        dst = up(rec0.clone())
        lives = [up(t.clone()) for t in live0]
        idx_dev = up(torch.zeros(1, mbh.SLOT_LEN, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        from models.tt_transformers.tt.common import copy_host_to_device

        exp_rec, exp_live = rec0.clone(), [t.clone() for t in live0]
        rounds = []
        for rnd in range(2):
            pick = torch.randint(0, T, (bw,), generator=g).tolist()
            pick[0], pick[-1] = 0, T - 1
            if rnd == 1 and bw > 2:
                pick[1] = mbh.SLOT_SKIP
            rounds.append(pick)
        results = {}
        try:
            for rnd, pick in enumerate(rounds + [None]):
                mustfail = pick is None
                if mustfail:
                    pick = [(p + 1) % T if p < T else p for p in rounds[-1]]
                page = [mbh.SLOT_SKIP] * mbh.SLOT_LEN
                page[:bw] = pick
                copy_host_to_device([_to_host_tt("gslot", torch.tensor(page, dtype=torch.int32).reshape(1, -1))],
                                    device_tensors=[idx_dev])
                if mustfail:
                    # reset, then run the shifted page against the unshifted round-1 reference
                    for src, d in [(rec0, dst)] + [(live0[i], lives[i]) for i in range(dn.K)]:
                        tmp = up(src.clone())
                        ttnn.copy(tmp, d)
                        ttnn.deallocate(tmp)
                    exp_rec, exp_live = rec0.clone(), [t.clone() for t in live0]
                    pick = rounds[-1]
                state_slot.commit_rec(dst, hists, idx_dev, bw, dn.Nv, dn.Dk, dn.Dv)
                state_slot.commit_conv(lives, snaps, idx_dev, bw)
                for b in range(bw):
                    if pick[b] < T:
                        exp_rec[b] = hist_t[pick[b]][b]
                        for i in range(dn.K):
                            src = snap_t[pick[b]][i] if pick[b] < T - 1 else exp_live[i]
                            exp_live[i][0, b] = src[0, b]
                got_rec = back(dst, rec0.shape)
                ok = got_rec.dtype == rec_t and torch.equal(got_rec, exp_rec)
                for i in range(dn.K):
                    got = back(lives[i], cshape)
                    ok = ok and got.dtype == conv_t and torch.equal(got, exp_live[i])
                key = "mustfail_shift1" if mustfail else f"round{rnd}"
                results[key] = "exact" if ok else "differs"
        finally:
            for t in hists + [x for sn in snaps for x in sn] + [dst, idx_dev] + lives:
                ttnn.deallocate(t)
        good = results.get("round0") == "exact" and results.get("round1") == "exact"
        blind = results.get("mustfail_shift1") != "differs"
        if not good or blind:
            raise RuntimeError(f"QWEN36_MTP_STATE_SLOT self-test failed at Bw={bw} T={T}: {results}")
        self.method = "slot"
        logger.info(
            f"QWEN36_MTP_BATCH commit select: slot-index gather (QWEN36_MTP_STATE_SLOT self-test exact at rec state "
            f"{dn.rec_state.dtype} / conv {dn.conv_states[0].dtype}, Bw={bw} T={T}: {results})"
        )

    def _selftest_conv_multi(self, bk):
        """QWEN36_MTP_CONV_MULTI warmup self-test, fail closed: the one-call conv path (window build, conv_gates at
        batch M, row split, window commit) against the sequential verify loop + slot commit, bitwise, on THIS image
        at the bucket's T/Bw with layer 0's taps and the live tap shapes (copies only). gdn/conv_multi.selftest."""
        from models.demos.blackhole.qwen36.tt.gdn import conv_multi

        dn = self._gdn_layers()[0]
        tw = dn.tw
        res = conv_multi.selftest(dn.conv_states, [tw["conv_taps"][i] for i in range(dn.K)], tw["dt_bias"],
                                  tw["neg_exp_A"], bk.T, bk.bw, seed=777 + bk.bw)
        if not conv_multi.selftest_ok(res):
            raise RuntimeError(f"QWEN36_MTP_CONV_MULTI self-test failed at Bw={bk.bw} T={bk.T}: {res}")
        logger.info(
            f"QWEN36_MTP_CONV_MULTI self-test exact at Bw={bk.bw} T={bk.T} (conv {dn.conv_states[0].dtype}): "
            f"{sorted(set(res.values()))} mustfail_last={res['mustfail_last']}"
        )

    def _selftest_conv_fold(self, bk):
        """QWEN36_MTP_CONV_FOLD warmup self-test, fail closed: the folded program against build_windows + the image's
        gdn_decode_conv_gates (the pair it replaces, with the arm's reader overlay), bitwise on every chip, at the
        bucket's T/Bw and the live tap shape/dtype, with layer 0's DEVICE conv weights / dt_bias / neg_exp_A and, under
        QWEN36_GDN_DECODE_CHAIN=direct, the served qkvzab column layout. Its wrong-input branch (last candidate one
        token too far) runs at every boot and must differ. Under QWEN36_MTP_CONV_FOLD_MUSTFAIL=1 the candidate carries
        the must-fail and the boot is REFUSED. gdn/conv_fold.selftest."""
        from models.demos.blackhole.qwen36.tt.gdn import conv_fold

        dn = self._gdn_layers()[0]
        tw = dn.tw
        lay = dn._decode_chain()  # None when QWEN36_GDN_DECODE_CHAIN is off
        res = conv_fold.selftest(dn.conv_states, [tw["conv_taps"][i] for i in range(dn.K)], tw["dt_bias"],
                                 tw["neg_exp_A"], bk.T, bk.bw, lay=lay, seed=313 + bk.bw)
        if not conv_fold.selftest_ok(res):
            raise RuntimeError(
                f"{conv_fold.SELFTEST_REFUSED} at Bw={bk.bw} T={bk.T}"
                f"{f' ({conv_fold.ENV_MF}=1: expected)' if conv_fold.mustfail() else ''}: {res}"
            )
        logger.info(
            f"{conv_fold.SELFTEST_OK} at Bw={bk.bw} T={bk.T} (conv {dn.conv_states[0].dtype}, "
            f"{'qkvzab layout' if lay is not None else 'qkv/a/b'}): {res}"
        )

    def _selftest_state_once(self, bk):
        """QWEN36_MTP_GDN_STATE_ONCE warmup self-test, fail closed: the state-once verify + recompute commit against T
        sequential decode_gated_delta_rule_packed launches + the slot commit (the path the flag replaces), bitwise on
        every chip, on THIS image at the bucket's T/Bw and the layer-0 state shape and dtype (fresh copies only):
        o of every token, the untouched base state after the verify, and the full committed state for three idx pages
        (depths 0 and T-1, a SKIP row, an out-of-range entry T, an out-of-bucket entry). Then the explicit must-fail
        commit (last row one token too many) must differ. Under QWEN36_MTP_GDN_STATE_ONCE_MUSTFAIL=1 the served
        commits carry the must-fail, the pages differ and the boot is REFUSED."""
        from models.demos.blackhole.qwen36.tt.gdn import state_once
        from models.demos.blackhole.qwen36.tt.gdn.tp import recurrent_gated_delta_rule_decode_packed_ttnn as seq

        dn = self._gdn_layers()[0]

        def seq_op(conv, beta, g, state):
            return seq(conv, beta, g, dn.Nk, dn.Nv, dn.Dk, dn.Dv, scale=dn.scale, initial_state=state, device=self.mesh,
                       high_precision=False, inplace_state=False, return_row_major=True)

        res = state_once.selftest(dn.rec_state, dn.Nk, dn.Nv, dn.Dk, dn.Dv, dn.scale, bk.T, bk.bw, seq_op,
                                  seed=555 + bk.bw)
        if not state_once.selftest_ok(res):
            raise RuntimeError(
                f"QWEN36_MTP_GDN_STATE_ONCE self-test REFUSED at Bw={bk.bw} T={bk.T}"
                f"{' (QWEN36_MTP_GDN_STATE_ONCE_MUSTFAIL=1: expected)' if state_once.mustfail() else ''}"
                f"{' (QWEN36_MTP_GDN_SO_T0_MUSTFAIL=1: expected)' if state_once.t0_mustfail() else ''}"
                f"{' (QWEN36_MTP_GDN_SO_SFPU_ROW0_MUSTFAIL=1: expected)' if state_once.row0_mustfail() else ''}"
                f"{' (QWEN36_MTP_GDN_SO_QKH_MUSTFAIL=1: expected)' if state_once.qkh_mustfail() else ''}"
                f"{' (QWEN36_MTP_GDN_SO_VSPLIT_MUSTFAIL=1: expected)' if state_once.vsplit_mustfail() else ''}: {res}"
            )
        logger.info(
            f"QWEN36_MTP_GDN_STATE_ONCE self-test exact at Bw={bk.bw} T={bk.T} (rec {dn.rec_state.dtype}, "
            f"{len(ttnn.get_device_tensors(dn.rec_state))} chips{', SO_T0' if 'verify_state_t0' in res else ''}"
            f"{', SFPU_ROW0' if 'mustfail_sfpu_row0' in res else ''}{', SO_QKH' if 'mustfail_so_qkh' in res else ''}): {res}"
        )
        if "verify_vsplit_vs_p1" in res:
            logger.info(f"QWEN36_MTP_GDN_SO_VSPLIT self-test exact at Bw={bk.bw} T={bk.T} (split vs P=1 verify: o, cin, "
                        f"state; mustfail_so_vsplit {res['mustfail_so_vsplit']})")
        if "mustfail_so_qkh" in res:
            logger.info(f"QWEN36_MTP_GDN_SO_QKH self-test exact at Bw={bk.bw} T={bk.T} (o and cin vs the sequential op; "
                        f"mustfail_so_qkh {res['mustfail_so_qkh']})")

    def _selftest_commit_pack(self, bk):
        """QWEN36_MTP_COMMIT_PACK warmup self-test, fail closed (read per call, so an arm switch re-runs it): the packed
        commit of 3 synthetic layers against the per-layer state_once.commit + state_slot.commit_conv_multi, bitwise on
        every chip, on THIS image at the bucket's T/Bw and layer 0's state / tap shapes and dtypes (fresh tensors
        only), for state_once.selftest_pages(); then the explicit must-fail (last layer fed the previous layer's inputs)
        must differ. Under QWEN36_MTP_COMMIT_PACK_MUSTFAIL=1 the served comparisons carry it and the boot is REFUSED.
        Unset: nothing runs."""
        from models.demos.blackhole.qwen36.tt.gdn import commit_pack

        kinds = commit_pack.mode()
        if not kinds:
            return
        dn = self._gdn_layers()[0]
        res = commit_pack.selftest(dn, bk.T, bk.bw, kinds, seed=909 + bk.bw)
        if not commit_pack.selftest_ok(res, kinds):
            raise RuntimeError(
                f"QWEN36_MTP_COMMIT_PACK self-test REFUSED at Bw={bk.bw} T={bk.T} packed={sorted(kinds)}"
                f"{' (QWEN36_MTP_COMMIT_PACK_MUSTFAIL=1: expected)' if commit_pack.mustfail() else ''}: {res}"
            )
        logger.info(f"{commit_pack.SELFTEST_BANNER} at Bw={bk.bw} T={bk.T} packed={sorted(kinds)}: {res}")

    def _selftest_devaccept(self, bk, ids_mc, flag_layout=None, flag_mc=None):
        """QWEN36_MTP_DEVACCEPT warmup self-test, fail closed: the device accept program on THIS image at the bucket's
        (Bw, K), with the served memory configs and layouts (ids as the verify's output, the rule's flags as the
        verify's, the rest as the persistent inputs), on synthetic plans and verify outputs through the REAL
        MTPBatchHost.plan + accept (all / none / partial / mixed acceptance, inactive rows incl. row 0, fresh and stale
        slots, capped KV capacity, FORCE_REJECT, commit_plus1; under the served rule each case again with exact rows
        and edge fp32 accept patterns, plus the chain-noise gather), bitwise on every output and on both chips, plus
        the row-0 must-fail. mtp_accept.selftest."""
        from models.demos.blackhole.qwen36.tt import mtp_accept

        c2 = 2 * spec_sample.k() if _serve() else None
        res = mtp_accept.selftest(self.mesh, bk.bw, bk.K, ids_mc=ids_mc, host_mc=bk.dev["v_tok"].memory_config(),
                                  spec_c2=c2, flag_layout=flag_layout, flag_mc=flag_mc, spec_nz=not _ssp())
        if not mtp_accept.selftest_ok(res):
            bad = {k: v for k, v in res.items() if v != "exact"}
            raise RuntimeError(f"{mbh.ENV_DEVACCEPT} self-test REFUSED at Bw={bk.bw} T={bk.T}: {bad}")
        logger.info(
            f"QWEN36_MTP_DEVACCEPT self-test exact at Bw={bk.bw} T={bk.T}: {len(res) - 1} cases bitwise vs the host "
            f"accept on every chip{' (exact rows + chain noise)' if c2 else ''}, mustfail_row0={res['mustfail_row0']}"
        )

    def _feed_ops(self, bk, outs, d):
        """QWEN36_MTP_DEVFEED: the last program of the sampled family's draft trace (mtp_accept.feed): the next verify's
        v_tok (row 0 = the device accept's t0n, rows j >= 1 = the drafts), v_sqp / v_sqi (the q blocks, ids mapped
        through the draft-vocab table) and v_sx, in place in the bucket's persistent inputs."""
        from models.demos.blackhole.qwen36.tt import mtp_accept

        K = bk.K
        if len(outs) < 3 * K:
            raise RuntimeError(f"{mbh.ENV_DEVFEED}: the chain returned {len(outs)} tensors; the feed needs the K draft ids "
                               f"and the K q blocks (the served rule's chain)")
        with _sc("feed"):
            mtp_accept.feed(d["t0n"], outs[:K], outs[K:2 * K], outs[2 * K:3 * K], self._feed_tab, d["v_tok"], d["v_sqp"],
                            d["v_sqi"], d["v_sx"], bk.bw, K, n_ids=self._feed_n_ids, mf=self._feed_mf)

    def _selftest_feed(self):
        """QWEN36_MTP_DEVFEED warmup self-test, fail closed: the feed program on THIS image at every bucket (Bw, K),
        synthetic chain outputs vs the real host path (finish + plan + the host_inputs builders), bitwise on both chips;
        the corrupted-draft variant must differ. mtp_accept.feed_selftest."""
        from models.demos.blackhole.qwen36.tt import mtp_accept

        dv = getattr(self.model, "_draft_vocab", None)
        ids = list(int(x) for x in dv.ids) if dv is not None else None
        C2 = 2 * spec_sample.k()
        out = {}
        for bw in self.widths:
            bk = self.buckets[bw]
            res = mtp_accept.feed_selftest(self.mesh, bw, bk.K, C2, ids=ids, pad=self._feed_pad)
            if not mtp_accept.feed_selftest_ok(res):
                raise RuntimeError(f"{mbh.ENV_DEVFEED} self-test REFUSED at Bw={bw} K={bk.K}: {res}")
            out[bw] = res
        ttnn.synchronize_device(self.mesh)
        logger.info(f"QWEN36_MTP_DEVFEED self-test exact: buckets {sorted(out)} C2={C2} map={ids is not None}: the fed "
                    f"v_tok / v_sqp / v_sqi / v_sx bitwise vs the host path on every chip, corrupted draft differs {out}")
        if self._feed_pad:
            pads = {bw: (r.get("pad"), r.get("pad_unmasked")) for bw, r in out.items() if "pad" in r}
            logger.info(f"QWEN36_MTP_DEVFEED_PAD self-test exact: buckets {sorted(pads)} with odd rows padded: the active "
                        f"rows bitwise under the row mask, the unmasked compare differs {pads}")

    def compile(self, nblk):
        """Warmup phase 1 (no trace): allocate every persistent input, self-test the commit
        select, and run V + C once eagerly per bucket so every program is in the cache before
        any trace capture (compile-after-capture clobbers parked traces)."""
        if self.compiled:
            return
        for bw in self.widths:
            K = mbh.k_for_bucket(bw, self.kcap, self.kmap)
            self.buckets[bw] = _Bucket(bw, K, self.Bmax, int(nblk), slot=self._slot, da=getattr(self, "_da", False))
        # The self-test fixes the method AND the rec-mask dtype, so it runs before the persistent
        # inputs (the masks among them) are allocated; every bucket re-checks the chosen pair.
        for bw in self.widths:
            if self._slot:
                self._selftest_slot(self.buckets[bw])
                if self._conv_multi:
                    self._selftest_conv_multi(self.buckets[bw])
                if getattr(self, "_conv_fold", False):
                    self._selftest_conv_fold(self.buckets[bw])
                if self._state_once:
                    self._selftest_state_once(self.buckets[bw])
                    self._selftest_commit_pack(self.buckets[bw])
            else:
                self._selftest_commit(self.buckets[bw])
        dv = getattr(self.model, "_draft_vocab", None)
        if dv is not None:
            dv.selftest(self.widths)  # QWEN36_MTP_DRAFT_VOCAB: fails closed (MUSTFAIL: must catch the shift)
        # QWEN36_MTP_CAP_TORCH: the vectorised KV-capacity scan must give the served scan's integer
        # on every row shape (fail closed; MUSTFAIL: the all-non-null count differs on a row with a
        # hole).
        if mbh.cap_torch_enabled():
            mbh.selftest_cap_torch()
        if mbh.cap_torch_enabled() and self._alloc_cap:  # the scan only runs on the async-capacity path
            logger.info(
                "QWEN36_MTP_CAP_TORCH engaged: the per-row KV capacity from a vectorised scan of the page "
                "table (no .tolist(), no python loop); the same integer as the served scan"
            )
        # QWEN36_ARGMAX_FOLD (argmax_fold.py): self-test the folded greedy combine at every (B, per) this boot's
        # argmax calls run (draft widths Bw, verify widths M; the full head's vocab/TP and the draft vocab's N/TP),
        # before any trace; fails closed.
        if argmax_fold.enabled():
            model = self.model
            vocab_per = int(model.args.vocab_size) // int(model.num_devices)
            served = [(int(self.buckets[w].K) + 1, int(w)) for w in self.widths]  # this boot's actual (T, Bw)
            cells = argmax_fold.served_cells(self.Bmax, vocab_per, draft_per=(int(dv.per) if dv is not None else None),
                                             served=served)
            argmax_fold.ensure_selftest(self.mesh, model.tt_ccl, model.args.ccl_topology(), cells=cells)
        # QWEN36_DEVICE_SAMPLE (dev_sample.py): the device-side sampled decode (the topk + the
        # all-gather + the Gumbel-max pick over the TP-sharded vocab). The serving path is NOT wired
        # yet (the MTP/plain-decode traces are shared across steps, so a per-step sampler needs a
        # second captured trace or a per-step select -- see the report); the boot self-test here
        # validates the lever's DEVICE CODE on the real card at the served ROW counts (the verify
        # widths). The full vocab (per = vocab/TP) needs ttnn.experimental.topk_large_indices (the
        # uint16 index path would wrap); if the image's tt-metal does not have it, the self-test
        # falls back to a reduced vocab and WARNS (the lever changes nothing at runtime yet).
        if dev_sample.enabled():
            model = self.model
            vocab_per = int(model.args.vocab_size) // int(model.num_devices)
            served = [(int(self.buckets[w].K) + 1, int(w)) for w in self.widths]  # this boot's actual (T, Bw)
            rows = sorted({int(t) * int(bw) for t, bw in served} | {int(bw) for _, bw in served}
                          | set(argmax_fold.plain_widths(self.Bmax)))
            try:
                dev_sample.ensure_selftest(self.mesh, model.tt_ccl, model.args.ccl_topology(),
                                           cells=[(r, vocab_per) for r in rows])
            except RuntimeError as e:
                if "topk_large_indices" not in str(e):
                    raise
                # The SERVED path needs the full vocab (per = vocab/TP), and the uint16 topk index
                # path would wrap there.  Refuse the boot: serving a wrong token is worse than not
                # booting (the two-stage top-K fallback is the follow-up).
                raise RuntimeError(
                    f"{dev_sample.ENV} REFUSED: the served vocab (per={vocab_per}) needs "
                    f"ttnn.experimental.topk_large_indices (the uint16 index path would wrap) and this "
                    f"image's tt-metal does not provide it ({e}). The device sampler cannot serve the "
                    f"served vocab without it; the boot is refused rather than serving a wrong token.") from e
        # QWEN36_MTP_SPEC_SAMPLE (spec_sample.py): exact speculative sampling. The boot self-test checks the
        # device rule on both chips at every verify row count, bitwise vs the host's fp32 arithmetic, with
        # the flipped-accept branch differing. _SERVE wires it into the served step.
        if spec_sample.enabled():
            served = [(int(self.buckets[w].K) + 1, int(w)) for w in self.widths]  # this boot's actual (T, Bw)
            if spec_sample.serve():
                if not dev_sample.enabled():
                    raise RuntimeError(f"{spec_sample.SERVE_ENV}=1 needs {dev_sample.ENV}=1 (the target's sample and "
                                       f"the per-row greedy select are the device sampler's verify block)")
                if dev_sample.k() != spec_sample.k():
                    raise RuntimeError(f"{spec_sample.SERVE_ENV}=1 needs {dev_sample.ENV}_K ({dev_sample.k()}) == "
                                       f"{spec_sample.K_ENV} ({spec_sample.k()}): the target's block is cut from the "
                                       f"verify's dev_sample block")
                if self.host.reject_all:
                    raise RuntimeError(f"{spec_sample.SERVE_ENV}=1 with QWEN36_MTP_FORCE_REJECT: a forced rejection "
                                       f"has no exact emission; refusing the boot")
                if self._timing:
                    raise RuntimeError(f"{spec_sample.SERVE_ENV}=1 with QWEN36_MTP_BATCH_TIMING: the timing step "
                                       f"does not carry the drafter's blocks; refusing the boot")
                self.host.spec_c2 = 2 * spec_sample.k()
            spec_sample.ensure_selftest(self.mesh, sorted({t * bw for t, bw in served}), K=spec_sample.k())
            spec_sample.log_serve_banner(spec_sample.k())
            spec_sample.log_fast_noise_banner()
            if spec_sample.serve():
                self._lean_selftests(served)
                spec_sample.log_lean_banners()
            if fused_sampler.enabled():
                if not spec_sample.serve():
                    raise RuntimeError(f"{fused_sampler.ENV}=1 needs {spec_sample.SERVE_ENV}=1 (it replaces the served "
                                       f"exact-sampling chain)")
                model = self.model
                dv = getattr(model, "_draft_vocab", None)
                vper = int(model.args.vocab_size) // int(model.num_devices)
                dper = int(dv.per) if dv is not None else vper
                cells = [(t * bw, vper, True) for t, bw in served] + [(bw, dper, False) for _t, bw in served]
                fused_sampler.ensure_selftest(self.mesh, model.tt_ccl, model.args.ccl_topology(), cells,
                                              spec_sample.k())
            if fused_sampler.ssp():
                # QWEN36_MTP_SPEC_SAMPLE_SSP (fused_sampler.py): the rule, the noise and the id map in fs_cut
                if not (fused_sampler.enabled() and spec_sample.serve()):
                    raise RuntimeError(f"{fused_sampler.SSP_ENV}=1 needs {fused_sampler.ENV}=1 and "
                                       f"{spec_sample.SERVE_ENV}=1 (it extends the served fused sampler)")
                if not getattr(self, "_da", False):
                    raise RuntimeError(f"{fused_sampler.SSP_ENV}=1 needs {mbh.ENV_DEVACCEPT}=1 (the chain's noise is "
                                       f"keyed by the device accept's m page)")
                if any(int(t) * int(bw) > fused_sampler.NP_MAX_VROWS or int(bw) > fused_sampler.NP_MAX_CROWS
                       for t, bw in served):
                    raise RuntimeError(f"{fused_sampler.SSP_ENV}=1: a bucket {served} exceeds the noise plan page")
                model = self.model
                dv = getattr(model, "_draft_vocab", None)
                table = map_ref = None
                if dv is not None:
                    table = fused_sampler.id_table(dv.ids, self.mesh, shift=1 if dv.mustfail else 0)
                    map_ref = dv.map_ids
                self.head._ssp_table = table
                vper = int(model.args.vocab_size) // int(model.num_devices)
                dper = int(dv.per) if dv is not None else vper
                cells = [(t * bw, vper, True) for t, bw in served] + [(bw, dper, False) for _t, bw in served]
                fused_sampler.ensure_ssp_selftest(self.mesh, model.tt_ccl, model.args.ccl_topology(), cells,
                                                  spec_sample.k(), depths=range(max(int(t) for t, _ in served) - 1),
                                                  table=table, map_ref=map_ref)
        for bw in self.widths:
            self.buckets[bw].rec_mask_dtype = self.rec_mask_dtype
            self._alloc_inputs(self.buckets[bw])
        da_cfg = {}
        if self._feed:
            dv = getattr(self.model, "_draft_vocab", None)
            if dv is not None:  # the served q-id map's table (mtp_batch._read_chain maps through dv.ids, unshifted)
                self._feed_tab = fused_sampler.id_table(dv.ids, self.mesh, shift=0)
                self._feed_n_ids = len(dv.ids)
        # QWEN36_MTP_SPEC_SAMPLE_DUAL: the greedy family's programs (the flags-off verify argmax, draft head and greedy
        # device accept) are compiled here too: nothing may compile after a trace is captured.
        fams = ("s", "g") if getattr(self, "_dual", False) else ("s",)
        for bw in self.widths:
            bk = self.buckets[bw]
            logger.info(f"QWEN36_MTP_BATCH compile: bucket Bw={bw} K={bk.K} M={bk.M} page-table blocks={bk.nblk}"
                        + (f" families {list(fams)}" if len(fams) > 1 else ""))
            for fam in fams:
                with _family(fam):
                    v_out = self._verify_ops(bk, bk.dev)
                    if bk.da:  # QWEN36_MTP_DEVACCEPT: the self-test runs the program at the served placements and layouts
                        cfg = {"ids_mc": v_out[0].memory_config()}
                        if len(v_out) >= 5:
                            cfg.update(flag_layout=v_out[3].layout, flag_mc=v_out[3].memory_config())
                        da_cfg[(bw, fam)] = cfg
                    drafts = self._commit_draft_ops(bk, bk.dev, v_out, release=True)
                    if self._feed and _serve():
                        self._feed_ops(bk, drafts, bk.dev)  # QWEN36_MTP_DEVFEED: compiled before any capture
                    for t in list(v_out) + drafts:
                        ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)
        if da_cfg:
            for bw in self.widths:
                for fam in fams:
                    with _family(fam):  # the greedy family's accept is the flags-off (no rule outputs) program
                        self._selftest_devaccept(self.buckets[bw], **da_cfg[(bw, fam)])
            ttnn.synchronize_device(self.mesh)
        if self._feed:
            self._selftest_feed()
        self.host.reset_all()
        self.compiled = True

    def _lean_selftests(self, served):
        """The opus/sscost levers' boot self-tests (fail closed), before any capture.

        HOSTSPLIT: per bucket, the split host builders bitwise vs the full one, and the shifted-chain branch must
        differ (HOSTSPLIT_MUSTFAIL=1 serves that branch, so the check refuses).  RMSHARE / DRAFTPICK: the device
        checks in spec_sample.lean_selftest at this boot's verify and chain cells."""
        model = self.model
        if spec_sample.host_split():
            res = {}
            for bw in self.widths:
                bk = self.buckets[bw]
                n, names = bk.hostsplit_selftest(mf=spec_sample.host_split_mustfail())
                nm, _ = bk.hostsplit_selftest(mf=True)
                res[f"B{bw}"] = {"diffs": n, "names": names, "mustfail_diffs": nm}
            if any(r["diffs"] or not r["mustfail_diffs"] for r in res.values()):
                raise RuntimeError(f"{spec_sample.HOSTSPLIT_ENV} self-test REFUSED: {res}")
            logger.info(f"QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT self-test exact: split host blocks bitwise vs the full builder "
                        f"at buckets {sorted(self.widths)} (the shifted-chain branch differs) {res}")
        if spec_sample.rm_share() and not argmax_fold.enabled():
            raise RuntimeError(f"{spec_sample.RMSHARE_ENV}=1 needs QWEN36_ARGMAX_FOLD=1 (greedy_rm is its row-major entry)")
        if spec_sample.rm_share() or spec_sample.draft_pick():
            dv = getattr(model, "_draft_vocab", None)
            vper = int(model.args.vocab_size) // int(model.num_devices)
            dper = int(dv.per) if dv is not None else vper
            spec_sample.ensure_lean_selftest(self.mesh, model.tt_ccl, model.args.ccl_topology(),
                                             verify_cells=[(t * bw, vper) for t, bw in served],
                                             draft_cells=[(bw, dper) for _t, bw in served], K=spec_sample.k())

    def _capture_family(self, bk):
        """Capture one trace family of bucket bk (the current _family): V, then R / cc / cd (or C2, or C) as the
        flags select, into the bucket's attributes (v_trace, v_out, ..., c_out)."""
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        try:
            bk.v_out = self._verify_ops(bk, bk.dev)
        finally:
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        bk.v_trace = tid
        if getattr(self, "_early_reseed", False):
            # QWEN36_MTP_EARLY_RESEED: the same ops as the one C trace with the reseed moved in front of the
            # commit (they touch disjoint buffers: MTP head KV vs GDN state), captured as R + C2 so R can be
            # enqueued before the host accept.
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._reseed_ops(bk, bk.dev, bk.v_out)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            bk.r_trace = tid
            if getattr(self, "_c_split", False):
                # QWEN36_MTP_SO1_CSPLIT under EARLY_RESEED: C2 captured as cc (commit) + cd (draft steps only;
                # the reseed already ran in R). Same ops as C2; _step_overlap enqueues cd before cc.
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    self._commit_ops(bk, bk.dev, release=False)
                finally:
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                bk.cc_trace = tid
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    self._accept_ops(bk, bk.dev, bk.v_out)  # QWEN36_MTP_DEVACCEPT: cd runs before cc
                    bk.c_out = self._draft_steps(bk, bk.dev, bk.v_out)
                    if self._feed and _serve():
                        self._feed_ops(bk, bk.c_out, bk.dev)  # QWEN36_MTP_DEVFEED: the last program of cd
                        if not self._feed_logged:
                            self._feed_logged = True
                            logger.info(f"QWEN36_MTP_DEVFEED engaged: the sampled draft trace ends with the next verify's "
                                        f"v_tok / v_sqp / v_sqi / v_sx built on device; fast steps enqueue the verify "
                                        f"before the draft readback (check {self._feed_check} steps)"
                                        f"{' MUSTFAIL' if self._feed_mf else ''}")
                finally:
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                bk.cd_trace = tid
            else:
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                try:
                    self._accept_ops(bk, bk.dev, bk.v_out)  # QWEN36_MTP_DEVACCEPT
                    self._commit_ops(bk, bk.dev, release=False)
                    bk.c_out = self._draft_steps(bk, bk.dev, bk.v_out)
                finally:
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                bk.c2_trace = tid
        elif self._timing or getattr(self, "_c_split", False):
            # Same ops as the one C trace, captured as two: TIMING times them apart; OVERLAP+so1
            # C-split enqueues cd then cc so wait_c can drop the commit recompute.
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._commit_ops(bk, bk.dev, release=False)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            bk.cc_trace = tid
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._accept_ops(bk, bk.dev, bk.v_out)  # QWEN36_MTP_DEVACCEPT (C-split: cd runs before cc)
                bk.c_out = self._draft_ops(bk, bk.dev, bk.v_out)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            bk.cd_trace = tid
        else:
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                bk.c_out = self._commit_draft_ops(bk, bk.dev, bk.v_out, release=False)
            finally:
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            bk.c_trace = tid
        if getattr(self, "_optrace", None) is not None:
            self._optrace.note_family(bk)
        # the verify leftovers this family's commit trace reads (gdn/tp.py keeps them in _mtp_verify[Bw]; a second
        # family's verify capture re-binds that entry, so the family record holds them for the life of its traces)
        return [getattr(dn, "_mtp_verify", {}).get(bk.bw) for dn in self._gdn_layers()]

    def capture(self, nblk):
        """Warmup phase 2: capture V and C per bucket. The outputs (and the GDN per-sub-step
        states) stay allocated for the life of the traces. QWEN36_MTP_SPEC_SAMPLE_DUAL: per bucket the sampled family
        first, then the greedy family if the TRACE region still holds it (else that bucket's greedy steps run the
        sampled family, logged and counted with the reason)."""
        if self.captured:
            return
        if not self.compiled:
            self.compile(nblk)
        dual = getattr(self, "_dual", False)
        start = _trace_region(self.mesh)
        prev_need = None  # bytes the previous (narrower) bucket's traces took (the sampled family under DUAL)
        prev_T = None  # and its verify depth
        g_total, g_skipped = 0, []
        ot = getattr(self, "_optrace", None)
        if ot is not None:
            ot.capture_begin()
        for bw in self.widths:
            bk = self.buckets[bw]
            ttnn.synchronize_device(self.mesh)
            before = _trace_region(self.mesh)
            # Fail closed BEFORE a capture that cannot fit (tt-trace-region-budget): an overflow is a
            # TT_FATAL at mesh_trace.cpp:82 late in boot that often needs SIGKILL. A wider bucket runs the
            # same programs as the narrower one, so its traces are budgeted at 1.5x the narrower's measured
            # bytes, times T / T_prev when it verifies deeper (QWEN36_MTP_BATCH_K_BY_BUCKET: the draft trace holds
            # K-1 MTP-head steps; T <= T_prev keeps the old 1.5x). The first bucket is left to the TT_FATAL.
            grow = 1.5 * max(1.0, bk.T / prev_T) if prev_T else 1.5
            if prev_need is not None and before is not None and before[1] < grow * prev_need:
                raise RuntimeError(
                    f"QWEN36_MTP_BATCH: TRACE region too small for bucket Bw={bw}: {before[1] / _MIB:.1f} MiB free, "
                    f"need ~{grow * prev_need / _MIB:.1f} MiB ({grow:.2f}x the previous bucket's {prev_need / _MIB:.1f} MiB); "
                    f"{before[0] / _MIB:.1f} MiB already used. Raise [tt_config] trace_region_size."
                )
            keep = self._capture_family(bk)
            bk.save_fam("s", keep)
            after = _trace_region(self.mesh)
            delta = ""
            if None not in (before, after):
                prev_need = after[0] - before[0]
                prev_T = bk.T
                delta = f", TRACE region +{prev_need / _MIB:.1f} MiB"
            gdelta = ""
            if dual:
                # The greedy family holds the sampled family's programs minus the sampler's (no wide top-K, cut, rule,
                # chain cut): budgeted at 1.1x the sampled family's measured bytes. Short of that the bucket keeps the
                # sampled family only (a logged, counted fallback; never an overflow).
                ttnn.synchronize_device(self.mesh)
                g_before = _trace_region(self.mesh)
                s_need = (after[0] - before[0]) if None not in (before, after) else None
                if s_need is not None and g_before is not None and g_before[1] < 1.1 * s_need:
                    g_skipped.append(bw)
                    logger.info(
                        f"QWEN36_MTP_SPEC_SAMPLE_DUAL greedy family NOT captured at Bw={bw}: TRACE region "
                        f"{g_before[1] / _MIB:.1f} MiB free < 1.1 x the sampled family's {s_need / _MIB:.1f} MiB; "
                        f"greedy steps at this bucket run the sampled family (fallback reason 'no greedy trace')"
                    )
                else:
                    with _family("g"):
                        keep = self._capture_family(bk)
                    bk.save_fam("g", keep)
                    g_after = _trace_region(self.mesh)
                    if None not in (g_before, g_after):
                        g_total += g_after[0] - g_before[0]
                        gdelta = f"; greedy family TRACE region +{(g_after[0] - g_before[0]) / _MIB:.1f} MiB"
                bk.use_fam("s")
            c_split = getattr(self, "_c_split", False)  # like _early_reseed: absent on objects built without __init__
            cdesc = "verify + cc/cd (so1 C-split)" if c_split else "verify + commit/draft"
            logger.info(f"QWEN36_MTP_BATCH traces captured: bucket Bw={bw} K={bk.K} ({cdesc}{delta}{gdelta})")
            if bk.da and not getattr(self, "_da_logged", False):
                self._da_logged = True
                logger.info(
                    f"QWEN36_MTP_DEVACCEPT engaged: accept on device as the first program of the "
                    f"{'cd' if c_split else ('C2' if getattr(self, '_early_reseed', False) else 'C')} trace "
                    f"({'greedy + exact-rule rows, chain noise gathered by m' if spec_sample.serve() else 'greedy rows'}); "
                    f"V, R and the commit/draft traces enqueued back to back, every host write before V, the host "
                    f"accept off the device's critical path and checked against the device m/t0n page at the drain"
                )
            if c_split and not getattr(self, "_csplit_logged", False):
                self._csplit_logged = True
                logger.info(
                    f"QWEN36_MTP_SO1_CSPLIT engaged: C{'2' if getattr(self, '_early_reseed', False) else ''} captured as cc (GDN commit) + cd "
                    f"({'draft steps, reseed in R' if getattr(self, '_early_reseed', False) else 'reseed+drafts'}); cd enqueued first, draft "
                    "event is wait_c; cc follows and is waited before the next verify"
                )
        if ot is not None:
            ot.capture_end()
        ttnn.synchronize_device(self.mesh)
        end = _trace_region(self.mesh)
        if None not in (start, end):
            logger.info(
                f"QWEN36_MTP_BATCH TRACE budget: MTP traces +{(end[0] - start[0]) / _MIB:.1f} MiB; region "
                f"{end[0] / _MIB:.1f} MiB used, {end[1] / _MIB:.1f} MiB free of {(end[0] + end[1]) / _MIB:.1f} MiB"
            )
        if dual and not getattr(self, "_dual_logged", False):
            self._dual_logged = True
            dual_bws = [w for w in self.widths if "g" in self.buckets[w].fams]
            logger.info(
                f"QWEN36_MTP_SPEC_SAMPLE_DUAL engaged: greedy trace family (no sampler program) captured beside the "
                f"sampled family at buckets {dual_bws} (sampled only: {g_skipped}); a step whose active rows are all "
                f"T=0 replays the greedy family, any active T>0 row the sampled one; greedy family TRACE region "
                f"+{g_total / _MIB:.1f} MiB"
            )
        self.captured = True

    # ------------------------------------------------------------------ serving
    def _fallback(self, why, tokens=None, start_pos=None, bw=None):
        self.drain(steady=True)  # QWEN36_MTP_DEVFEED left the pending step for the step to drain (idempotent)
        self._ov_part = None  # a plain step in between: the next speculative step's gap is not steady
        self._ensure_committed()  # plain decode reads the same GDN state the deferred commit would write
        self._fallbacks[why] = self._fallbacks.get(why, 0) + 1
        n = self._fallbacks[why]
        if n == 1 or n % 1000 == 0:
            logger.info(f"QWEN36_MTP_BATCH plain-decode fallback ({why}): count={n}")
        if self.host.trace and tokens is not None and start_pos is not None:
            try:
                toks = tokens.reshape(-1).tolist()
                posl = start_pos.reshape(-1).tolist()
                w = len(toks) if bw is None else int(bw)
                self._plain_steps = getattr(self, "_plain_steps", 0) + 1
                for line in self.host.trace_plain(self._plain_steps, toks, posl, w, why):
                    logger.info(line)
            except Exception as e:  # the trace must never change what the step does
                logger.info(f"QWEN36_MTP_BATCH trace: plain-step trace failed ({type(e).__name__}: {e})")
        return None

    def log_stats(self, tag="stats", burst_end=False):
        # t= is the last speculative step's clock (the counters cover steps up to it), burst_end=1 marks
        # the line on_release logs when the batch drained: ttx assigns lines to bench cells by t.
        t = f" t={self._last_step_t:.3f}" if self._last_step_t is not None else ""
        da = (f" da_checked={self._da_stats['checked']} da_mismatch={self._da_stats['mismatch']}"
              if getattr(self, "_da", False) else "")
        st = getattr(self, "_dual_stats", None)
        dual = (f" dual_g={st['g']} dual_s={st['s']} dual_switch={st['switch']} dual_fallback={st['fallback']} "
                f"dual_qdrop={self.host.dual_qdrop}" if getattr(self, "_dual", False) and st is not None else "")
        fs = getattr(self, "_feed_stats", None)
        feed = (f" feed_fast={fs['fast']} feed_slow={fs['slow']} feed_miss={fs['miss']} feed_checked={fs['checked']} "
                f"feed_by_bw={ {k: f'{v[0]}/{v[1]}' for k, v in sorted(fs['by_bw'].items())} }"
                + (f" feed_pad_fast={fs['pad_fast']} feed_pad_checked={fs['pad_checked']}"
                   if getattr(self, "_feed_pad", False) else "")
                if getattr(self, "_feed", False) and fs is not None else "")
        logger.info(f"QWEN36_MTP_BATCH {tag}: {self.host.summary()}{da}{dual}{feed} fallbacks={self._fallbacks}{t} "
                    f"burst_end={int(bool(burst_end))}")

    def on_release(self):
        """A request left. When the last one has gone, log the final counters of this burst: the
        periodic line (every QWEN36_MTP_BATCH_STATS_EVERY steps) misses the tail of a batch."""
        if self.host.idle() and self._steps > self._idle_logged_at:
            self._idle_logged_at = self._steps
            for timer in (self._timer, self._ov_timer):
                if timer is not None:  # close the TIMING window: it must not span two bursts
                    for line in timer.lines(fallbacks=dict(self._fallbacks), flush="idle"):
                        logger.info(line)
            self.log_stats("stats", burst_end=True)

    def _ensure_committed(self):
        """OVERLAP+so1 C-split: the deferred GDN commit event must complete before the next verify (or any
        path that moves slot/GDN state). Idempotent."""
        ev, self._commit_event = self._commit_event, None
        if ev is not None:
            ttnn.event_synchronize(ev)

    def defers_drain(self):
        """QWEN36_MTP_DEVFEED: True when a steady caller (decode_forward without a remap, the row-id hook without a
        trace) should leave the pending step to maybe_step (which drains it after enqueueing a fast step's verify)."""
        return bool(getattr(self, "_feed", False)) and not self.host.trace

    def drain(self, steady=False):
        """QWEN36_MTP_OVERLAP: finish the deferred step (wait for its draft event, read the draft ids,
        store them per slot, log its trace/stats). Idempotent; every caller that reads or moves the host slot
        state calls it first (qwen36_vllm: decode_forward, prefill_forward, release_request, the row-id hook).
        steady=False (a prefill, release or remap follows) drops the open OVERLAP_TIMING sample: its gap is
        not the steady host work between two speculative steps, and any deferred C-split commit is waited
        here too. steady=True under C-split waits only for the draft event (wait_c); the commit event is
        deferred to _ensure_committed() right before the next verify so host prep can hide it."""
        if not steady:
            self._ov_part = None
            self._ensure_committed()
        p = self._pending
        if p is None:
            return
        self._pending = None
        t0 = time.perf_counter()
        ttnn.event_synchronize(p.event)
        t1 = time.perf_counter()
        drafts, blocks = self._read_chain([ttnn.to_torch(ttnn.get_device_tensors(h)[0]) for h in p.reads],
                                          p.plan.K, p.bw)
        t2 = time.perf_counter()
        if p.da_m is not None:
            self._da_check(p)
        t3 = time.perf_counter()
        self._ov_wait_ms += (t3 - t0) * 1000.0
        if self._ov_timer is not None:
            sub = self._ov_sub
            for k, v in (("d_ev", t1 - t0), ("d_read", t2 - t1), ("d_chk", t3 - t2)):
                sub[k] = sub.get(k, 0.0) + v * 1000.0
        if p.commit_event is not None:
            if steady:
                # Next verify needs the commit; leave it for _ensure_committed after plan/write.
                self._commit_event = p.commit_event
            else:
                ttnn.event_synchronize(p.commit_event)
        self.host.complete(drafts, blocks)
        self._log_step(p.plan, p.acc, p.ids, drafts)

    def _da_check(self, p):
        """QWEN36_MTP_DEVACCEPT: the device's m / t0n page (every chip) must equal the host accept's m and t0_next for
        the step whose commit and drafts it drove. Fail closed (RuntimeError) unless the must-fail is set, which logs
        and counts instead, so the must-fail reaches the served equivalence check."""
        bad = []
        for di, t in enumerate(ttnn.get_device_tensors(p.da_m)):
            words = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(t).reshape(-1)[: mbh.DA_M_LEN].tolist()]
            rows = mbh.da_check_page(words, p.acc, p.bw)
            if rows:
                bad.append((di, rows, words[: p.bw], words[32: 32 + p.bw]))
        self._da_stats["checked"] += 1
        if not bad:
            return
        self._da_stats["mismatch"] += 1
        n = self._da_stats["mismatch"]
        exp = ([int(x) for x in p.acc.m], [int(x) for x in p.acc.t0_next])
        if not self._da_mf:
            raise RuntimeError(f"{mbh.ENV_DEVACCEPT}: device (chip, rows, m, t0n) {bad} != host (m, t0n) {exp} at "
                               f"Bw={p.bw} (step {self._steps}): the commit/draft traces ran on a different accepted "
                               f"length; refusing to go on")
        if n == 1 or n % 256 == 0:
            logger.info(f"{mbh.ENV_DEVACCEPT_MUSTFAIL}: device {bad} != host {exp} (expected under the must-fail), "
                        f"count={n}")

    def _read_chain(self, ts, K, bw):
        """The chain's host reads [K draft ids, (QWEN36_MTP_SPEC_SAMPLE) K probs blocks, K id blocks] ->
        (drafts K*bw k-major, (qprobs [K][bw, C2], qids [K][bw, C2] FULL ids) or None). Under
        QWEN36_MTP_DRAFT_VOCAB the block's ids are sub-ids: mapped here through the head's id table."""
        drafts = []
        for t in ts[:K]:
            drafts.extend(int(v) for v in t.reshape(-1)[:bw].tolist())
        if len(ts) <= K:
            return drafts, None
        qpb = [t.reshape(bw, -1).float() for t in ts[K:2 * K]]
        qib = [t.reshape(bw, -1).float() for t in ts[2 * K:3 * K]]
        dv = getattr(self.model, "_draft_vocab", None)
        if dv is not None:
            tab = getattr(self, "_dv_ids_t", None)
            if tab is None:
                tab = self._dv_ids_t = torch.as_tensor(dv.ids, dtype=torch.int64)
            qib = mbh.map_q_ids(qib, tab)
        return drafts, (qpb, qib)

    def maybe_step(self, tokens, start_pos, page_table, sampling_params, enable_trace, slot_remap_set):
        """One speculative step, or None to run plain decode. Returns (first_ids [W] torch.int32,
        extras [W] lists): row b emits first_ids[b] followed by extras[b]."""
        t_entry = time.perf_counter()
        if not (getattr(self, "_feed", False) and self._pending is not None):
            self.drain(steady=True)  # QWEN36_MTP_DEVFEED: a pending step is drained by the step (or _fallback)
        if not self.captured or not enable_trace:
            self._ov_part = None
            return self._fallback("not captured")
        if self._plain_only:
            return self._fallback("plain only", tokens, start_pos)
        if sampling_params is None:
            return self._fallback("host sampling", tokens, start_pos)
        temps = getattr(sampling_params, "temperature", None)
        if temps is not None and not dev_sample.enabled():
            tl = temps if isinstance(temps, (list, tuple)) else [temps]
            if any(t is not None and float(t) != 0.0 for t in tl):
                return self._fallback("temperature>0", tokens, start_pos)
        if tokens is None or start_pos is None or page_table is None:
            return self._fallback("missing inputs")
        W = int(tokens.shape[0])
        pos = start_pos.reshape(-1)
        num_active = int((pos != -1).sum())
        if self._bucketing:
            bw = mbh.bucket_for(num_active, W, slot_remap_set)
        else:
            bw = W
        bk = self.buckets.get(bw)
        if bk is None:
            return self._fallback(f"bucket {bw} not speculating", tokens, start_pos, bw)
        # QWEN36_DEVICE_SAMPLE: the per-row seed and temperature key the verify's Gumbel block.
        bk.sp = sampling_params
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        if int(pt.shape[-1]) != bk.nblk:
            return self._fallback(f"page-table width {int(pt.shape[-1])} != captured {bk.nblk}", tokens, start_pos, bw)
        toks = tokens.reshape(-1)[:bw].tolist()
        posl = pos[:bw].tolist()
        if bk.mpt:
            bk.pt_shift = list(self.host.pt_shift[:bw])  # QWEN36_PREFIX_CACHING: v_mpt = v_pt shifted past each row's hit
        cap = [bk.nblk * self.block_size] * bw
        if self._alloc_cap:
            if mbh.cap_torch_enabled():
                # QWEN36_MTP_CAP_TORCH: the same bound from a vectorised scan of the page-table row
                # (no .tolist(), no python loop) -- this runs in the "prep" phase, where the device
                # is idle waiting for the V trace.
                cap = [min(cap[b], mbh.alloc_capacity_t(pt[b], self.block_size)) for b in range(bw)]
            else:
                cap = [min(cap[b], mbh.alloc_capacity(pt[b].tolist(), self.block_size)) for b in range(bw)]
        fam = self._dual_pick(bk, bw, sampling_params, posl)
        ot = getattr(self, "_optrace", None)
        if ot is not None:
            # QWEN36_MTP_OPTRACE: open / continue / close a named device-time window (diagnostic, host-side only)
            ot.pre(bw, fam, posl, num_active)
            out = self._dispatch(bk, bw, W, toks, posl, cap, pt, t_entry, fam)
            ot.post(posl)
            return out
        return self._dispatch(bk, bw, W, toks, posl, cap, pt, t_entry, fam)

    def _dispatch(self, bk, bw, W, toks, posl, cap, pt, t_entry, fam):
        """maybe_step's step under the family `fam`: the timed, device-accept, overlap or plain step."""
        with _family(fam):
            bk.use_fam(fam)
            if self._timing:
                return self._step_timed(bk, bw, W, toks, posl, cap, pt)
            if getattr(self, "_da", False):
                return self._step_devaccept(bk, bw, W, toks, posl, cap, pt, t_entry)
            if self._overlap:
                return self._step_overlap(bk, bw, W, toks, posl, cap, pt, t_entry)
            return self._step_plain(bk, bw, W, toks, posl, cap, pt)

    def _dual_pick(self, bk, bw, sp, posl):
        """QWEN36_MTP_SPEC_SAMPLE_DUAL: this step's trace family ("g" when every active row is T = 0, else "s"; see
        mbh.dual_family) and the plan() arguments it implies (self._plan_kw). Without the flag: "s" and no arguments,
        i.e. the single-family step exactly as before."""
        if not getattr(self, "_dual", False):
            self._plan_kw = {}
            return "s"
        fam, need_q, n, why = mbh.dual_family(sp, posl, bw, rules=spec_sample.row_rules(sp, bw),
                                              mustfail=getattr(self, "_dual_mf", False),
                                              have_greedy="g" in bk.fams)
        st = self._dual_stats
        st[fam] += 1
        if why is not None:
            st["fallback"][why] = st["fallback"].get(why, 0) + 1
            if st["fallback"][why] == 1:
                logger.info(f"QWEN36_MTP_SPEC_SAMPLE_DUAL: step at Bw={bw} with {n} sampled row(s) runs family {fam} "
                            f"({why}; counted on the stats line)")
        if st["last"] is not None and st["last"] != fam:
            st["switch"] += 1
        st["last"] = fam
        self._plan_kw = {"need_q": need_q, "spec": fam == "s"}
        return fam

    def _step_plain(self, bk, bw, W, toks, posl, cap, pt):
        """The synchronous step (no OVERLAP): plan, V, host accept, C, draft readback."""
        plan = self.host.plan(toks, posl, bw, bk.K, cap, **getattr(self, "_plan_kw", {}))
        split = spec_sample.host_split()  # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: build only what each write sends
        vals = bk.host_inputs(plan=plan, acc=_PlanOnly(bk), pt=pt[:bw],
                              want=(lambda n: n.startswith("v_")) if split else None)
        self._write_inputs(bk, {n: v for n, v in vals.items() if n.startswith("v_")})
        ttnn.execute_trace(self.mesh, bk.v_trace, cq_id=0, blocking=False)
        sreads = self._spec_reads(bk)
        ids = _read_ids(bk.v_out[0], bk.M)
        acc = self.host.accept(plan, ids, self._spec_data(bk, sreads))
        vals = bk.host_inputs(plan=plan, acc=acc, pt=pt[:bw],
                              want=(lambda n: not n.startswith("v_")) if split else None)
        self._write_inputs(bk, {n: v for n, v in vals.items() if not n.startswith("v_")})
        ttnn.execute_trace(self.mesh, bk.c_trace, cq_id=0, blocking=False)
        drafts, blocks = self._read_chain([ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in bk.c_out], bk.K, bw)
        self.host.finish(plan, acc, drafts, blocks)
        return self._step_done(plan, acc, ids, drafts, W)

    def _spec_reads(self, bk):
        """QWEN36_MTP_SPEC_SAMPLE: non-blocking host reads of the rule's outputs (issued right after the verify
        so the early-reseed path does not wait on them). None when the served rule is off."""
        if not _serve():
            return None
        vo = bk.v_out
        if vo is None or len(vo) < 5:
            raise RuntimeError("QWEN36_MTP_SPEC_SAMPLE: the verify returned no rule outputs; refusing to serve")
        return [vo[3].cpu(blocking=False), vo[4].cpu(blocking=False)]

    def _spec_data(self, bk, reads):
        """The rule's outputs for host.accept (after the verify completed): accept [M], resid [M] and the
        step's per-row exact / greedy flags."""
        if reads is None:
            return None
        M = bk.M
        a, r = (ttnn.to_torch(ttnn.get_device_tensors(h)[0]).reshape(-1)[:M] for h in reads)
        return {"accept": [float(v) for v in a.tolist()], "resid": [int(v) for v in r.tolist()],
                "exact": list(bk.spec_exact), "greedy": list(bk.spec_greedy)}

    def _step_done(self, plan, acc, ids, drafts, W):
        first, extras = self._emit(plan, acc, W)
        self._log_step(plan, acc, ids, drafts)
        return first, extras

    def _emit(self, plan, acc, W):
        """The step's output: row b emits first[b] then extras[b] (acc.emitted). Needs no drafts."""
        first = torch.zeros(W, dtype=torch.int32)
        extras = [[] for _ in range(W)]
        for b in range(plan.bw):
            if acc.emitted[b]:
                first[b] = acc.emitted[b][0]
                extras[b] = acc.emitted[b][1:]
        self._steps += 1
        self._last_step_t = time.time()
        return first, extras

    def _log_step(self, plan, acc, ids, drafts):
        """Trace lines and the periodic stats line, once the step's drafts are stored (host.finish)."""
        if self.host.trace:
            for line in self.host.trace_step(self._steps, plan, acc, ids, drafts):
                logger.info(line)
        if self._steps == 1 or (self._stats_every > 0 and self._steps % self._stats_every == 0):
            self.log_stats("stats")

    def _step_overlap(self, bk, bw, W, toks, posl, cap, pt, t_entry):
        """QWEN36_MTP_OVERLAP: maybe_step's step with the commit/draft trace and the draft readback left in
        flight. Same inputs, same ops; only the host no longer waits for C before returning. drain()
        (next entry, prefill, release, remap) finishes the drafts; under so1 C-split the GDN commit is
        waited just before this verify so between-step host work can hide it."""
        clk = time.perf_counter
        self._ov_close(t_entry)
        plan = self.host.plan(toks, posl, bw, bk.K, cap, **getattr(self, "_plan_kw", {}))
        early = self._early_reseed
        pre = _PlanOnly(bk, plan if early else None)
        first_inputs = ("v_", "rs_") if early else ("v_",)
        split = spec_sample.host_split()  # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: build only what each write sends
        vals = bk.host_inputs(plan=plan, acc=pre, pt=pt[:bw],
                              want=(lambda n: n.startswith(first_inputs)) if split else None)
        self._write_inputs(bk, {n: v for n, v in vals.items() if n.startswith(first_inputs)})
        # C-split: previous step's deferred commit must land before this verify reads GDN state.
        # Counted in prep (host plan + input write + residual commit wait).
        self._ensure_committed()
        t_p = clk()
        ttnn.execute_trace(self.mesh, bk.v_trace, cq_id=0, blocking=False)
        sreads = self._spec_reads(bk)
        if early:
            ids_host = bk.v_out[0].cpu(blocking=False)
            ev_v = ttnn.record_event(self.mesh, 0)
            ttnn.execute_trace(self.mesh, bk.r_trace, cq_id=0, blocking=False)
            ttnn.event_synchronize(ev_v)
            ids = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(ids_host)[0]).reshape(-1)[: bk.M].tolist()]
        else:
            ids = _read_ids(bk.v_out[0], bk.M)
        t_v = clk()
        acc = self.host.accept(plan, ids, self._spec_data(bk, sreads))
        vals = bk.host_inputs(plan=plan, acc=acc, pt=pt[:bw],
                              want=(lambda n: not n.startswith(first_inputs)) if split else None)
        self._write_inputs(bk, {n: v for n, v in vals.items() if not n.startswith(first_inputs)})
        commit_event = None
        if self._c_split:
            # QWEN36_MTP_SO1_CSPLIT: cd first (draft steps; under EARLY_RESEED the reseed already ran in R, else cd
            # also reseeds), so the draft event (wait_c) does not include the so1 commit recompute; cc follows on
            # the CQ and is waited before the next verify (_ensure_committed).
            ttnn.execute_trace(self.mesh, bk.cd_trace, cq_id=0, blocking=False)
            reads = [t.cpu(blocking=False) for t in bk.c_out]
            event = ttnn.record_event(self.mesh, 0)
            ttnn.execute_trace(self.mesh, bk.cc_trace, cq_id=0, blocking=False)
            commit_event = ttnn.record_event(self.mesh, 0)
        elif early:
            ttnn.execute_trace(self.mesh, bk.c2_trace, cq_id=0, blocking=False)
            reads = [t.cpu(blocking=False) for t in bk.c_out]
            event = ttnn.record_event(self.mesh, 0)
        else:
            ttnn.execute_trace(self.mesh, bk.c_trace, cq_id=0, blocking=False)
            reads = [t.cpu(blocking=False) for t in bk.c_out]
            event = ttnn.record_event(self.mesh, 0)
        self.host.defer(plan, acc)
        self._pending = _Pending(plan, acc, ids, bw, reads, event, commit_event=commit_event)
        out = self._emit(plan, acc, W)
        t_ret = clk()
        if self._ov_timer is not None:
            ms = 1000.0
            self._ov_part = {"bw": bw, "t_entry": t_entry, "t_ret": t_ret, "at": time.time(),
                             "counts": mbh.step_counts(plan, acc),
                             "prep": (t_p - t_entry) * ms, "verify": (t_v - t_p) * ms, "accept": (t_ret - t_v) * ms}
        return out

    def _step_devaccept(self, bk, bw, W, toks, posl, cap, pt, t_entry):
        """QWEN36_MTP_DEVACCEPT: _step_overlap with the accept on device. Every input is written BEFORE V (the verify
        inputs, the plan-only reseed positions, the da_plan page and, under the served rule, the m-independent chain
        parameters and the chain noise for every m); V, the verify-output reads, R and the commit/draft traces are then
        enqueued back to back, the draft trace's first program computing gslot / anchor / t0n / draft positions / chain
        noise from the verify outputs. Only then does the host wait for the verify outputs (an event after V; the device
        is already running R and the drafts) and run accept() for the bookkeeping: emitted tokens, the deferred slot
        update. drain() reads the drafts and the m page and checks it against the host's m and t0n."""
        clk = time.perf_counter
        if self._pending is not None:  # QWEN36_MTP_DEVFEED: the previous step is still in flight
            pred = self._feed_try(bw, bk.K, toks, posl, cap)
            if pred is not None:
                return self._step_feed(bk, bw, W, toks, posl, cap, pt, t_entry, pred)
            self.drain(steady=True)
        self._ov_close(t_entry)
        t_0 = clk()
        plan = self.host.plan(toks, posl, bw, bk.K, cap, **getattr(self, "_plan_kw", {}))
        t_pl = clk()
        first = ("v_", "rs_")
        split = spec_sample.host_split()  # QWEN36_MTP_SPEC_SAMPLE_HOSTSPLIT: build only what this write sends
        vals = bk.host_inputs(plan=plan, acc=_PlanOnly(bk, plan), pt=pt[:bw],
                              want=(lambda n: n.startswith(first)) if split else None)
        vals = {n: v for n, v in vals.items() if n.startswith(first)}
        exact = list(bk.spec_exact) if _serve() else None  # set by the v_ build just above
        page = mbh.da_plan_page(plan, self.host.reject_all, self.host.mustfail, self._da_mf, exact=exact)
        vals.update(bk.da_inputs(plan, page))
        t_in = clk()
        self._write_inputs(bk, vals)
        t_wr = clk()
        # C-split: the previous step's deferred commit must land before this verify reads GDN state.
        self._ensure_committed()
        t_p = clk()
        ttnn.execute_trace(self.mesh, bk.v_trace, cq_id=0, blocking=False)
        sreads = self._spec_reads(bk)
        ids_host = bk.v_out[0].cpu(blocking=False)
        ev_v = ttnn.record_event(self.mesh, 0)
        if self._early_reseed:
            ttnn.execute_trace(self.mesh, bk.r_trace, cq_id=0, blocking=False)
        commit_event = None
        if self._c_split:
            ttnn.execute_trace(self.mesh, bk.cd_trace, cq_id=0, blocking=False)
            reads = [t.cpu(blocking=False) for t in bk.c_out]
            m_read = bk.da_m.cpu(blocking=False)
            event = ttnn.record_event(self.mesh, 0)
            ttnn.execute_trace(self.mesh, bk.cc_trace, cq_id=0, blocking=False)
            commit_event = ttnn.record_event(self.mesh, 0)
        else:
            ttnn.execute_trace(self.mesh, bk.c2_trace if self._early_reseed else bk.c_trace, cq_id=0, blocking=False)
            reads = [t.cpu(blocking=False) for t in bk.c_out]
            m_read = bk.da_m.cpu(blocking=False)
            event = ttnn.record_event(self.mesh, 0)
        ttnn.event_synchronize(ev_v)
        ids = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(ids_host)[0]).reshape(-1)[: bk.M].tolist()]
        t_v = clk()
        acc = self.host.accept(plan, ids, self._spec_data(bk, sreads))
        self.host.defer(plan, acc)
        self._pending = _Pending(plan, acc, ids, bw, reads, event, commit_event=commit_event, da_m=m_read)
        out = self._emit(plan, acc, W)
        t_ret = clk()
        if self._ov_timer is not None:
            ms = 1000.0
            self._ov_part = {"bw": bw, "t_entry": t_entry, "t_ret": t_ret, "at": time.time(),
                             "counts": mbh.step_counts(plan, acc),
                             "prep": (t_p - t_entry) * ms, "verify": (t_v - t_p) * ms, "accept": (t_ret - t_v) * ms,
                             "p_plan": (t_pl - t_0) * ms, "p_in": (t_in - t_pl) * ms, "p_wr": (t_wr - t_in) * ms,
                             "p_cc": (t_p - t_wr) * ms}
        return out

    def _feed_try(self, bw, K, toks, posl, cap):
        """QWEN36_MTP_DEVFEED: this step's predicted plan if it is a fast step (counted per bucket), else None."""
        st = self._feed_stats
        if _FAM["cur"] != "s":
            pred, why = None, "family"
        else:
            pred, why = self.host.feed_predict(toks, posl, bw, K, cap, pad=self._feed_pad,
                                               **getattr(self, "_plan_kw", {}))
        bb = st["by_bw"].setdefault(bw, [0, 0])
        bb[1] += 1
        if pred is not None:
            st["fast"] += 1
            bb[0] += 1
            if not all(pred.active):
                st["pad_fast"] += 1
        else:
            st["slow"] += 1
            st["miss"][why] = st["miss"].get(why, 0) + 1
        return pred

    def _step_feed(self, bk, bw, W, toks, posl, cap, pt, t_entry, pred):
        """QWEN36_MTP_DEVFEED fast step: the previous step's draft trace already wrote this verify's draft-dependent
        inputs (its last program, mtp_accept.feed). The host writes every other pre-verify input from the PREDICTED plan
        (feed_predict: the real plan except the drafts and their q blocks) and enqueues V, R, cd, cc at once: the CQ
        orders them after the previous cd (and its feed) and cc, so the host waits for neither. Only then does it drain
        the previous step (draft readback, m-page check, host.complete), build the real plan and require it to equal
        the prediction in every draft-independent field (fail closed: the verify already ran). The first
        QWEN36_MTP_DEVFEED_CHECK fast steps also read the fed buffers back (every chip) and compare them bitwise with the
        host's build from the real plan; a corrupted copy of the host build must then differ (the wrong-input branch)."""
        clk = time.perf_counter
        self._ov_close(t_entry)
        t_0 = clk()
        first = ("v_", "rs_")
        split = spec_sample.host_split()
        vals = bk.host_inputs(plan=pred, acc=_PlanOnly(bk, pred), pt=pt[:bw],
                              want=(lambda n: n.startswith(first)) if split else None)
        vals = {n: v for n, v in vals.items() if n.startswith(first) and n not in mbh.FEED_NAMES}
        exact = list(bk.spec_exact)
        page = mbh.da_plan_page(pred, self.host.reject_all, self.host.mustfail, self._da_mf, exact=exact)
        vals.update(bk.da_inputs(pred, page))
        t_in = clk()
        self._write_inputs(bk, vals)
        t_wr = clk()
        padded = not all(pred.active)
        check = (self._feed_stats["pad_checked"] if padded else self._feed_stats["checked"]) < self._feed_check
        fed = [bk.dev[n].cpu(blocking=False) for n in mbh.FEED_NAMES] if check else None
        t_p = clk()
        ttnn.execute_trace(self.mesh, bk.v_trace, cq_id=0, blocking=False)
        sreads = self._spec_reads(bk)
        ids_host = bk.v_out[0].cpu(blocking=False)
        ev_v = ttnn.record_event(self.mesh, 0)
        ttnn.execute_trace(self.mesh, bk.r_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh, bk.cd_trace, cq_id=0, blocking=False)
        reads = [t.cpu(blocking=False) for t in bk.c_out]
        m_read = bk.da_m.cpu(blocking=False)
        event = ttnn.record_event(self.mesh, 0)
        ttnn.execute_trace(self.mesh, bk.cc_trace, cq_id=0, blocking=False)
        commit_event = ttnn.record_event(self.mesh, 0)
        t_q = clk()
        # the previous step: its drafts (already consumed on device), its m-page check, its slot state
        sub0 = dict(self._ov_sub)
        self.drain(steady=True)
        self._commit_event = None  # its cc precedes this step's V on the CQ
        self._ov_wait_ms = 0.0  # the drain ran inside this step (counted in f_drain below, not as wait_c)
        t_d = clk()
        before = ({k: self.host.stats[k] for k in mbh.FEED_STATS}, self.host.dual_qdrop)
        plan = self.host.plan(toks, posl, bw, bk.K, cap, **getattr(self, "_plan_kw", {}))
        after = ({k: self.host.stats[k] for k in mbh.FEED_STATS}, self.host.dual_qdrop)
        bad = mbh.feed_plan_diff(pred, plan, bw) + (["counters"] if after != before else [])
        if bad:
            raise RuntimeError(f"{mbh.ENV_DEVFEED}: the real plan disagrees with the fast-path prediction in {bad} at "
                               f"Bw={bw} (step {self._steps}; counters {before} -> {after}); the verify already ran on "
                               f"the predicted inputs: refusing to go on")
        t_pl = clk()
        ttnn.event_synchronize(ev_v)
        ids = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(ids_host)[0]).reshape(-1)[: bk.M].tolist()]
        if check:
            self._feed_selfcheck(bk, plan, exact, fed)
        t_v = clk()
        acc = self.host.accept(plan, ids, self._spec_data(bk, sreads))
        self.host.defer(plan, acc)
        self._pending = _Pending(plan, acc, ids, bw, reads, event, commit_event=commit_event, da_m=m_read)
        out = self._emit(plan, acc, W)
        t_ret = clk()
        if self._ov_timer is not None:
            ms = 1000.0
            sub = {k: v - sub0.get(k, 0.0) for k, v in self._ov_sub.items()}
            self._ov_part = {"bw": bw, "t_entry": t_entry, "t_ret": t_ret, "at": time.time(),
                             "counts": mbh.step_counts(plan, acc),
                             "prep": (t_p - t_entry) * ms, "verify": (t_v - t_p) * ms, "accept": (t_ret - t_v) * ms,
                             "f_pred": (t_0 - t_entry) * ms, "f_in": (t_in - t_0) * ms, "f_wr": (t_wr - t_in) * ms,
                             "f_enq": (t_q - t_p) * ms, "f_drain": (t_d - t_q) * ms, "f_plan": (t_pl - t_d) * ms,
                             "f_fast": 1.0, "f_sub": sub}
        return out

    def _feed_selfcheck(self, bk, plan, exact, fed):
        """QWEN36_MTP_DEVFEED in-boot step self-test: the fed buffers (read back before this step's draft trace) on
        every chip vs the host's build from the real plan (host_inputs' v_tok, mbh.spec_v_blocks), bitwise; then the
        host build with draft 0 of the first active row + 1 must differ (the wrong-input branch). Fails closed.
        QWEN36_MTP_DEVFEED_PAD: a step with padded rows is counted separately and compares the ACTIVE rows only (a padded
        row's fed values are don't-care: nothing reads its verify outputs); PAD_MUSTFAIL compares every row."""
        from models.demos.blackhole.qwen36.tt import mtp_accept

        bw, K, M = bk.bw, bk.K, bk.M
        C2 = 2 * spec_sample.k()
        padded = not all(plan.active)
        rows = None
        if padded and not self._feed_pad_mf:
            tok, q = mbh.feed_active_rows(plan, M, K)
            rows = {"v_tok": tok, "v_sqp": q, "v_sqi": q, "v_sx": q}
        b0 = next(b for b in range(bw) if plan.active[b])
        qp, qi, xs, _ = mbh.spec_v_blocks(plan, bw, K, M, C2, exact)
        exp = {"v_tok": torch.tensor(plan.verify_tokens, dtype=torch.int32).reshape(M, 1), "v_sqp": qp, "v_sqi": qi,
               "v_sx": xs}
        wrong = {n: v.clone() for n, v in exp.items()}
        wrong["v_tok"][bw + b0, 0] += 1
        wrong["v_sx"][b0, 0] = float(int(plan.drafts[b0][0]) + 1)
        bad, blind = {}, []
        chips = [[ttnn.to_torch(x) for x in ttnn.get_device_tensors(h)] for h in fed]
        for d in range(len(chips[0])):
            got = {n: chips[i][d] for i, n in enumerate(mbh.FEED_NAMES)}
            for n, v in mtp_accept.feed_diff(got, exp, rows).items():
                bad[f"{n}@{d}"] = v
            if not mtp_accept.feed_diff(got, wrong, rows):
                blind.append(d)
        key = "pad_checked" if padded else "checked"
        self._feed_stats[key] += 1
        n = self._feed_stats[key]
        what = "padded fast step" if padded else "fast step"
        if bad or blind:
            env = mbh.ENV_DEVFEED_PAD if padded else mbh.ENV_DEVFEED
            raise RuntimeError(f"{env} step self-test REFUSED at Bw={bw} ({what} {n}, active {plan.active}): device-fed "
                               f"inputs differ from the host build {bad}; wrong-input branch blind on chips {blind}")
        if n == 1 or n == self._feed_check:
            if padded:
                logger.info(f"QWEN36_MTP_DEVFEED_PAD step self-test exact: {n} padded fast step(s) checked (Bw={bw}, "
                            f"{sum(plan.active)} active), the fed v_tok / v_sqp / v_sqi / v_sx of the active rows bitwise "
                            f"vs the host build on every chip; the corrupted-draft branch differs")
            else:
                logger.info(f"QWEN36_MTP_DEVFEED step self-test exact: {n} fast step(s) checked, the fed v_tok / v_sqp / "
                            f"v_sqi / v_sx bitwise vs the host build on every chip; the corrupted-draft branch differs")

    def _ov_close(self, t_entry):
        """OVERLAP_TIMING: close the previous step's sample at this steady entry. host = the time from its
        return to this entry minus the draft-event wait (wait_c, the part of commit + draft the host work
        did not hide); total = entry to entry."""
        part, self._ov_part = self._ov_part, None
        wait, self._ov_wait_ms = self._ov_wait_ms, 0.0
        sub, self._ov_sub = getattr(self, "_ov_sub", {}), {}
        if part is None or self._ov_timer is None:
            return
        ms = 1000.0
        gap = (t_entry - part["t_ret"]) * ms
        t = {"host": max(0.0, gap - wait), "wait_c": wait, "prep": part["prep"], "verify": part["verify"],
             "accept": part["accept"], "total": (t_entry - part["t_entry"]) * ms}
        t.update({k: part[k] for k in ("p_plan", "p_in", "p_wr", "p_cc") if k in part})
        # QWEN36_MTP_DEVFEED: a fast step's own split, and the drain it ran inside (its d_* sub-phases)
        t.update({k: part[k] for k in mbh.FEED_OV_PHASES if k in part})
        t.update(part.get("f_sub") if "f_sub" in part else sub)
        if self._ov_timer.add(part["bw"], t, at=part["at"], counts=part["counts"]):
            for line in self._ov_timer.lines(fallbacks=dict(self._fallbacks)):
                logger.info(line)

    def _step_timed(self, bk, bw, W, toks, posl, cap, pt):
        """QWEN36_MTP_BATCH_TIMING: the same step as maybe_step, phase by phase, each device phase
        closed by ttnn.synchronize_device so its wall time is its own. The phases are the ones the
        C++ lever choice needs: prep (host plan + verify input write), verify forward, verify-id readback, host accept (+ C input write),
        GDN commit where-chain, MTP reseed + K draft steps, draft-id readback."""
        clk = time.perf_counter
        sync = ttnn.synchronize_device
        t = {}
        t0 = clk()
        plan = self.host.plan(toks, posl, bw, bk.K, cap, **getattr(self, "_plan_kw", {}))
        vals = bk.host_inputs(plan=plan, acc=_PlanOnly(bk), pt=pt[:bw])
        self._write_inputs(bk, {n: v for n, v in vals.items() if n.startswith("v_")})
        sync(self.mesh)
        tp = clk()
        ttnn.execute_trace(self.mesh, bk.v_trace, cq_id=0, blocking=False)
        sync(self.mesh)
        t1 = clk()
        ids = _read_ids(bk.v_out[0], bk.M)
        t2 = clk()
        acc = self.host.accept(plan, ids)
        vals = bk.host_inputs(plan=plan, acc=acc, pt=pt[:bw])
        self._write_inputs(bk, {n: v for n, v in vals.items() if not n.startswith("v_")})
        sync(self.mesh)
        t3 = clk()
        ttnn.execute_trace(self.mesh, bk.cc_trace, cq_id=0, blocking=False)
        sync(self.mesh)
        t4 = clk()
        ttnn.execute_trace(self.mesh, bk.cd_trace, cq_id=0, blocking=False)
        sync(self.mesh)
        t5 = clk()
        drafts = []
        for x in bk.c_out:
            drafts.extend(_read_ids(x, bw))
        t6 = clk()
        self.host.finish(plan, acc, drafts)
        ms = 1000.0
        t.update(prep=(tp - t0) * ms, verify=(t1 - tp) * ms, rb_verify=(t2 - t1) * ms, accept=(t3 - t2) * ms, commit=(t4 - t3) * ms,
                 draft=(t5 - t4) * ms, rb_draft=(t6 - t5) * ms, total=(t6 - t0) * ms)
        if self._timer.add(bw, t, at=time.time(), counts=mbh.step_counts(plan, acc)):
            for line in self._timer.lines(fallbacks=dict(self._fallbacks)):
                logger.info(line)
        return self._step_done(plan, acc, ids, drafts, W)


# ---------------------------------------------------------------------------------------- QWEN36_MTP_OPTRACE
ENV_OPTRACE = "QWEN36_MTP_OPTRACE"
OPTRACE_PHASES = (("v_trace", "V"), ("r_trace", "R"), ("cd_trace", "cd"), ("cc_trace", "cc"), ("c2_trace", "C2"),
                  ("c_trace", "C"))


def optrace_enabled():
    v = _tt_public_getenv(ENV_OPTRACE, "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{ENV_OPTRACE}={v!r}: expected unset, 0 or 1")
    if v == "1" and _tt_public_getenv("QWEN36_OPMARK", "").strip() != "1":
        raise RuntimeError(f"{ENV_OPTRACE}=1 needs QWEN36_OPMARK=1 (the op names of the captured programs)")
    return v == "1"


class _OpTrace:
    """QWEN36_MTP_OPTRACE=1 (diagnostic, host-side only; MTP-BATCH section 29): a NAMED device-time table of the served
    MTP step, per (bucket, family, context) window.

    Capture: the MTP trace captures run inside an opmark window with a scope tag per trace (V, R, cd, cc / C2 / C), per
    draft depth (cd/d0..d{K-1}), and for the verify's embed / head, so every captured program has an op name and a
    scope (layer kind + sublayer from opmark's layer scopes). Serving: after SKIP consecutive full-bucket steps (active
    rows == Bw) at one (Bw, family, ctx), the device is synchronised, the realtime profiler callback is registered and
    an opmark window opened; STEPS served steps later the device is synchronised again, the records are aligned to the
    window's replay stream (opmark.expected_stream/align, the c177 method) and ONE line is logged:
    "QWEN36_MTP_OPTRACE window bw=.. fam=.. ctx=..: {json}". Device programs, their inputs and the traces are the
    served ones (opmark adds host-side id reads only); the two synchronisations and the profiler callback cost host
    time, so the window's steps (and the cell's wall clock) are diagnostic-only, never a speed number."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.steps = int(_tt_public_getenv(ENV_OPTRACE + "_STEPS", "8"))
        self.skip = int(_tt_public_getenv(ENV_OPTRACE + "_SKIP", "16"))
        self.min_pos = int(_tt_public_getenv(ENV_OPTRACE + "_MIN_POS", "2048"))
        self.max_windows = int(_tt_public_getenv(ENV_OPTRACE + "_MAX", "24"))
        self.phase_of = {}
        self.cap_rows, self.captures, self.owner_cap = [], {}, None
        self.done, self.run_key, self.run_n = set(), None, 0
        self.open_key, self.n_in, self.t_open, self.rt, self.windows = None, 0, 0.0, None, 0
        logger.info(f"QWEN36_MTP_OPTRACE engaged: named device-time window of {self.steps} served steps per (Bw, family, "
                    f"ctx) after {self.skip} full-bucket steps, pos >= {self.min_pos}, <= {self.max_windows} windows; "
                    f"host-side only (opmark ids + realtime profiler), device programs unchanged; window steps are "
                    f"diagnostic-only")

    # ---- capture
    def capture_begin(self):
        from models.demos.blackhole.qwen36.tt import opmark

        ok, why = opmark.RTWindow.available()
        if not ok:
            raise RuntimeError(f"{ENV_OPTRACE}=1 REFUSED: realtime profiler unavailable ({why})")
        if not opmark.engaged():
            raise RuntimeError(f"{ENV_OPTRACE}=1 REFUSED: QWEN36_OPMARK is not engaged (install() did not run)")
        opmark.begin()

    def note_family(self, bk):
        from models.demos.blackhole.qwen36.tt import opmark

        for attr, ph in OPTRACE_PHASES:
            tid = getattr(bk, attr, None)
            if tid is not None:
                self.phase_of[opmark._trace_key(None, tid)] = (bk.bw, _FAM["cur"], ph)

    def capture_end(self):
        from models.demos.blackhole.qwen36.tt import opmark

        res = opmark.end()
        self.cap_rows, self.captures = res["rows"], res["captures"]
        named = sum(1 for r in self.cap_rows if r[2] not in ("<scope>", "<trace-replay>"))
        logger.info(f"QWEN36_MTP_OPTRACE capture rows: {len(self.cap_rows)} ({named} named ops), {len(self.phase_of)} MTP "
                    f"traces {sorted(set(v[2] for v in self.phase_of.values()))}, dropped {res['dropped']}")

    # ---- serving
    def pre(self, bw, fam, posl, num_active):
        live = [int(p) for p in posl if int(p) >= 0]
        ctx = max(live) if live else 0
        cb = 1 << max(0, int(round(math.log2(max(ctx, 1)))))
        key = (int(bw), fam, cb)
        if self.open_key is not None:
            if key != self.open_key or num_active != bw:
                self._close(aborted=f"step at {key} with {num_active} active rows")
            return
        if num_active != bw or ctx < self.min_pos or key in self.done or self.windows >= self.max_windows:
            self.run_key, self.run_n = None, 0
            return
        if key != self.run_key:
            self.run_key, self.run_n = key, 0
        self.run_n += 1
        if self.run_n <= self.skip:
            return
        from models.demos.blackhole.qwen36.tt import opmark

        ttnn.synchronize_device(self.mesh)
        time.sleep(0.3)  # records of the steps before the window are delivered to no callback
        self.rt = opmark.RTWindow()
        self.rt.open()
        opmark.begin()
        self.open_key, self.n_in, self.ctx_seen = key, 0, []
        self.t_open = time.perf_counter()

    def post(self, posl):
        if self.open_key is None:
            return
        self.n_in += 1
        self.ctx_seen.append(max([int(p) for p in posl if int(p) >= 0] or [0]))
        if self.n_in >= self.steps:
            self._close()

    def _close(self, aborted=None):
        from models.demos.blackhole.qwen36.tt import opmark

        ttnn.synchronize_device(self.mesh)
        wall_ms = (time.perf_counter() - self.t_open) * 1000.0
        recs, dropped = self.rt.close()
        win = opmark.end()
        key, n = self.open_key, self.n_in
        self.open_key, self.rt, self.run_key, self.run_n = None, None, None, 0
        if aborted is not None and n < 2:
            logger.info(f"{ENV_OPTRACE} window at {key} dropped after {n} step(s): {aborted}")
            return
        self.done.add(key)
        self.windows += 1
        try:
            out = self.analyse(win["rows"], win["captures"], recs, dropped, n)
        except Exception as e:  # noqa: BLE001  a diagnostic never takes the server down
            logger.warning(f"{ENV_OPTRACE} window at {key}: analysis failed: {e!r}")
            return
        out.update({"bw": key[0], "fam": key[1], "ctx_bin": key[2], "ctx_min": min(self.ctx_seen),
                    "ctx_max": max(self.ctx_seen), "steps": n, "window_wall_ms": round(wall_ms, 3),
                    "aborted": aborted})
        logger.info(f"QWEN36_MTP_OPTRACE window bw={key[0]} fam={key[1]} ctx={key[2]}: {json.dumps(out, separators=(',', ':'))}")

    def analyse(self, win_rows, captures, recs, dropped, nsteps):
        from models.demos.blackhole.qwen36.tt import opmark

        if self.owner_cap is None:
            self.owner_cap = opmark.owner_index(self.cap_rows)
        return opmark.analyse_window(win_rows, captures, self.cap_rows, self.owner_cap, self.phase_of, recs, dropped,
                                     nsteps)


class _Pending:
    """QWEN36_MTP_OVERLAP: a speculative step whose tokens are out and whose drafts are still in flight.

    commit_event: OVERLAP+so1 C-split only — GDN commit still in flight after the draft event; drain
    under steady=True defers it to _ensure_committed before the next verify."""

    __slots__ = ("plan", "acc", "ids", "bw", "reads", "event", "commit_event", "da_m")

    def __init__(self, plan, acc, ids, bw, reads, event, commit_event=None, da_m=None):
        self.plan, self.acc, self.ids, self.bw, self.reads, self.event = plan, acc, ids, bw, reads, event
        self.commit_event = commit_event
        self.da_m = da_m  # QWEN36_MTP_DEVACCEPT: the host copy (in flight) of the device's m / t0n page


class _PlanOnly:
    """Placeholder Accept for the verify-input write (only v_* inputs are taken from it). With a plan
    (QWEN36_MTP_EARLY_RESEED) the reseed positions are the real ones: they depend on the plan alone."""

    def __init__(self, bk, plan=None):
        T, K, bw = bk.T, bk.K, bk.bw
        self.gdn_onehot = [[1.0 if j == 0 else 0.0] * bw for j in range(T)]
        self.anchor_idx = list(range(bw))
        self.t0_next = [0] * bw
        if plan is not None:
            self.reseed_pos, self.reseed_rope_pos = mbh.MTPBatchHost.reseed_positions(plan)
        else:
            self.reseed_pos = [[-1] * bw for _ in range(T)]
            self.reseed_rope_pos = [-1] * (T * bw)
        self.draft_pos = [[-1] * bw for _ in range(K)]
        self.draft_rope_pos = [[-1] * bw for _ in range(K)]
