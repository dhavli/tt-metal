# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MTP_DEVACCEPT=1: the MTP-at-batch accept on device (docs/MTP-BATCH.md section 25).

Today's step reads the verify outputs back to the host, runs MTPBatchHost.accept(), builds and writes the inputs of
the commit/draft traces that depend on the accepted lengths m_b (the slot-index commit page gslot, the draft anchor
rows, t0n, the K draft KV / RoPE positions and, under the served exact rule, the K chain-noise blocks d_sgum{k}) and
only then enqueues them; the device idles for that round trip. Here ONE ttnn.generic_op program (a single-core JIT
dataflow kernel, kernels/mtp_accept/accept.cpp), captured as the first program of the draft trace, computes those
inputs from
  ids     the verify's ids, ROW_MAJOR uint32 [M, 1]  (bk.v_out[0])
  v_tok   the verify tokens, ROW_MAJOR uint32 [M, 1]; rows j >= 1 are this step's drafts (plan.drafts)
  plan    one ROW_MAJOR int32 [1, 256] page written by the host BEFORE V (mtp_batch_host.da_plan_page)
  accept, resid   (served exact rule) the rule's outputs, fp32 / uint32 [M, 1] (bk.v_out[3], bk.v_out[4])
  tab[k]  (served exact rule) the chain noise of depth k for every possible m, fp32 ROW_MAJOR [1, 1, (K+1)*bw, C2]
and writes them in place into the bucket's persistent inputs, plus an m page [1, 64] (m_b, then t0n_b) the host checks
against its own accept() at the next drain. mtp_batch_host.da_reference is the kernel line for line; da_expected is what
the host path writes.

Every address is a runtime arg (the ids / flags are the V trace's own outputs, every other buffer is a persistent input
allocated before capture), so the trace replays with new values. Integer compares and copies only: the outputs are exact
by construction, and the boot self-test (selftest below, run by Qwen36MTPBatch._selftest_devaccept before any capture)
checks them bitwise against the host accept on synthetic cases, on both chips.

c152's QWEN36_MTP_DEVICE_ACCEPT (branch mtp-device-accept, greedy only) is this program's ancestor.
"""

import os
from pathlib import Path

import ttnn

try:
    from models.demos.blackhole.qwen36.tt import mtp_batch_host as _mbh
except ImportError:  # loaded by path (ttsim probes): the sibling module
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("mtp_batch_host", Path(__file__).resolve().parent / "mtp_batch_host.py")
    _mbh = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mbh)

ENV = _mbh.ENV_DEVACCEPT
PLAN_LEN = _mbh.DA_PAGE_LEN
IDX_LEN = 32
M_LEN = _mbh.DA_M_LEN
_KSRC = Path(__file__).resolve().parent / "kernels" / "mtp_accept" / "accept.cpp"
_SRC = {}
_CB_IN, _CB_OUT = 0, 1
_SLOT, _OSLOT = 64, 256


def enabled():
    return _mbh.devaccept()


def _src():
    if "k" not in _SRC:
        _SRC["k"] = _KSRC.read_text()
    return _SRC["k"]


def _shape(t):
    return [int(t.shape[i]) for i in range(len(t.shape))]


def _page(t):
    try:
        return int(t.buffer_aligned_page_size())
    except Exception as e:  # the kernel's transfers are whole aligned pages; never guess the alignment
        raise RuntimeError(f"{ENV}: this ttnn has no Tensor.buffer_aligned_page_size ({e})") from e


def _interleaved(t, what):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{ENV}: {what} must be interleaved, got {t.memory_config()}")


def _rm(t, what, dtype, shape):
    if t.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"{ENV}: {what} must be ROW_MAJOR, got {t.layout}")
    if t.dtype not in dtype:
        raise RuntimeError(f"{ENV}: {what} dtype {t.dtype} not in {dtype}")
    if _shape(t) != list(shape):
        raise RuntimeError(f"{ENV}: {what} shape {list(t.shape)} != {list(shape)}")
    _interleaved(t, what)


def _flag_tensor(t, what, dtype, M):
    """The rule's [M, 1] output: TILE (one 32x32 32-bit tile) or ROW_MAJOR (M pages). Returns A_TILE."""
    if t.dtype != dtype:
        raise RuntimeError(f"{ENV}: {what} dtype {t.dtype} != {dtype}")
    if _shape(t)[-2:] != [M, 1] or any(s != 1 for s in _shape(t)[:-2]):
        raise RuntimeError(f"{ENV}: {what} shape {list(t.shape)} != [.., {M}, 1]")
    _interleaved(t, what)
    if t.layout == ttnn.TILE_LAYOUT:
        if _page(t) != 4096:
            raise RuntimeError(f"{ENV}: {what} TILE page {_page(t)} != 4096 (a 32x32 32-bit tile)")
        return 1
    if t.layout == ttnn.ROW_MAJOR_LAYOUT:
        return 0
    raise RuntimeError(f"{ENV}: {what} layout {t.layout}")


def run(ids, vtok, plan, gslot, anchor, t0n, d_pos, d_rpos, m_out, bw, T, accept=None, resid=None, tabs=None,
        nz=None):
    """One device accept for a bucket of width bw and T = K+1 verify rows per user. Writes gslot [1,32] u32,
    anchor [1,bw] u32, t0n [bw,1] u32, d_pos[k] / d_rpos[k] [bw] i32 (k < K), m_out [1,64] u32 and (served exact rule:
    accept / resid / tabs / nz given) nz[k] [1,1,bw,C2] fp32 in place. Returns m_out."""
    K, M = T - 1, bw * T
    spec = accept is not None
    # QWEN36_MTP_SPEC_SAMPLE_SSP: the exact accept without the chain-noise copy (tabs = nz = None; the chain's fs_cut
    # makes its noise from the m page this program writes)
    spec_nz = tabs is not None
    if spec != (resid is not None) or (spec_nz and not spec) or spec_nz != (nz is not None):
        raise RuntimeError(f"{ENV}: accept and resid go together, and tabs / nz need them (the served exact rule)")
    if not (1 <= K <= 15 and M <= 32 and len(d_pos) == K and len(d_rpos) == K):
        raise RuntimeError(f"{ENV}: bw={bw} T={T}: needs 1 <= K <= 15, M <= 32 and K draft position tensors")
    u32, i32 = (ttnn.uint32,), (ttnn.int32, ttnn.uint32)
    _rm(ids, "ids", u32, (M, 1))
    _rm(vtok, "v_tok", u32, (M, 1))
    _rm(plan, "plan page", i32, (1, PLAN_LEN))
    _rm(gslot, "gslot", u32, (1, IDX_LEN))
    _rm(anchor, "anchor", u32, (1, bw))
    _rm(t0n, "t0n", u32, (bw, 1))
    _rm(m_out, "m page", u32, (1, M_LEN))
    for k in range(K):
        _rm(d_pos[k], f"d_pos{k}", i32, (bw,))
        _rm(d_rpos[k], f"d_rpos{k}", i32, (bw,))
    a_tile, p_acc, p_res, p_nz = 0, 0, 0, 0
    if spec:
        if spec_nz and (len(tabs) != K or len(nz) != K):
            raise RuntimeError(f"{ENV}: K={K} noise tables and blocks needed, got {len(tabs)} / {len(nz)}")
        ta = _flag_tensor(accept, "accept flags", ttnn.float32, M)
        tr = _flag_tensor(resid, "residual ids", ttnn.uint32, M)
        if ta != tr:
            raise RuntimeError(f"{ENV}: accept and resid layouts differ ({accept.layout} / {resid.layout})")
        a_tile, p_acc, p_res = ta, _page(accept), _page(resid)
    if spec_nz:
        C2 = _shape(nz[0])[-1]
        for k in range(K):
            _rm(tabs[k], f"noise table {k}", (ttnn.float32,), (1, 1, (K + 1) * bw, C2))
            _rm(nz[k], f"d_sgum{k}", (ttnn.float32,), (1, 1, bw, C2))
        p_nz = _page(nz[0])
        if any(_page(t) != p_nz for t in list(tabs) + list(nz)):
            raise RuntimeError(f"{ENV}: the noise tables and blocks must share one row page size")
    host = [vtok, plan, gslot, anchor, t0n] + list(d_pos) + list(d_rpos)
    if spec_nz:
        host += list(tabs) + list(nz)
    host.append(m_out)  # generic_op returns the last
    # One accessor class for every host-written buffer. Interleaved TensorAccessorArgs are {config, aligned page
    # size}; the kernel passes each tensor's own page size explicitly, so only the rest must agree.
    h_ct = list(ttnn.TensorAccessorArgs(plan).get_compile_time_args())

    def _cls(ct):
        return [ct[0]] + ct[2:] if len(ct) >= 2 else ct

    for t in host:
        if _cls(list(ttnn.TensorAccessorArgs(t).get_compile_time_args())) != _cls(h_ct):
            raise RuntimeError(f"{ENV}: the plan page, v_tok, the noise tables and every output must share one placement "
                               f"({t.memory_config()}: {list(ttnn.TensorAccessorArgs(t).get_compile_time_args())} vs {h_ct})")
    i_ct = list(ttnn.TensorAccessorArgs(ids).get_compile_time_args())
    a_ct = list(ttnn.TensorAccessorArgs(accept).get_compile_time_args()) if spec else list(i_ct)
    r_ct = list(ttnn.TensorAccessorArgs(resid).get_compile_time_args()) if spec else list(i_ct)
    p_ids, p_vtok, p_plan, p_gslot = _page(ids), _page(vtok), _page(plan), _page(gslot)
    p_anchor, p_t0n, p_dpos, p_m = _page(anchor), _page(t0n), _page(d_pos[0]), _page(m_out)
    if any(_page(t) != p_dpos for t in list(d_pos) + list(d_rpos)):
        raise RuntimeError(f"{ENV}: the draft position tensors must share one page size")
    if max(p_ids, p_vtok, p_t0n) > _SLOT or max(p_gslot, p_anchor, p_dpos, p_m) > _OSLOT or p_plan < 4 * PLAN_LEN \
            or p_plan % 64 or (spec and not a_tile and max(p_acc, p_res) > _SLOT) \
            or (spec_nz and (p_nz % 64 or p_nz > 4096 or p_nz == 0)):
        raise RuntimeError(f"{ENV}: page sizes ids={p_ids} v_tok={p_vtok} t0n={p_t0n} gslot={p_gslot} anchor={p_anchor} "
                           f"d_pos={p_dpos} m={p_m} plan={p_plan} acc={p_acc} res={p_res} nz={p_nz} outside the kernel's slots")
    a_bytes = (4096 if a_tile else M * _SLOT) if spec else 0
    in_bytes = p_plan + 2 * M * _SLOT + 2 * a_bytes + (K * bw * p_nz if spec_nz else 0)
    out_bytes = 3 * _OSLOT + 2 * K * _OSLOT + bw * _SLOT
    out_bytes = (out_bytes + _OSLOT - 1) // _OSLOT * _OSLOT
    core = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    ct = [bw, T, int(spec), a_tile, p_ids, p_vtok, p_plan, p_gslot, p_anchor, p_t0n, p_dpos, p_m, p_acc, p_res, p_nz,
          _CB_IN, _CB_OUT] + h_ct + i_ct + a_ct + r_ct
    rt = ttnn.RuntimeArgs()
    args = [ids.buffer_address(), vtok.buffer_address(), plan.buffer_address(), gslot.buffer_address(),
            anchor.buffer_address(), t0n.buffer_address(), m_out.buffer_address(),
            accept.buffer_address() if spec else 0, resid.buffer_address() if spec else 0]
    args += [t.buffer_address() for t in d_pos] + [t.buffer_address() for t in d_rpos]
    if spec_nz:
        args += [t.buffer_address() for t in tabs] + [t.buffer_address() for t in nz]
    rt[0][0] = args

    def cb(idx, size):
        return ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=size)])

    desc = ttnn.ProgramDescriptor(
        kernels=[ttnn.KernelDescriptor(kernel_source=_src(), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                       core_ranges=crs, compile_time_args=ct, runtime_args=rt,
                                       config=ttnn.ReaderConfigDescriptor())],
        semaphores=[],
        cbs=[cb(_CB_IN, in_bytes), cb(_CB_OUT, out_bytes)],
    )
    io = [ids] + ([accept, resid] if spec else []) + host
    return ttnn.generic_op(io, desc)


# ------------------------------------------------------------------------------------------------ self-test
_SENTINEL = 0x5A5A5A5A


class Buffers:
    """One bucket's device-accept tensors (inputs and outputs) on `device` (a MeshDevice: replicated), at the served
    dtypes, layouts and shapes. ids_mc / host_mc: the memory configs of the served ids (bk.v_out[0]) and of the
    persistent inputs, so the self-test compiles and runs the same kernel variant (same accessor args) as the trace.
    spec_c2 (served exact rule): the chain-noise width C2; the rule's flags are then TILE [M, 1] tensors (flag_layout)."""

    def __init__(self, device, bw, K, ids_mc=None, host_mc=None, spec_c2=None, flag_layout=None, flag_mc=None,
                 spec_nz=True):
        import torch

        self.device, self.bw, self.K, self.T, self.M = device, bw, K, K + 1, bw * (K + 1)
        self.mesh = hasattr(ttnn, "MeshDevice") and isinstance(device, ttnn.MeshDevice)
        self.rep = ttnn.ReplicateTensorToMesh(device) if self.mesh else None
        ids_mc = ids_mc or ttnn.DRAM_MEMORY_CONFIG
        host_mc = host_mc or ttnn.DRAM_MEMORY_CONFIG
        self.spec_c2 = spec_c2
        self.spec_nz = bool(spec_c2) and bool(spec_nz)  # QWEN36_MTP_SPEC_SAMPLE_SSP: False (no chain-noise copy)
        self.flag_layout = flag_layout or ttnn.TILE_LAYOUT
        self.flag_mc = flag_mc or ttnn.DRAM_MEMORY_CONFIG
        R = ttnn.ROW_MAJOR_LAYOUT
        self.spec = {"ids": ((self.M, 1), ttnn.uint32, ids_mc, R), "v_tok": ((self.M, 1), ttnn.uint32, host_mc, R),
                     "plan": ((1, PLAN_LEN), ttnn.int32, host_mc, R), "gslot": ((1, IDX_LEN), ttnn.uint32, host_mc, R),
                     "anchor": ((1, bw), ttnn.uint32, host_mc, R), "t0n": ((bw, 1), ttnn.uint32, host_mc, R),
                     "m": ((1, M_LEN), ttnn.uint32, host_mc, R)}
        for k in range(K):
            self.spec[f"d_pos{k}"] = ((bw,), ttnn.int32, host_mc, R)
            self.spec[f"d_rpos{k}"] = ((bw,), ttnn.int32, host_mc, R)
        if spec_c2:
            self.spec["accept"] = ((self.M, 1), ttnn.float32, self.flag_mc, self.flag_layout)
            self.spec["resid"] = ((self.M, 1), ttnn.uint32, self.flag_mc, self.flag_layout)
            for k in range(K if self.spec_nz else 0):
                self.spec[f"tab{k}"] = ((1, 1, (K + 1) * bw, spec_c2), ttnn.float32, host_mc, R)
                self.spec[f"nz{k}"] = ((1, 1, bw, spec_c2), ttnn.float32, host_mc, R)
        self.t = {n: ttnn.from_torch(self._host(n, None), dtype=dt, layout=lay, device=device, mesh_mapper=self.rep,
                                     memory_config=mc) for n, (shp, dt, mc, lay) in self.spec.items()}

    def _host(self, name, values):
        import torch

        shp, dt, _, _ = self.spec[name]
        n = _numel(shp)
        if dt == ttnn.float32:
            vals = [0.0] * n if values is None else [float(v) for v in values]
            return torch.tensor(vals, dtype=torch.float32).reshape(*shp)
        if values is None:
            values = [0] * n
        if dt == ttnn.uint32:
            return torch.tensor([int(v) & 0xFFFFFFFF for v in values], dtype=torch.int64).reshape(*shp)
        return torch.tensor([_mbh._i32(v) for v in values], dtype=torch.int32).reshape(*shp)

    def write(self, name, values):
        shp, dt, _, lay = self.spec[name]
        host = ttnn.from_torch(self._host(name, values), dtype=dt, layout=lay, mesh_mapper=self.rep)
        ttnn.copy_host_to_device_tensor(host, self.t[name])

    def write_bits(self, name, bits):
        """fp32 values given as raw bit patterns (NaN payloads, -0.0 survive)."""
        import struct

        self.write(name, [struct.unpack("<f", struct.pack("<I", int(b) & 0xFFFFFFFF))[0] for b in bits])

    def read(self, name, device_index=0):
        t = self.t[name]
        if self.mesh:
            t = ttnn.get_device_tensors(t)[device_index]
        return ttnn.to_torch(t).reshape(-1).tolist()

    def n_devices(self):
        return len(ttnn.get_device_tensors(self.t["m"])) if self.mesh else 1

    def sentinel(self):
        names = ["gslot", "anchor", "t0n", "m"] + [f"d_pos{k}" for k in range(self.K)] + [f"d_rpos{k}" for k in range(self.K)]
        for n in names:
            self.write(n, [_SENTINEL] * _numel(self.spec[n][0]))
        if self.spec_nz:
            for k in range(self.K):
                self.write(f"nz{k}", [-12345.0] * _numel(self.spec[f"nz{k}"][0]))

    def run(self):
        t, K = self.t, self.K
        kw = {}
        if self.spec_c2:
            kw = {"accept": t["accept"], "resid": t["resid"]}
            if self.spec_nz:
                kw.update(tabs=[t[f"tab{k}"] for k in range(K)], nz=[t[f"nz{k}"] for k in range(K)])
        return run(t["ids"], t["v_tok"], t["plan"], t["gslot"], t["anchor"], t["t0n"],
                   [t[f"d_pos{k}"] for k in range(K)], [t[f"d_rpos{k}"] for k in range(K)], t["m"], self.bw, self.T,
                   **kw)

    def got(self, device_index=0):
        u = lambda n: [int(v) & 0xFFFFFFFF for v in self.read(n, device_index)]  # noqa: E731
        s = lambda n: [_mbh._i32(int(v)) for v in self.read(n, device_index)]  # noqa: E731
        out = {"gslot": u("gslot"), "anchor": u("anchor"), "t0n": s("t0n"), "m": u("m"),
               "d_pos": [s(f"d_pos{k}") for k in range(self.K)], "d_rpos": [s(f"d_rpos{k}") for k in range(self.K)]}
        if self.spec_nz:
            C2 = self.spec_c2
            out["d_sgum"] = []
            for k in range(self.K):
                v = [float(x) for x in self.read(f"nz{k}", device_index)]
                out["d_sgum"].append([v[b * C2:(b + 1) * C2] for b in range(self.bw)])
        return out

    def free(self):
        for x in self.t.values():
            ttnn.deallocate(x)
        self.t = {}


def _numel(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def synth_noise(bw, K, C2, seed):
    """Synthetic chain-noise tables: K tables of (K+1)*bw rows of C2 fp32 values, every row distinct (so a wrong row
    index cannot pass), with a few edge patterns (-0.0, a denormal, +-inf, a quiet NaN) in row 0 of table 0."""
    import random

    rng = random.Random(f"nz/{bw}/{K}/{C2}/{seed}")
    tabs = []
    for k in range(K):
        rows = []
        for i in range((K + 1) * bw):
            rows.append([float(rng.getrandbits(16)) / 64.0 + 1000.0 * k + i for _ in range(C2)])
        tabs.append(rows)
    tabs[0][0][:5] = [-0.0, 1e-40, float("inf"), float("-inf"), 7.0]
    return tabs


def _exact_rows(bw, seed):
    return [((b + seed) % 3) != 1 for b in range(bw)]


def selftest(device, bw, K, ids_mc=None, host_mc=None, seeds=(0, 1), cases=None, spec_c2=None, flag_layout=None,
             flag_mc=None, spec_nz=True):
    """Fail-closed check of the device accept on THIS device and image at bucket (bw, K): for every synthetic case of
    mtp_batch_host.da_case (all / none / partial / mixed acceptance, inactive rows incl. row 0, fresh and stale
    slots, a capped KV capacity, FORCE_REJECT, MUSTFAIL=commit_plus1) and every seed, the REAL host plan() and accept()
    give the expected inputs (da_expected); the kernel runs on the same persistent buffers (outputs pre-filled with a
    sentinel, so a field the kernel fails to write cannot pass) and must match them bitwise on every device. With
    spec_c2 (the served exact rule) every case also runs with a mix of exact rows (the rule's fp32 accept flags with
    edge bit patterns, residual ids) and the chain-noise gather. Must-fail: the QWEN36_MTP_DEVACCEPT_MUSTFAIL flag on
    the 'mixed' case must differ from the host accept AND equal da_reference with the flag (the kernel read the page,
    not a constant). Returns {check: "exact" | ...}."""
    buf = Buffers(device, bw, K, ids_mc, host_mc, spec_c2=spec_c2, flag_layout=flag_layout, flag_mc=flag_mc,
                  spec_nz=spec_nz)
    nzc = buf.spec_nz  # the chain-noise copy (off under QWEN36_MTP_SPEC_SAMPLE_SSP)
    res = {}

    def one(name, seed, exact_rows, da_mf=False):
        host, plan, ids, acc, spec = _mbh.da_case(name, bw, K, seed, exact_rows=exact_rows)
        page = _mbh.da_plan_page(plan, host.reject_all, host.mustfail, da_mustfail=da_mf, exact=exact_rows)
        tabs = synth_noise(bw, K, spec_c2, seed) if nzc else None
        sg = [[tabs[k][int(acc.m[b]) * bw + b] for b in range(bw)] for k in range(K)] if nzc else None
        exp = _mbh.da_expected(plan, acc, d_sgum=sg)
        acc_f = spec["accept"] if spec else [0.0] * buf.M
        rs = spec["resid"] if spec else [0] * buf.M
        ref = _mbh.da_reference(page, plan.verify_tokens, ids, bw, K + 1, accept=acc_f if spec_c2 else None,
                                resid=rs if spec_c2 else None, noise=tabs)
        buf.write("ids", ids)
        buf.write("v_tok", plan.verify_tokens)
        buf.write("plan", page)
        if spec_c2:
            buf.write_bits("accept", [_mbh.f32_bits(a) for a in acc_f])
            buf.write("resid", rs)
            for k in range(K if nzc else 0):
                buf.write(f"tab{k}", [x for row in tabs[k] for x in row])
        buf.sentinel()
        buf.run()
        return exp, ref

    try:
        modes = [None] + ([True] if spec_c2 else [])
        for name in (cases or _mbh.DA_CASES):
            for seed in seeds:
                for mode in modes:
                    if mode and name == "reject_all":
                        continue  # FORCE_REJECT is refused under the served rule
                    ex = _exact_rows(bw, seed) if mode else ([False] * bw if spec_c2 else None)
                    key = f"{name}/{seed}{'/exact' if mode else ''}"
                    exp, ref = one(name, seed, ex)
                    if _mbh.da_diff(ref, exp):  # the Python mirror itself disagrees with the host accept
                        res[key] = f"reference differs {_mbh.da_diff(ref, exp)}"
                        continue
                    bad = sorted({f for d in range(buf.n_devices()) for f in _mbh.da_diff(buf.got(d), exp)})
                    res[key] = "exact" if not bad else f"differs {bad}"
        ex = _exact_rows(bw, 0) if spec_c2 else None
        exp, ref = one("mixed", 0, ex, da_mf=True)
        got = buf.got(0)
        res["mustfail_row0"] = "differs" if (_mbh.da_diff(got, exp) and not _mbh.da_diff(got, ref)) else (
            "blind" if not _mbh.da_diff(got, exp) else f"wrong {_mbh.da_diff(got, ref)}")
    finally:
        buf.free()
    return res


def selftest_ok(res):
    return bool(res) and res.get("mustfail_row0") == "differs" and all(
        v == "exact" for k, v in res.items() if k != "mustfail_row0")


# ------------------------------------------------------------------------------------------- QWEN36_MTP_DEVFEED
# docs/MTP-BATCH.md section 31: the next verify's draft-dependent inputs built on device (kernels/mtp_accept/feed.cpp),
# the last program of the sampled family's draft trace.
FEED_ENV = _mbh.ENV_DEVFEED
_FSRC = Path(__file__).resolve().parent / "kernels" / "mtp_accept" / "feed.cpp"
_FEED_CB = 0


def _fsrc():
    if "f" not in _SRC:
        _SRC["f"] = _FSRC.read_text()
    return _SRC["f"]


def _frm(t, what, dtype, shape):
    if t.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"{FEED_ENV}: {what} must be ROW_MAJOR, got {t.layout}")
    if t.dtype != dtype:
        raise RuntimeError(f"{FEED_ENV}: {what} dtype {t.dtype} != {dtype}")
    s = _shape(t)
    if s[-len(shape):] != list(shape) or any(v != 1 for v in s[:-len(shape)]):
        raise RuntimeError(f"{FEED_ENV}: {what} shape {s} != [.., {', '.join(map(str, shape))}]")
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise RuntimeError(f"{FEED_ENV}: {what} must be DRAM interleaved, got {mc}")
    return _page(t)


def feed(t0n, drafts, qps, qis, tab, vtok, vqp, vqi, vx, bw, K, n_ids=0, mf=False):
    """One device feed for a bucket of width bw and K drafts per row: writes vtok [M,1] u32, vqp / vqi [1,1,M,C2] fp32 and
    vx [1,1,M,1] fp32 (M = (K+1)*bw) in place from t0n [bw,1] u32, the K draft-id tensors [bw,1] u32 and the K q-prob /
    q-id blocks [bw,C2] fp32 (tab: the [N/16,16] u32 draft-vocab id table, n_ids = N; None = no map). mf: the corrupted
    draft (QWEN36_MTP_DEVFEED_MUSTFAIL). Returns vx."""
    K, bw = int(K), int(bw)
    M = (K + 1) * bw
    if not (1 <= K and M <= 32 and len(drafts) == K and len(qps) == K and len(qis) == K):
        raise RuntimeError(f"{FEED_ENV}: bw={bw} K={K}: needs (K+1)*bw <= 32 and K draft / probs / id tensors")
    C2 = _shape(qps[0])[-1]
    u32, f32 = ttnn.uint32, ttnn.float32
    pid = [_frm(t0n, "t0n", u32, (bw, 1)), _frm(vtok, "v_tok", u32, (M, 1)), _frm(vx, "v_sx", f32, (M, 1))]
    pid += [_frm(t, f"draft ids {k}", u32, (bw, 1)) for k, t in enumerate(drafts)]
    pq = [_frm(vqp, "v_sqp", f32, (M, C2)), _frm(vqi, "v_sqi", f32, (M, C2))]
    pq += [_frm(t, f"q probs {k}", f32, (bw, C2)) for k, t in enumerate(qps)]
    pq += [_frm(t, f"q ids {k}", f32, (bw, C2)) for k, t in enumerate(qis)]
    if len(set(pid)) != 1 or len(set(pq)) != 1:
        raise RuntimeError(f"{FEED_ENV}: id pages {pid} / q pages {pq} must agree")
    P_ID, P_Q = pid[0], pq[0]
    if P_ID > 64 or P_Q % 64 or P_Q < 4 * C2:
        raise RuntimeError(f"{FEED_ENV}: page sizes id={P_ID} q={P_Q} outside the kernel's slots")
    MAP = tab is not None
    P_TB = _frm(tab, "id table", u32, (int(tab.shape[-2]), 16)) if MAP else 64
    if MAP and not (1 <= int(n_ids) <= 16 * int(tab.shape[-2])):
        raise RuntimeError(f"{FEED_ENV}: n_ids={n_ids} outside the table's {16 * int(tab.shape[-2])} entries")
    every = [t0n, vtok, vqp, vqi, vx] + list(drafts) + list(qps) + list(qis) + ([tab] if MAP else [])
    h_ct = list(ttnn.TensorAccessorArgs(vtok).get_compile_time_args())

    def _cls(ct):
        return [ct[0]] + ct[2:] if len(ct) >= 2 else ct

    for t in every:
        if _cls(list(ttnn.TensorAccessorArgs(t).get_compile_time_args())) != _cls(h_ct):
            raise RuntimeError(f"{FEED_ENV}: every tensor must share one placement ({t.memory_config()})")
    QS, TS = (P_Q + 63) // 64 * 64, (P_TB + 63) // 64 * 64
    cb_bytes = 3 * 64 + 4 * QS + (C2 * TS if MAP else 0)
    dev = vtok.device()
    g = dev.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
    nc = K * bw
    if nc > gx * gy:
        raise RuntimeError(f"{FEED_ENV}: {nc} rows > {gx * gy} cores")
    cores = [(c % gx, c // gx) for c in range(nc)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    ct = [bw, K, C2, int(MAP), int(bool(mf)), int(n_ids) if MAP else 1, P_ID, P_Q, P_TB, _FEED_CB] + h_ct
    common = [t0n.buffer_address(), vtok.buffer_address(), vqp.buffer_address(), vqi.buffer_address(),
              vx.buffer_address(), tab.buffer_address() if MAP else 0]
    common += [t.buffer_address() for t in drafts] + [t.buffer_address() for t in qps] + [t.buffer_address() for t in qis]
    rt = ttnn.RuntimeArgs()
    for u, (x, y) in enumerate(cores):
        rt[x][y] = [u] + common
    desc = ttnn.ProgramDescriptor(
        kernels=[ttnn.KernelDescriptor(kernel_source=_fsrc(), source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                       core_ranges=crs, compile_time_args=ct, runtime_args=rt,
                                       config=ttnn.ReaderConfigDescriptor())],
        semaphores=[],
        cbs=[ttnn.CBDescriptor(total_size=cb_bytes, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_FEED_CB, data_format=ttnn.uint32, page_size=cb_bytes)])],
    )
    io = [t0n] + list(drafts) + list(qps) + list(qis) + ([tab] if MAP else []) + [vtok, vqp, vqi, vx]
    return ttnn.generic_op(io, desc)


def feed_case(bw, K, C2, n_ids, seed, vocab=248320):
    """A synthetic chain output: t0 / base positions / accepted lengths per row, K draft ids per row (FULL ids, incl.
    0 and vocab-1), and K q blocks: probs with edge bit patterns (-0.0, a denormal, 1.0, tiny) and q ids that are
    draft-vocab sub-ids (incl. 0 and n_ids-1; n_ids None = full ids, no map) as fp32."""
    import random
    import struct

    import torch

    rng = random.Random(f"feed/{bw}/{K}/{C2}/{n_ids}/{seed}")
    t0 = [rng.randrange(vocab) for _ in range(bw)]
    p0 = [rng.randrange(0, 60000) for _ in range(bw)]
    m = [rng.randrange(K + 1) for _ in range(bw)]
    drafts = [[rng.randrange(vocab) for _ in range(bw)] for _ in range(K)]
    drafts[0][0], drafts[K - 1][bw - 1] = 0, vocab - 1
    span = n_ids if n_ids else vocab
    qp, qi = [], []
    for k in range(K):
        bits = [[rng.getrandbits(30) for _ in range(C2)] for _ in range(bw)]  # positive finite fp32 patterns
        if k == 0:
            bits[0][:4] = [0x80000000, 0x00000001, 0x3F800000, 0x00800000]
        qp.append(torch.tensor([[struct.unpack("<f", struct.pack("<I", v))[0] for v in r] for r in bits],
                               dtype=torch.float32))
        ids = [[float(rng.randrange(span)) for _ in range(C2)] for _ in range(bw)]
        if k == 0:
            ids[0][:2] = [0.0, float(span - 1)]
        qi.append(torch.tensor(ids, dtype=torch.float32))
    return {"t0": t0, "p0": p0, "m": m, "drafts": drafts, "qp": qp, "qi": qi}


def feed_expected(case, bw, K, C2, ids_tab=None, padded=()):
    """What the S10 host path writes for the step after `case`: the REAL MTPBatchHost.finish (with the served q-id map,
    mtp_batch_host.map_q_ids) then the REAL plan() and the served builders (host_inputs' v_tok, spec_v_blocks).
    Returns {name: torch tensor} in the host_inputs shapes.
    padded (QWEN36_MTP_DEVFEED_PAD): rows whose next position is -1 (plan()'s padded rows)."""
    import torch

    T, M = K + 1, (K + 1) * bw
    host = _mbh.MTPBatchHost(max_batch=bw, kcap=K)
    host.spec_c2 = C2
    p0, m, t0 = case["p0"], case["m"], case["t0"]
    cap = [max(p0) + 4 * T + 64] * bw
    for b in range(bw):
        host.slots[b] = _mbh.SlotState(origin=0, next_p=p0[b], t0=0)
    pplan = _mbh.Plan(bw=bw, K=K, T=T, active=[True] * bw, p=list(p0), t0=[0] * bw, n_draft=[K] * bw,
                      drafts=[[0] * K for _ in range(bw)], origin=[0] * bw, verify_tokens=[], verify_pos=[],
                      verify_rope_pos=[], kv_capacity=list(cap))
    pacc = _mbh.Accept(m=list(m), emitted=[[] for _ in range(bw)], gdn_onehot=[], anchor_idx=[], t0_next=list(t0),
                       reseed_pos=[], reseed_rope_pos=[], draft_pos=[], draft_rope_pos=[])
    draft_ids = [case["drafts"][k][b] for k in range(K) for b in range(bw)]
    qib = [t.clone() for t in case["qi"]]
    if ids_tab is not None:
        qib = _mbh.map_q_ids(qib, torch.as_tensor(ids_tab, dtype=torch.int64))
    host.finish(pplan, pacc, draft_ids, ([t.clone() for t in case["qp"]], qib))
    plan = host.plan(list(t0), [-1 if b in padded else p0[b] + m[b] + 1 for b in range(bw)], bw, K, cap)
    if any(int(n) != K for n, a in zip(plan.n_draft, plan.active) if a):
        raise RuntimeError(f"{FEED_ENV}: the reference plan has n_draft {plan.n_draft} (expected {K} everywhere)")
    qp, qi, xs, _ = _mbh.spec_v_blocks(plan, bw, K, M, C2, [1] * bw)
    out = {"v_tok": torch.tensor(plan.verify_tokens, dtype=torch.int32).reshape(M, 1), "v_sqp": qp, "v_sqi": qi,
           "v_sx": xs}
    if padded:
        tok, q = _mbh.feed_active_rows(plan, M, K)
        out["_rows"] = {"v_tok": tok, "v_sqp": q, "v_sqi": q, "v_sx": q}
    return out


def feed_bits(name, t):
    """A fed buffer's value as a flat list of 32-bit words (ints: & 0xFFFFFFFF; floats: the IEEE bits)."""
    import torch

    t = t.reshape(-1)
    if t.dtype == torch.float32:
        return [int(v) & 0xFFFFFFFF for v in t.view(torch.int32).tolist()]
    return [int(v) & 0xFFFFFFFF for v in t.to(torch.int64).tolist()]


def feed_diff(got, exp, rows=None):
    """{name: number of differing words} over the fed buffers (got: device reads as torch, any shape).
    rows (QWEN36_MTP_DEVFEED_PAD): {name: row indices} compares only those rows of exp ([M, W]; W words per row)."""
    bad = {}
    for n in _mbh.FEED_NAMES:
        g, e = feed_bits(n, got[n]), feed_bits(n, exp[n])
        if len(g) < len(e):
            bad[n] = f"short {len(g)} < {len(e)}"
            continue
        if rows is not None and n in rows:
            w = int(exp[n].shape[-1]) if exp[n].dim() > 1 else 1
            keep = {r * w + c for r in rows[n] for c in range(w)}
            d = sum(1 for i, (a, b) in enumerate(zip(g, e)) if i in keep and a != b)
        else:
            d = sum(1 for a, b in zip(g, e) if a != b)
        if d:
            bad[n] = d
    return bad


def feed_selftest(device, bw, K, C2, ids=None, seeds=(0, 1), pad=False):
    """Fail-closed check of the device feed on THIS device and image at bucket (bw, K), C2-wide q blocks, with the
    draft-vocab map when ids (the draft vocab's full ids) is given: for every seed, synthetic chain outputs on device,
    outputs pre-filled with a sentinel, the feed's four buffers must equal feed_expected (the real host path) bitwise on
    every device; the corrupted-draft variant (mf) must differ. Returns {check: "exact" | ...}.
    pad (QWEN36_MTP_DEVFEED_PAD, bw > 1): one more case with rows 1.. (odd rows) padded in the next step: the ACTIVE rows
    must be exact under the padded self-test's row mask ("pad"), and the unmasked compare must differ ("pad_unmasked":
    the padded rows hold the chain's values, the host build zeros / tokens; the PAD_MUSTFAIL branch)."""
    import torch

    from models.demos.blackhole.qwen36.tt import fused_sampler as _fs

    mesh = hasattr(ttnn, "MeshDevice") and isinstance(device, ttnn.MeshDevice)
    rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if mesh else {}
    M = (K + 1) * bw
    R, DR = ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG

    def up(t, dt):
        if dt == ttnn.uint32:
            t = t.to(torch.int64) & 0xFFFFFFFF
        return ttnn.from_torch(t, dtype=dt, layout=R, device=device, memory_config=DR, **rep)

    def back(t):
        ts = ttnn.get_device_tensors(t) if mesh else [t]
        return [ttnn.to_torch(x) for x in ts]

    n_ids = len(ids) if ids is not None else None
    tab = _fs.id_table(list(ids), device, shift=0) if ids is not None else None
    res = {}
    outs, ins = {}, []
    try:
        outs = {"v_tok": up(torch.zeros(M, 1, dtype=torch.int64), ttnn.uint32),
                "v_sqp": up(torch.zeros(1, 1, M, C2), ttnn.float32), "v_sqi": up(torch.zeros(1, 1, M, C2), ttnn.float32),
                "v_sx": up(torch.zeros(1, 1, M, 1), ttnn.float32)}
        cases = list(seeds) + ["mf"] + (["pad"] if pad and bw > 1 else [])
        for seed in cases:
            mf = seed == "mf"
            padded = tuple(range(1, bw, 2)) if seed == "pad" else ()
            case = feed_case(bw, K, C2, n_ids, 0 if mf else (2 if seed == "pad" else seed))
            exp = feed_expected(case, bw, K, C2, ids_tab=ids, padded=padded)
            rows = exp.pop("_rows", None)
            ins = [up(torch.tensor(case["t0"], dtype=torch.int64).reshape(bw, 1), ttnn.uint32)]
            ins += [up(torch.tensor(case["drafts"][k], dtype=torch.int64).reshape(bw, 1), ttnn.uint32) for k in range(K)]
            ins += [up(case["qp"][k].clone(), ttnn.float32) for k in range(K)]
            ins += [up(case["qi"][k].clone(), ttnn.float32) for k in range(K)]
            # the sentinel: a field the kernel fails to write cannot pass
            for n, t in outs.items():
                shp = [int(s) for s in t.shape]
                fill = torch.full(shp, 0x5A5A5A5A, dtype=torch.int64) if n == "v_tok" else torch.full(shp, -12345.0)
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(fill, dtype=t.dtype, layout=R, **rep), t)
            feed(ins[0], ins[1:1 + K], ins[1 + K:1 + 2 * K], ins[1 + 2 * K:1 + 3 * K], tab, outs["v_tok"],
                 outs["v_sqp"], outs["v_sqi"], outs["v_sx"], bw, K, n_ids=n_ids or 0, mf=mf)
            reads = {n: back(t) for n, t in outs.items()}
            nd = len(reads["v_tok"])
            bad = {}
            for d in range(nd):
                for n, v in feed_diff({n: reads[n][d] for n in outs}, exp, rows).items():
                    bad[f"{n}@{d}"] = v
            if seed == "pad":
                res["pad"] = "exact" if not bad else f"differs {bad}"
                unmasked = any(feed_diff({n: reads[n][d] for n in outs}, exp) for d in range(nd))
                res["pad_unmasked"] = "differs" if unmasked else "blind"
            elif mf:
                # must differ exactly in draft 0 of row 0 (v_tok row bw, v_sx row 0) on every device
                want = {f"{n}@{d}": 1 for d in range(nd) for n in ("v_tok", "v_sx")}
                res["mustfail_draft"] = "differs" if bad == want else ("blind" if not bad else f"wrong {bad}")
            else:
                res[f"seed{seed}"] = "exact" if not bad else f"differs {bad}"
            for t in ins:
                ttnn.deallocate(t)
            ins = []
    finally:
        for t in list(outs.values()) + ins:
            ttnn.deallocate(t)
        if tab is not None:
            ttnn.deallocate(tab)
    return res


def feed_selftest_ok(res):
    return bool(res) and res.get("mustfail_draft") == "differs" and res.get("pad_unmasked", "differs") == "differs" and all(
        v == "exact" for k, v in res.items() if k not in ("mustfail_draft", "pad_unmasked"))
