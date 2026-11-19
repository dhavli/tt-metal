# QWEN36_OPMARK: name every device program of a profiled window by the ttnn op that launched it.
#
# Why: the realtime profiler (and the device profiler without a host Tracy capture) gives each program a runtime id,
# device timestamps and kernel source paths, but no op name. Under trace replay the kernel-source lookup is keyed by the
# 16-bit runtime id and collides, so a traced prefill chunk files ~90% of its time under one kernel name (c127,
# docs/STATUS.md). The AGMM's five roles share one op and can only be told apart by duration clusters.
#
# What: when QWEN36_OPMARK=1 at boot, install() wraps ttnn.decorators.FastOperation.__call__ (every ttnn op called from
# Python, including the experimental and custom ones) and the forward of each decoder layer, its attention/GDN mixer and
# its MLP. While a recording window is open (begin() ... end(), driven by the profiler client, e.g.
# tools/ttx/clients/wb_backend.py profile), each call appends one row: the device-operation id range it consumed
# (ttnn._ttnn.get_device_operation_id before and after = the runtime ids of its programs), the op's Python name, the
# layer/sublayer scope, and a compact signature (input shapes and dtypes, program-config grid, math fidelity, scalar
# kwargs such as chunk_start_idx). tools/ttx/clients/opmark_join.py joins those rows to profiler records by runtime id.
# Rows recorded during trace CAPTURE name the programs of every later replay of that trace (same runtime ids).
# Trace bookkeeping (c177): ttnn.begin/end_trace_capture are wrapped so every capture's device-operation id range is
# kept per trace id (persistent, window or not), and ttnn.execute_trace inside a window appends a zero-width
# "<trace-replay>" row (sig "trace=<key>") at its issue point. The joiner then knows exactly which capture each replay
# re-issued: without that, the realtime profiler's 16-bit runtime ids collide across captures and a low-16-bit match
# filed chunk-trace programs under the 512 bucket and decode captures (c169).
#
# Off (unset/0): install() returns before touching anything; no wrapper exists, so the served path is byte-identical.
# On: device programs are unchanged too (host-side counter reads only); with no window open each wrapped call costs
# one global check. The banner "QWEN36_OPMARK engaged" proves it is on; no other code reads this module.
from __future__ import annotations
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import bisect
import contextlib
import os
import threading
import time

from loguru import logger

_STATE = {"installed": False, "getid": None, "rows": None, "cap": 0, "dropped": 0, "scope": [], "windows": 0,
          "captures": {}, "open_caps": {}}
_LOCK = threading.Lock()
_SCALAR_KW = ("chunk_start_idx", "fuse_swiglu", "num_links", "cluster_axis", "is_causal", "scale", "dim",
              "output_head_major", "chunk_size", "topology", "transpose_a", "transpose_b", "activation")


def enabled_env():
    v = _tt_public_getenv("QWEN36_OPMARK", "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"QWEN36_OPMARK={v!r}: expected unset, 0 or 1")
    return v == "1"


def engaged():
    return _STATE["installed"]


def _tensor_sig(t):
    try:
        shp = tuple(int(x) for x in t.shape)
    except Exception:  # noqa: BLE001
        return None
    try:
        dt = str(t.dtype).rsplit(".", 1)[-1]
    except Exception:  # noqa: BLE001
        dt = "?"
    return f"{'x'.join(map(str, shp))}:{dt}"


def _grid_of(obj):
    """Cores of a program config / core grid object, best effort: compute_with_storage_grid_size (x, y), core_grid,
    or a CoreRangeSet's num_cores()."""
    for attr in ("compute_with_storage_grid_size", "core_grid", "grid_size", "grid"):
        g = getattr(obj, attr, None)
        if g is None:
            continue
        x, y = getattr(g, "x", None), getattr(g, "y", None)
        if isinstance(x, int) and isinstance(y, int):
            return f"{x}x{y}"
        n = getattr(g, "num_cores", None)
        if callable(n):
            try:
                return str(int(n()))
            except Exception:  # noqa: BLE001
                pass
    return None


def _signature(args, kwargs):
    ins = []
    for a in list(args)[:4]:
        if isinstance(a, (list, tuple)) and a and hasattr(a[0], "shape"):
            # generic_op(io_tensors, descriptor): the io list's size and its first two tensors tell the custom
            # programs apart (QWEN36_MTP_OPTRACE splits ttnn.generic_op rows by it)
            sub = [x for x in (_tensor_sig(t) for t in a[:2] if hasattr(t, "shape")) if x]
            ins.append(f"[{'|'.join(sub)}]x{len(a)}")
            continue
        s = _tensor_sig(a) if hasattr(a, "shape") and hasattr(a, "dtype") else None
        if s:
            ins.append(s)
    for k in ("input_tensor", "input_tensor_a", "input_tensor_b", "weight_tensor", "weight", "input_tensor_q",
              "input_tensor_k", "input_tensor_v"):
        v = kwargs.get(k)
        if v is not None and hasattr(v, "shape"):
            s = _tensor_sig(v)
            if s:
                ins.append(f"{k}={s}")
    parts = ["in=" + ",".join(ins)] if ins else []
    for k in ("program_config", "config"):
        pc = kwargs.get(k)
        if pc is not None:
            g = _grid_of(pc)
            parts.append(f"pc={type(pc).__name__}" + (f"@{g}" if g else ""))
    ckc = kwargs.get("compute_kernel_config")
    if ckc is not None:
        f = getattr(ckc, "math_fidelity", None)
        if f is not None:
            parts.append(f"fid={str(f).rsplit('.', 1)[-1]}")
    for k in _SCALAR_KW:
        v = kwargs.get(k)
        if isinstance(v, (bool, int, float, str)):
            parts.append(f"{k}={v}")
    dt = kwargs.get("dtype")
    if dt is not None:
        parts.append(f"dtype={str(dt).rsplit('.', 1)[-1]}")
    return ";".join(parts)


def _record(name, lo, hi, sig, scope=None):
    rows = _STATE["rows"]
    if rows is None or hi <= lo:
        return
    if len(rows) >= _STATE["cap"]:
        _STATE["dropped"] += 1
        return
    rows.append((lo, hi, name, "/".join(_STATE["scope"]) if scope is None else scope, sig))


def _wrap_op_call(orig):
    getid = _STATE["getid"]

    def __call__(self, *args, **kwargs):
        if _STATE["rows"] is None:
            return orig(self, *args, **kwargs)
        lo = int(getid())
        try:
            return orig(self, *args, **kwargs)
        finally:
            hi = int(getid())
            if hi > lo:
                try:
                    sig = _signature(args, kwargs)
                except Exception:  # noqa: BLE001
                    sig = "?"
                _record(getattr(self, "python_fully_qualified_name", None) or "?", lo, hi, sig)

    __call__._qwen36_opmark = True
    return __call__


def _wrap_scope(obj, attr, tag):
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, "_qwen36_opmark", False):
        return False
    getid = _STATE["getid"]

    def scoped(*args, **kwargs):
        st = _STATE["scope"]
        if _STATE["rows"] is None or (st and st[-1] == tag):  # forward -> forward_prefill: one scope level
            return fn(*args, **kwargs)
        st.append(tag)
        path = "/".join(st)
        lo = int(getid())
        try:
            return fn(*args, **kwargs)
        finally:
            hi = int(getid())
            if st and st[-1] == tag:
                st.pop()
            # the scope's own range: programs no FastOperation claimed (C++-internal launches) still get a scope
            _record("<scope>", lo, hi, "", scope=path)

    scoped._qwen36_opmark = True
    setattr(obj, attr, scoped)
    return True


def _trace_key(device, tid):
    """The trace id alone: one MeshDevice per serving process (TP=2 is one (1,2) mesh), and capture and replay may
    reach it through different Python handles (model.device vs model_args[i].mesh_device), so id(device) is no key."""
    try:
        return str(int(tid))
    except Exception:  # noqa: BLE001
        return str(tid)


def _wrap_trace_api(ttnn):
    """begin/end_trace_capture -> _STATE["captures"][key] = [id_lo, id_hi); execute_trace -> a "<trace-replay>" row."""
    getid = _STATE["getid"]
    begin, endc, execute = (getattr(ttnn, n, None) for n in ("begin_trace_capture", "end_trace_capture",
                                                             "execute_trace"))
    if begin is None or endc is None or execute is None or getattr(execute, "_qwen36_opmark", False):
        return False

    def begin_trace_capture(device, *args, **kwargs):
        lo = int(getid())
        tid = begin(device, *args, **kwargs)
        _STATE["open_caps"][_trace_key(device, tid)] = lo
        return tid

    def end_trace_capture(device, tid, *args, **kwargs):
        out = endc(device, tid, *args, **kwargs)
        key = _trace_key(device, tid)
        lo = _STATE["open_caps"].pop(key, None)
        if lo is not None:
            _STATE["captures"][key] = [lo, int(getid())]
        return out

    def execute_trace(device, tid, *args, **kwargs):
        rows = _STATE["rows"]
        if rows is not None and len(rows) < _STATE["cap"]:
            at = int(getid())
            rows.append((at, at, "<trace-replay>", "/".join(_STATE["scope"]), f"trace={_trace_key(device, tid)}"))
        return execute(device, tid, *args, **kwargs)

    for f in (begin_trace_capture, end_trace_capture, execute_trace):
        f._qwen36_opmark = True
    ttnn.begin_trace_capture, ttnn.end_trace_capture, ttnn.execute_trace = (begin_trace_capture, end_trace_capture,
                                                                            execute_trace)
    return True


def install(model):
    """QWEN36_OPMARK=1: wrap FastOperation.__call__ and every layer's forward scopes. Returns True when engaged."""
    if not enabled_env():
        return False
    if _STATE["installed"]:
        return True
    import ttnn

    getid = getattr(getattr(ttnn, "_ttnn", None), "get_device_operation_id", None)
    fast = getattr(getattr(ttnn, "decorators", None), "FastOperation", None)
    if getid is None or fast is None:
        logger.warning("QWEN36_OPMARK set but NOT engaged: this ttnn has no _ttnn.get_device_operation_id / "
                       "decorators.FastOperation")
        return False
    _STATE["getid"] = getid
    if not getattr(fast.__call__, "_qwen36_opmark", False):
        fast.__call__ = _wrap_op_call(fast.__call__)
    traces = _wrap_trace_api(ttnn)
    n = 0
    for i, layer in enumerate(getattr(model, "layers", []) or []):
        kind = "attn" if getattr(layer, "is_full_attention", False) else "gdn"
        n += _wrap_scope(layer, "forward", f"L{i}.{kind}")
        mixer = getattr(layer, "attention", None)
        if mixer is not None:
            for meth in ("forward", "forward_prefill", "forward_decode"):
                _wrap_scope(mixer, meth, kind)
        mlp = getattr(layer, "feed_forward", None)
        if mlp is not None:
            for meth in ("forward", "forward_prefill", "forward_decode"):
                _wrap_scope(mlp, meth, "mlp")
    _STATE["installed"] = True
    logger.info(f"QWEN36_OPMARK engaged: host-side op-id annotation (FastOperation.__call__ + {n} layer scopes); "
                f"device programs unchanged; recording only inside a profiler window; trace capture/replay "
                f"bookkeeping {'on' if traces else 'OFF (ttnn has no begin/end_trace_capture/execute_trace)'}")
    return True


def begin(cap=2_000_000):
    """Open a recording window (rows accumulate until end()). Nested begin() keeps the outer window."""
    with _LOCK:
        if not _STATE["installed"]:
            return False
        if _STATE["rows"] is None:
            _STATE["rows"], _STATE["cap"], _STATE["dropped"] = [], int(cap), 0
            _STATE["windows"] += 1
        return True


def end():
    """Close the window: -> {"rows": [[id_lo, id_hi, name, scope, sig], ...], "dropped": n,
    "captures": {trace key: [id_lo, id_hi)} of every capture this process made (latest per key)}."""
    with _LOCK:
        rows, _STATE["rows"] = _STATE["rows"], None
        _STATE["scope"] = []
        return {"rows": [list(r) for r in (rows or [])], "dropped": _STATE["dropped"],
                "captures": {k: list(v) for k, v in _STATE["captures"].items()}}



# ------------------------------------------------------------------ QWEN36_MTP_OPTRACE (served MTP step windows)
# Used only by mtp_batch.py when QWEN36_MTP_OPTRACE=1 (which requires QWEN36_OPMARK=1). Host-side: a scope tag on the
# opmark stack, the realtime profiler's program records over a window of served steps, and the c177 stream alignment
# (tools/ttx/clients/opmark_join.py on opmark-c177, ported) that names each record by the op row whose device-op id
# range holds it. Nothing here enqueues a device program.


@contextlib.contextmanager
def scope(tag):
    """Push `tag` on the opmark scope stack while a window is open (else a no-op): rows recorded inside carry it, and a
    "<scope>" row covers the whole range."""
    st = _STATE["scope"]
    if _STATE["rows"] is None:
        yield
        return
    st.append(tag)
    path = "/".join(st)
    getid = _STATE["getid"]
    lo = int(getid())
    try:
        yield
    finally:
        hi = int(getid())
        if st and st[-1] == tag:
            st.pop()
        _record("<scope>", lo, hi, "", scope=path)  # programs no FastOperation claimed still get the scope


class RTWindow:
    """The realtime profiler's program records between open() and close() (ttnn.device callback; host thread)."""

    def __init__(self):
        self.records, self.dropped, self.h = [], 0, None

    def _cb(self, batch):
        self.dropped += int(getattr(batch, "dropped", 0) or 0)
        for r in batch.records:
            self.records.append((int(r.runtime_id), str(r.chip_id), int(r.start_timestamp), int(r.end_timestamp),
                                 float(r.frequency or 0)))

    @staticmethod
    def available():
        import ttnn

        dev = getattr(ttnn, "device", None)
        if dev is None or not hasattr(dev, "RegisterProgramRealtimeProfilerCallback"):
            return False, "ttnn.device.RegisterProgramRealtimeProfilerCallback is not on this image"
        try:
            if not dev.IsProgramRealtimeProfilerActive():
                return False, "IsProgramRealtimeProfilerActive() is False"
        except Exception as e:  # noqa: BLE001
            return False, f"IsProgramRealtimeProfilerActive() raised {e!r}"
        return True, ""

    def open(self):
        import ttnn

        self.records, self.dropped = [], 0
        self.h = ttnn.device.RegisterProgramRealtimeProfilerCallback(self._cb)

    def close(self, settle_s=0.5):
        import ttnn

        time.sleep(settle_s)  # the callback thread delivers the last batches after the device is idle
        if self.h is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(self.h)
            self.h = None
        return self.records, self.dropped


def _mhz(f):
    f = float(f or 0)
    return f / 1e6 if f > 1e5 else (f * 1e3 if f < 100 else f)


def owner_index(rows):
    """{full device-op id: row index}; the narrowest row that holds an id owns it (op rows beat <scope> rows)."""
    order = sorted(range(len(rows)), key=lambda i: -(int(rows[i][1]) - int(rows[i][0])))
    owner = {}
    for i in order:
        lo, hi = int(rows[i][0]), int(rows[i][1])
        for x in range(lo, hi):
            owner[x] = i
    return owner


def expected_stream(window_rows, captures):
    """-> (ids, replay_of) in issue order: every eager id of the window and, at each "<trace-replay>" row, that
    capture's [lo, hi). replay_of[i] = the index of the replay row the id came from (or -1 for an eager id)."""
    ops = [r for r in window_rows if r[2] != "<trace-replay>" and int(r[1]) > int(r[0])]
    reps = [(int(r[0]), k, str(r[4])[len("trace="):]) for k, r in enumerate(window_rows) if r[2] == "<trace-replay>"]
    if not ops and not reps:
        return [], [], 0
    lo = min([int(r[0]) for r in ops] + [a for a, _, _ in reps])
    hi = max([int(r[1]) for r in ops] + [a for a, _, _ in reps])
    reps.sort()
    ids, rep_of, nxt, missing = [], [], lo, 0
    for at, k, key in reps:
        ids.extend(range(nxt, at))
        rep_of.extend([-1] * max(0, at - nxt))
        nxt = max(nxt, at)
        cap = captures.get(key)
        if cap is None:
            missing += 1
            continue
        ids.extend(range(int(cap[0]), int(cap[1])))
        rep_of.extend([k] * (int(cap[1]) - int(cap[0])))
    ids.extend(range(nxt, hi))
    rep_of.extend([-1] * max(0, hi - nxt))
    return ids, rep_of, missing


def align(records, ids, search=4096, near=64, confirm=4):
    """records [(runtime_id, chip, start, end, freq)] of ONE chip in start order -> [stream position or None]. A record
    matches the next stream position with its low 16 bits (within `search`; a hit further than `near` must be
    confirmed by the next `confirm` records); an unmatched record never moves the pointer (c177 align)."""
    low = [x & 0xFFFF for x in ids]
    pos_of = {}
    for q, v in enumerate(low):
        pos_of.setdefault(v, []).append(q)

    def find(p, lid, limit):
        c = pos_of.get(lid)
        if not c:
            return None
        j = bisect.bisect_left(c, p)
        return c[j] if j < len(c) and c[j] < p + limit else None

    out, p = [], 0
    for k, r in enumerate(records):
        q = find(p, r[0] & 0xFFFF, search)
        if q is not None and q - p > near:
            pp = q + 1
            for r2 in records[k + 1:k + 1 + confirm]:
                q2 = find(pp, r2[0] & 0xFFFF, near)
                if q2 is None:
                    q = None
                    break
                pp = q2 + 1
        if q is None:
            out.append(None)
            continue
        out.append(q)
        p = q + 1
    return out


def mtp_classify(phase, scope):
    """(phase, scope path) -> class. V: embed | head | <kind>.<mixer|mlp|layer> (layer = the decoder layer's own
    ops: norms, residual adds, all-reduce outside the mixer/MLP); cd: accept | pre | d<k>; other phases: phase."""
    parts = [p for p in str(scope or "").split("/") if p]
    if parts and parts[0] in ("V", "R", "cd", "cc", "C2", "C"):
        parts = parts[1:]
    if phase == "V":
        if parts and parts[0].startswith("L") and "." in parts[0]:
            kind = parts[0].split(".", 1)[1]
            sub = parts[1] if len(parts) > 1 else "layer"
            return f"{kind}.{'mixer' if sub in ('gdn', 'attn') else sub}"
        return parts[0] if parts else "V.other"
    if phase in ("cd", "C2", "C"):
        return parts[0] if parts else "pre"
    return phase


def align_window(records, ids, lead=64):
    """align() for a window: a record of the steps BEFORE the window that is delivered after the callback was registered
    matches the stream within `near` (the same traces replay every step) and shifts every later record by one replay
    (test_mtp_optrace). So the alignment also starts at each of the first `lead` records whose low 16 bits equal the
    stream's first id, the records before it left unmatched; the start that matches the most records wins."""
    if not records or not ids:
        return [None] * len(records)
    first = ids[0] & 0xFFFF
    starts = [0] + [j for j in range(1, min(lead, len(records))) if records[j][0] & 0xFFFF == first]
    best = None
    for j in starts:
        pos = [None] * j + align(records[j:], ids)
        n = sum(1 for q in pos if q is not None)
        if best is None or n > best[0]:
            best = (n, pos)
    return best[1]


def analyse_window(win_rows, captures, cap_rows, owner_cap, phase_of, recs, dropped, nsteps):
    """One QWEN36_MTP_OPTRACE window -> {records, chips: {coverage, clock}, tables: {chip: {phase, class, ops}}}: every
    realtime-profiler record aligned to the window's replay stream, named by its op row (capture rows for replayed
    programs, window rows for eager ones), classed by mtp_classify, summed per step. Spans are first-start to last-end
    per replay (per phase, and per phase:class), so span - busy is the device idle/dispatch gap inside it."""
    nw = len(win_rows)
    owner_win = owner_index([r for r in win_rows])
    ids, rep_of, missing = expected_stream(win_rows, captures)
    chips = sorted({r[1] for r in recs})
    out = {"records": len(recs), "dropped": dropped, "replays_without_capture": missing, "chips": {}}
    tables = {}
    for chip in chips:
        rs = sorted((r for r in recs if r[1] == chip), key=lambda r: r[2])
        pos = align_window(rs, ids)
        tot = um = 0.0
        mhz = []
        ops = {}
        span = {}  # (replay, phase, class) -> [start, end, busy]
        for r, q in zip(rs, pos):
            f = _mhz(r[4]) or 1350.0
            mhz.append(f)
            us = (r[3] - r[2]) / f
            tot += us
            if q is None:
                um += us
                continue
            full, k = ids[q], rep_of[q]
            if k >= 0:
                tkey = str(win_rows[k][4])[len("trace="):]
                ph = phase_of.get(tkey, (None, None, "trace:" + tkey))[2]
                ri = owner_cap.get(full)
                row = cap_rows[ri] if ri is not None else None
            else:
                ph = "eager"
                ri = owner_win.get(full)
                row = win_rows[ri] if ri is not None and ri < nw else None
            name = row[2] if row is not None else "?"
            if name == "<scope>":
                name = "(unnamed)"
            if name.endswith("generic_op") and row is not None and str(row[4]).startswith("in=["):
                name = f"{name}{str(row[4]).split(';', 1)[0][len('in='):]}"  # which custom program (its io list)
            cl = mtp_classify(ph, row[3] if row is not None else "")
            o = ops.setdefault((ph, cl, name), [0.0, 0, []])
            o[0] += us
            o[1] += 1
            o[2].append(us)
            for sk in ((k, ph, None), (k, ph, cl)):
                sp = span.setdefault(sk, [r[2], r[3], 0.0])
                sp[0], sp[1], sp[2] = min(sp[0], r[2]), max(sp[1], r[3]), sp[2] + us
        f = sum(mhz) / len(mhz) if mhz else 1350.0
        sp_ph, sp_cl = {}, {}
        for (k, ph, cl), (a, b, busy) in span.items():
            d = sp_ph if cl is None else sp_cl
            e = d.setdefault(ph if cl is None else f"{ph}:{cl}", [0.0, 0.0, 0])
            e[0] += (b - a) / f
            e[1] += busy
            e[2] += 1
        per = float(max(1, nsteps))
        # Device idle between consecutive replays (prev phase end -> next phase start), labelled "prev>next":
        # the per-boundary gap of the step (cc>V is the gap the host's next-step work leaves on the device).
        reps = sorted(((a, b, ph) for (k, ph, cl), (a, b, _busy) in span.items() if cl is None and k >= 0))
        gaps = {}
        for (a0, b0, p0), (a1, b1, p1) in zip(reps, reps[1:]):
            g = gaps.setdefault(f"{p0}>{p1}", [0.0, 0, []])
            us = max(0.0, (a1 - b0) / f)
            g[0] += us
            g[1] += 1
            g[2].append(us)
        out["chips"][chip] = {"records": len(rs), "matched": sum(1 for q in pos if q is not None),
                              "device_us": round(tot, 1), "unmatched_us": round(um, 1),
                              "coverage": round((tot - um) / tot, 4) if tot else None,
                              "mhz_mean": round(f, 1), "mhz_min": round(min(mhz), 1) if mhz else None}
        tables[chip] = {
            "phase": {p: {"span_us": round(v[0] / per, 1), "busy_us": round(v[1] / per, 1),
                          "replays_per_step": round(v[2] / per, 3)} for p, v in sorted(sp_ph.items())},
            "class": {k: {"span_us": round(v[0] / per, 1), "busy_us": round(v[1] / per, 1)}
                      for k, v in sorted(sp_cl.items())},
            "gaps": {k: {"us_per_step": round(v[0] / per, 1), "n": v[1], "us_med": round(sorted(v[2])[len(v[2]) // 2], 1),
                         "us_max": round(max(v[2]), 1)} for k, v in sorted(gaps.items())},
            "ops": [{"phase": ph, "class": cl, "op": nm, "us": round(v[0] / per, 2),
                     "programs": round(v[1] / per, 2), "us_med": round(sorted(v[2])[len(v[2]) // 2], 2)}
                    for (ph, cl, nm), v in sorted(ops.items(), key=lambda kv: -kv[1][0])[:120]],
        }
    first = chips[0] if chips else None
    out["tables"] = {c: (tables[c] if c == first else {"phase": tables[c]["phase"], "class": tables[c]["class"],
                                                       "gaps": tables[c]["gaps"]})
                     for c in tables}
    return out
