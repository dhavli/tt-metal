# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warm ttnn weight cache for Qwen3.5/3.6/3.8: build the model from on-disk artifacts alone.

Why this exists
---------------
``ttnn.as_tensor(cache_file_name=...)`` already persists every tilized, mesh-sharded weight as a
``.tensorbin`` and reloads it on the next run, IGNORING the torch tensor it is handed. So on a
warm cache the only thing the HF checkpoint is still needed for is the handful of weights the
loaders consume on the HOST -- everything else is a ``torch.empty`` stand-in whose shape satisfies
the reshape ops that run before ``as_tensor``. tt_transformers ships that machinery
(``models/common/weight_cache.py``: marker + manifest + host-weight sidecar); upstream left it
switched off for this family because the Gated-DeltaNet loader reads conv weights from the state
dict with plain ``ttnn.from_torch`` (no cache file), so a dataless placeholder would silently feed
those layers garbage. This module turns it on by naming exactly those host-consumed weights and
capturing them to the sidecar at cold-build time.

Host-consumed weights in this model tree (anything served from a placeholder here is a bug):

* ``layers.N.linear_attn.conv1d.weight`` (TP loader, raw FP8 remap) -- ``gdn/tp.py`` derives the
  four conv taps AND the host-held ``conv_w1d`` from it. The taps get cache files; ``conv_w1d``
  does not.
* ``layers.N.linear_attn.{q,k,v}_conv.{weight,bias}`` (single-device bf16 remap) --
  ``gdn/weights.py::load_conv_weight`` is a bare ``from_torch``.
* ``layers.N.self_attn.{q,k}_norm.weight`` -- ``attention/tp.py`` replicates them with
  ``cache_path=None`` (it adds the +1 unit offset on host first).

Everything else in the state dict reaches the device through ``as_tensor`` with a cache file:
embeddings, per-layer norms, all projections, the LM head.

Fail-closed rules (deliberate; this deployment controls its own cache directory):

* A warm build is accepted only when the upstream marker for this exact build variant exists,
  every tensorbin it recorded is present, and the sidecar loads.
* The cache directory carries a ``.cache_identity.json`` naming the checkpoint (content hashes of
  ``config.json`` and the safetensors index) and the ttnn build that wrote it. A mismatch with the
  running configuration is a hard error, cold OR warm: ``as_tensor`` would otherwise reload the old
  tensorbins for a new checkpoint or a new ttnn, which is precisely the silent-wrong-weights hazard.
  ``QWEN36_WEIGHT_CACHE_ACCEPT_MISMATCH=1`` overrides, for someone who knows why.
* A warm build must not create a single new tensorbin. If it does, the build needed a file the
  marker never recorded, the file was generated FROM A PLACEHOLDER, and it is deleted and the run
  aborted rather than left on disk as a permanent poisoned cache entry.
* Placeholders are NaN-filled (small tensors) so that any placeholder that does leak to the device
  produces NaN output -- loud -- instead of plausible token soup.
* During a warm build ``ttnn.from_torch`` is wrapped: a floating host tensor containing NaN reaching
  it means a placeholder (or something derived from one) is being pushed to the device outside the
  cache-hit path, i.e. a host-consumed weight this module does not know about. The build aborts
  naming the shape. Real weights never contain NaN, so there are no false positives. This is the
  check that catches a NEW host-consumed weight appearing in a later tt-metal, without trusting the
  suffix list above -- the list decides what goes into the sidecar, the guard proves nothing else
  was needed.

How the list was enumerated: every ``ttnn.from_torch`` call and every ``as_tensor`` call with
``cache_file_name=None`` in ``models/demos/blackhole/qwen36/tt`` that takes a state_dict value (or
something derived from one) as input, at ``v0.79.0-dev20260903``; then confirmed at runtime by the
guard above on a full 64-layer warm build (80 tensors: 48 GDN ``conv1d.weight`` + 16 x 2 attention
norms for the 27B). Re-run that audit -- or simply a warm build, which fails loudly -- on every pin bump.

``TT_TRANSFORMERS_FORCE_MODEL_LOAD=1`` forces the cold path (reads the checkpoint, reuses existing
tensorbins, rewrites marker + sidecar). Note it does NOT regenerate tensorbins; for that, delete the
cache directory.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from loguru import logger

from models.common.weight_cache import CachedStateDict, list_cache_files, load_host_sidecar, marker_path
from models.common.weight_cache import mark_weight_cache_complete as _mark_complete
from models.common.weight_cache import weight_cache_is_complete as _is_complete

IDENTITY_FILE = ".cache_identity.json"
ACCEPT_MISMATCH_ENV = "QWEN36_WEIGHT_CACHE_ACCEPT_MISMATCH"
# Placeholders at or below this many elements are NaN-filled; larger ones stay torch.empty. Reading
# untouched anonymous memory maps the kernel zero page, so an un-filled 2.5 GB embedding stand-in
# costs nothing until its `.T.contiguous()` copy -- filling it would double that transient.
NAN_FILL_MAX_NUMEL = 64 * 1024 * 1024

_HOST_WEIGHT_SUFFIXES = (
    "linear_attn.conv1d.weight",
    "linear_attn.conv1d.bias",
    "linear_attn.q_conv.weight",
    "linear_attn.k_conv.weight",
    "linear_attn.v_conv.weight",
    "linear_attn.q_conv.bias",
    "linear_attn.k_conv.bias",
    "linear_attn.v_conv.bias",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)


def is_host_weight(key: str) -> bool:
    """True for the weights the qwen36 loaders consume on the host (see module docstring)."""
    return key.endswith(_HOST_WEIGHT_SUFFIXES)


# ---------------------------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------------------------


def _sha256_file(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def ttnn_build_id():
    """The ttnn wheel version, which for the vendor containers embeds the tt-metal commit
    (``0.1.dev1+gd2f4b3a``). A tensorbin written by one build is not promised to be read the same by
    another, so this is part of the cache identity."""
    for probe in (lambda: importlib.metadata.version("ttnn"), lambda: __import__("ttnn").__version__):
        try:
            v = probe()
            if v:
                return str(v)
        except Exception:
            continue
    return "unknown"


def checkpoint_identity(ckpt_dir):
    """Content identity of the checkpoint: hashes of config.json and the safetensors index, plus the
    resolved directory name (a HF snapshot hash when the checkpoint lives in the hub cache)."""
    ckpt = Path(ckpt_dir)
    ident = {"dir": Path(os.path.realpath(ckpt)).name}
    for name in ("config.json", "model.safetensors.index.json"):
        p = ckpt / name
        ident[name] = _sha256_file(p) if p.is_file() else None
    return ident


def build_variant(args):
    """Every knob that changes WHICH cache files the qwen36 loaders ask for, so a marker seeded under
    one variant never certifies a build needing different files. Mirrors the gates in
    attention/tp.py, gdn/tp.py, mlp.py and model.py -- if a new `.suffix` cache name is added there,
    add its gate here."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    tp = int(getattr(args, "num_devices", 1))
    return {
        "loader": "qwen36",
        "tp": tp,
        "weight_dtype": str(getattr(args, "weight_dtype", None)),
        "proj_1d_decode": bool(getattr(args, "proj_1d_decode", False)),
        "mlp_1d_decode": bool(getattr(args, "mlp_1d_decode", False)),
        "mlp_down_bf4": _tt_public_getenv("QWEN36_MLP_DOWN_BF4") == "1",
        "gdn_qkvz_sharded": getattr(args, "gdn_qkvz_weight_memcfg", None) is not None,
        "gdn_out_sharded": getattr(args, "gdn_out_weight_memcfg", None) is not None,
        "attn_qkv_fused": getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None,
        "attn_qg_sharded": getattr(args, "attn_qg_weight_memcfg", None) is not None,
        "attn_wo_sharded": getattr(args, "attn_wo_weight_memcfg", None) is not None,
        "mlp_w1_sharded": getattr(args, "mlp_w1_weight_memcfg", None) is not None,
        "mlp_gateup_agmm": bool(tpc.mlp_gateup_agmm_enabled(tp)) if tp > 1 else False,
        "kv_replication": bool(getattr(args, "kv_replication", False)),
        "layer_indices": list(getattr(args, "layer_indices", None) or []),
    }


def _identity(args, mesh_device):
    return dict(
        model_name=args.model_name,
        n_layers=int(args.n_layers),
        mesh_shape=tuple(int(d) for d in mesh_device.shape),
        components="text",
        build_variant=build_variant(args),
    )


def _write_identity(cache_path, args, mesh_device):
    body = {
        "checkpoint": checkpoint_identity(args.CKPT_DIR),
        "ttnn": ttnn_build_id(),
        "mesh_shape": list(int(d) for d in mesh_device.shape),
        "model_name": args.model_name,
    }
    tmp = Path(cache_path) / f"{IDENTITY_FILE}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
    os.replace(tmp, Path(cache_path) / IDENTITY_FILE)
    return body


def check_identity(cache_path, args, mesh_device):
    """Refuse to (re)use a cache directory written for a different checkpoint or ttnn build.

    Runs before BOTH the warm and the cold path: on a cold path ``as_tensor`` reloads whatever
    tensorbins already exist, so a stale directory is just as wrong when the checkpoint is being
    read as when it is not. Returns the on-disk identity, or None when the directory has none yet."""
    cache_path = Path(cache_path)
    ident_file = cache_path / IDENTITY_FILE
    have_tensorbins = bool(list_cache_files(cache_path)) if cache_path.is_dir() else False
    if not ident_file.is_file():
        if have_tensorbins:
            logger.warning(
                f"Weight cache {cache_path} holds tensorbins but no {IDENTITY_FILE}: it predates the identity "
                f"check, so nothing proves it was built from {args.CKPT_DIR}. It will be adopted and "
                f"stamped when this build completes."
            )
        return None
    on_disk = json.loads(ident_file.read_text())
    now = {
        "checkpoint": checkpoint_identity(args.CKPT_DIR),
        "ttnn": ttnn_build_id(),
        "mesh_shape": list(int(d) for d in mesh_device.shape),
    }
    mismatches = {k: (on_disk.get(k), now[k]) for k in now if on_disk.get(k) != now[k]}
    if mismatches:
        msg = (
            f"Weight cache {cache_path} was built for a different configuration; refusing to reuse its "
            f"tensorbins. Mismatches (on disk -> now): {json.dumps(mismatches, default=str)}. Point "
            f"TT_CACHE_PATH at a fresh directory (or delete this one) to rebuild cold. Set "
            f"{ACCEPT_MISMATCH_ENV}=1 only if you know the tensorbins are valid for this build."
        )
        if _tt_public_getenv(ACCEPT_MISMATCH_ENV) == "1":
            logger.warning(msg + " (accepted by env override)")
        else:
            raise RuntimeError(msg)
    return on_disk


# ---------------------------------------------------------------------------------------------
# State dicts
# ---------------------------------------------------------------------------------------------


class Qwen36CachedStateDict(CachedStateDict):
    """Upstream hybrid stand-in (real host weights, dataless placeholders elsewhere) with two
    qwen36 additions: NaN-filled small placeholders so a leak is loud, and a ``substate`` that
    filters on KEYS instead of materializing every value (the tree calls it once per layer)."""

    def __getitem__(self, key):
        t = super().__getitem__(key)
        if key in self._host or key in self._overrides:
            return t
        if t.numel() <= NAN_FILL_MAX_NUMEL and t.is_floating_point():
            t.fill_(float("nan"))
        return t

    def substate(self, prefix):
        p = prefix + "."
        return {k[len(p) :]: self[k] for k in self if k.startswith(p)}


def substate(state_dict, prefix):
    """``models.demos.blackhole.qwen36.utils.substate.substate`` for plain dicts; the key-filtering
    method for the lazy / cached mappings (which must not have every value touched per layer)."""
    if hasattr(state_dict, "substate"):
        return state_dict.substate(prefix)
    p = prefix + "."
    return {k[len(p) :]: v for k, v in state_dict.items() if k.startswith(p)}


def _manifest_dict(state_dict):
    """What ``mark_weight_cache_complete`` needs: shape/dtype per key, real tensors for host weights.
    A lazy loader exposes ``manifest()`` so this does not dequantize the whole checkpoint again."""
    if hasattr(state_dict, "manifest"):
        return state_dict.manifest(is_host_weight)
    return state_dict


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


class _PlaceholderLeakGuard:
    """Context manager wrapping ``ttnn.from_torch`` for the duration of a warm build (see module
    docstring). Also counts calls so the log can say how many host tensors were pushed."""

    def __init__(self):
        import ttnn

        self._ttnn = ttnn
        self._orig = ttnn.from_torch
        self.calls = 0

    def __enter__(self):
        orig = self._orig

        def guarded(tensor, *a, **kw):
            self.calls += 1
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.numel() > 0:
                if bool(torch.isnan(tensor).any()):
                    raise RuntimeError(
                        f"Warm-cache build: a NaN-filled placeholder of shape {tuple(tensor.shape)} reached "
                        f"ttnn.from_torch. A weight is consumed on the host that weight_cache.py does not list "
                        f"as host-consumed (see _HOST_WEIGHT_SUFFIXES). Refusing to build silent garbage; run "
                        f"cold (TT_TRANSFORMERS_FORCE_MODEL_LOAD=1) and add the key to the list."
                    )
            return orig(tensor, *a, **kw)

        self._ttnn.from_torch = guarded
        return self

    def __exit__(self, *exc):
        self._ttnn.from_torch = self._orig
        return False


def build_model(mesh_device, args, model_cls, cache_path):
    """Build ``model_cls(mesh_device, args, state_dict, tensor_cache_path=cache_path)`` from the warm
    cache when it is complete for this build, else from the checkpoint (and seal the cache after).

    Returns ``(model, state_dict, warm)``. ``state_dict`` is the hybrid stand-in on a warm build."""
    cache_path = Path(cache_path)
    force = _tt_public_getenv("TT_TRANSFORMERS_FORCE_MODEL_LOAD") == "1"
    ident = _identity(args, mesh_device)
    check_identity(cache_path, args, mesh_device)
    # A layer subset (profiling) writes a partial tensorbin set that must never be sealed as complete.
    # Compare against the checkpoint's own layer count: create_tt_model truncates attention_type_list
    # alongside n_layers, so the two agree even when the model is truncated.
    full_layers = int(getattr(args.hf_config.get_text_config(), "num_hidden_layers", args.n_layers))
    partial = bool(getattr(args, "layer_indices", None)) or int(args.n_layers) != full_layers

    warm = (not force) and (not partial) and _is_complete(cache_path, **ident)
    if warm:
        logger.info(f"Warm ttnn weight cache at {cache_path}: building from tensorbins + host sidecar, no HF load.")
        before = set(list_cache_files(cache_path))
        meta = json.loads(marker_path(cache_path, ident["build_variant"]).read_text())
        host = load_host_sidecar(cache_path, consume=True) or {}
        state_dict = Qwen36CachedStateDict(meta["weights"], host)
        missing_host = [k for k in state_dict if is_host_weight(k) and k not in host]
        if missing_host:
            raise RuntimeError(
                f"Warm cache sidecar lacks {len(missing_host)} host-consumed weights (e.g. {missing_host[0]}); "
                f"the cache was sealed by an older predicate. Rebuild with TT_TRANSFORMERS_FORCE_MODEL_LOAD=1."
            )
        for k, t in host.items():
            if not torch.isfinite(t.float()).all() or (t.numel() > 0 and float(t.float().abs().max()) == 0.0):
                raise RuntimeError(f"Warm cache sidecar weight {k} is non-finite or all-zero; refusing to serve.")
        logger.info(f"Warm state_dict: {len(meta['weights'])} weights, {len(host)} real host weights from the sidecar.")
        with _PlaceholderLeakGuard() as guard:
            model = model_cls(mesh_device, args, state_dict, tensor_cache_path=cache_path)
        logger.info(f"Warm build: {guard.calls} ttnn.from_torch calls, none carried a placeholder.")
        leaked = sorted(set(list_cache_files(cache_path)) - before)
        if leaked:
            for f in leaked:
                try:
                    (cache_path / f).unlink()
                except OSError:
                    pass
            raise RuntimeError(
                f"Warm build generated {len(leaked)} new tensorbins from placeholders (e.g. {leaked[0]}); they were "
                f"deleted. The marker did not cover this build -- run cold with TT_TRANSFORMERS_FORCE_MODEL_LOAD=1."
            )
        return model, state_dict, True

    if force:
        logger.info("TT_TRANSFORMERS_FORCE_MODEL_LOAD=1: cold load requested.")
    logger.info("Cold build: loading the checkpoint via Qwen36ModelArgs.load_state_dict() ...")
    state_dict = args.load_state_dict()
    model = model_cls(mesh_device, args, state_dict, tensor_cache_path=cache_path)
    if partial:
        logger.info("Partial model (layer subset): not sealing the weight cache.")
        return model, state_dict, False
    _mark_complete(cache_path, _manifest_dict(state_dict), is_host_weight=is_host_weight, **ident)
    _write_identity(cache_path, args, mesh_device)
    logger.info(f"Sealed weight cache at {cache_path}; the next start is warm.")
    return model, state_dict, False
