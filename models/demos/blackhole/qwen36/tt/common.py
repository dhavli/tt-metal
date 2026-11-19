# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model construction entry point (gpt_oss/gemma4 `create_tt_model` convention)."""
import os

from loguru import logger

from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.weight_cache import build_model


def create_tt_model(
    mesh_device,
    max_batch_size=1,
    max_seq_len=2048,
    n_layers=None,
    layer_indices=None,
    hf_model=None,
):
    """Build the Qwen3.5-9B model. Returns (args, model, state_dict).

    HF_MODEL (env var) is the single source of truth. `hf_model`, if given, sets it.
    `layer_indices` runs ONLY the listed checkpoint layers (profiling); it takes precedence
    over `n_layers` (first-N truncation). See Qwen36Model.from_pretrained for details.
    """
    if hf_model is not None:
        os.environ["HF_MODEL"] = hf_model

    args = Qwen36ModelArgs(
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if layer_indices is not None:
        layer_indices = list(layer_indices)
        assert layer_indices, "layer_indices must be non-empty"
        assert all(
            0 <= i < len(args.attention_type_list) for i in layer_indices
        ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
        args.layer_indices = layer_indices
        args.n_layers = len(layer_indices)
    elif n_layers is not None:
        args.n_layers = n_layers
        args.attention_type_list = args.attention_type_list[:n_layers]

    # Warm-cache build (no checkpoint read) when the sealed ttnn cache covers this build; cold
    # otherwise. The host-consumed weights that made upstream disable this for qwen3.6 (GDN conv
    # taps, attention q/k norms) are served real from the sidecar -- see tt/weight_cache.py.
    cache_path = args.weight_cache_path()
    model, state_dict, _warm = build_model(mesh_device, args, Qwen36Model, cache_path)

    return args, model, state_dict
