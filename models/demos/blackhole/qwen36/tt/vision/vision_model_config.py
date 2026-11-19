# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

from loguru import logger

import ttnn
from models.demos.qwen3_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.model_config import ModelArgs


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = False
        # self.bfp4_mlp = "Qwen3-VL-32B" in model_name


class VisionModelArgs(ModelArgs):
    # Base __init__ checks the TEXT config's 4 KV heads; the vision tower's own 16 MHA heads
    # (set below) shard exactly at TP=8, so only the base check needs relaxing.
    SUPPORTS_KV_REPLICATION = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The vision tower is always tensor-parallel (Megatron-style): the
        # vision blocks shard their weights across the mesh devices along
        # cluster axis 1.

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.unpadded_hidden_dim, self.tile_size * self.num_devices
        )
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        self.n_heads = self.hf_config.vision_config.num_heads
        self.n_kv_heads = self.hf_config.vision_config.num_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        self.optimizations = ModelOptimizations(
            self.model_name
        )  # todo)) implement finer grained control similar to tt_transformers'

        num_rows = lambda seq_len: min(seq_len, 1024 if self.is_galaxy else 2048)
        k_dim = self.dim // self.cluster_shape[0] if self.is_galaxy else self.dim
        n_dim = self.dim // self.cluster_shape[1] if self.is_galaxy else self.dim
        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=1 if self.is_galaxy else self.dim // 1024,
            fuse_batch=seq_len <= 1024,
        )

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

        # Sanity-check the divisibility requirements that the TP code relies on.
        tp = self.cluster_shape[1]
        assert self.n_heads % tp == 0, f"vision n_heads ({self.n_heads}) must be divisible by TP={tp}"
        assert self.qkv_size % tp == 0, f"vision qkv_size ({self.qkv_size}) must be divisible by TP={tp}"
        assert self.dim % tp == 0, f"vision dim ({self.dim}) must be divisible by TP={tp}"
        assert self.hidden_dim % tp == 0, f"vision hidden_dim ({self.hidden_dim}) must be divisible by TP={tp}"
        # PatchMerger shards the merger MLP Megatron-style; its post-shuffle
        # inner dim (mlp_size = hidden * spatial_merge_size^2) and the final
        # out_hidden_size must both divide cleanly.
        vision_cfg = self.hf_config.vision_config
        mlp_size = vision_cfg.hidden_size * (vision_cfg.spatial_merge_size**2)
        out_hidden_size = vision_cfg.out_hidden_size
        assert mlp_size % tp == 0, f"vision merger mlp_size ({mlp_size}) must be divisible by TP={tp}"
        assert out_hidden_size % tp == 0, f"vision out_hidden_size ({out_hidden_size}) must be divisible by TP={tp}"

    def prepare_residual_tensor_prefill(self, x_bsh):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim

        The vision blocks consume tensors fractured along the hidden dim
        (dim=3 of the 4D tensor), so we shard at load time across cluster
        axis 1.
        """

        x_1BSH = x_bsh.unsqueeze(0)

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            dims=(None, -1),
            mesh_shape=self.cluster_shape,
        )

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num=None, deepstack_merger_num=None):
        layer_prefix = f"visual.blocks.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "VisionAttention": "attention",
            "VisionBlock": "",
            "VisionTransformer": "visual",
            "PatchMerger": "visual.merger",
            "norm1": "norm1",
            "norm2": "norm2",
            "DeepstackMerger": f"visual.deepstack_merger_list.{deepstack_merger_num}",
            "": "",  # If no module is given, just get layer prefix
        }
        return layer_prefix + module_map[module_name]

    def reference_vision_model(self, depth=None):
        """The HF vision tower, built standalone from its own ``vision_config`` and loaded from the
        ``model.visual.*`` safetensors entries only.

        The previous form -- ``Qwen3_5ForConditionalGeneration.from_pretrained(...).model.visual`` --
        instantiated the WHOLE checkpoint (27B text tower included, dequantized to bf16 on a CPU
        host) to reach a ~1 GB submodule, and was the largest single host-RAM consumer of model
        init: 111 GB of virtual memory and a guaranteed OOM on a 16 GB host. The tower's runtime
        dependencies on the reference module are ``patch_embed`` and ``fast_pos_embed_interpolate``
        (plus ``.dtype``), all satisfied by the standalone module."""
        import torch
        from safetensors import safe_open
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5VisionModel

        config = Qwen3_5Config.from_pretrained(self.CKPT_DIR)
        vcfg = config.vision_config
        if depth is not None:
            vcfg.depth = depth
        dtype = getattr(vcfg, "dtype", None) or getattr(config, "dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        logger.info(f"Loading Qwen3.5 vision tower standalone ({vcfg.depth} blocks, {dtype}) from model.visual.*")
        with torch.device("cpu"):
            model = Qwen3_5VisionModel._from_config(vcfg, dtype=dtype)

        prefix = "model.visual."
        ckpt = self.CKPT_DIR
        index = os.path.join(ckpt, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted({fn for k, fn in weight_map.items() if k.startswith(prefix)})
        else:
            files = sorted(fn for fn in os.listdir(ckpt) if fn.endswith(".safetensors"))
        keep_blocks = {f"blocks.{i}." for i in range(vcfg.depth)}
        state = {}
        for fn in files:
            with safe_open(os.path.join(ckpt, fn), framework="pt") as sf:
                for k in sf.keys():
                    if not k.startswith(prefix):
                        continue
                    rel = k[len(prefix) :]
                    if rel.startswith("blocks.") and not any(rel.startswith(b) for b in keep_blocks):
                        continue  # depth-truncated tower (tests)
                    state[rel] = sf.get_tensor(k).to(dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # inv_freq is a non-persistent buffer computed in __init__; nothing else may be missing.
        missing = [m for m in missing if not m.endswith("inv_freq")]
        if missing or unexpected:
            raise RuntimeError(f"vision tower state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        model.eval()
        return model

    def reference_vision_block(self, layer_num=0):
        return self.reference_vision_model().blocks[layer_num]

    def reference_mlp(self):
        return self.reference_vision_block().mlp

    def reference_attention(self):
        return self.reference_vision_block().attn

    def reference_rms_norm(self):
        return self.reference_vision_block().norm2

    def reference_patch_merger(self):
        return self.reference_vision_model().merger

    def reference_patch_embed(self):
        return self.reference_vision_model().patch_embed
