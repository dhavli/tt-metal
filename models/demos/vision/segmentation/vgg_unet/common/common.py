# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import os

import torch


def load_torch_model(torch_model, model_location_generator=None):
    if model_location_generator == None or (not _tt_public_contains('TT_GH_CI_INFRA')):
        weights_path = "models/demos/vision/segmentation/vgg_unet/common/vgg_unet_torch.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/demos/vision/segmentation/vgg_unet/common/weights_download.sh")
    else:
        weights_path = (
            model_location_generator("vision-models/unet_vgg", model_subdir="", download_if_ci_v2=True)
            / "vgg_unet_torch.pth"
        )

    torch_dict = torch.load(weights_path)
    new_state_dict = dict(zip(torch_model.state_dict().keys(), torch_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    return torch_model
