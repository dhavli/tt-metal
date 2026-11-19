# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import os


def get_model_path(model_location_generator=None):
    if model_location_generator is None or (not _tt_public_contains('TT_GH_CI_INFRA')):
        return "lerobot/smolvla_base"
    else:
        return str(model_location_generator("vision-models/smolvla_base", model_subdir="", download_if_ci_v2=True))
