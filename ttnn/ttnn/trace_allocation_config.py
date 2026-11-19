# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Process-start configuration for trace allocation tracking."""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os
import warnings


def _env_enabled(name: str) -> bool:
    return _tt_public_getenv(name) == "1"


def _env_nonnegative_int(name: str, default: int) -> int:
    value = _tt_public_getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        warnings.warn(f"{name} must be a non-negative integer; using {default}", stacklevel=2)
        return default


# These values are intentionally captured once, when ttnn starts importing.
TRACE_ALLOC_TRACKING = _env_enabled("TT_METAL_TRACE_ALLOC_TRACKING")
TRACE_ALLOC_DIAGNOSTICS = TRACE_ALLOC_TRACKING and _env_enabled("TT_METAL_TRACE_ALLOC_TRACEBACKS")
TRACE_ALLOC_REFERRER_DEPTH = (
    _env_nonnegative_int("TT_METAL_TRACE_ALLOC_REFERRER_DEPTH", 10) if TRACE_ALLOC_DIAGNOSTICS else 10
)
