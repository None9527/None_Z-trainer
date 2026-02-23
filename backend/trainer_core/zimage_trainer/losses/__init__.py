# -*- coding: utf-8 -*-
"""
Z-Image Losses — DEPRECATED

All losses have been moved to shared/losses/.
This file re-exports for backward compatibility.
Import from shared.losses directly for new code.
"""

# Re-export everything from shared/losses for backward compatibility
from shared.losses import *  # noqa: F401,F403
from shared.losses import __all__  # noqa: F401
