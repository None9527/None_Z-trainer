# -*- coding: utf-8 -*-
"""
Shared Components — Model-agnostic, reusable across all models

Modules:
    flow_matching/  - Flow matching paradigm (noising, samplers, loss weighting)
    losses/         - Loss functions (MSE, Charbonnier, Cosine, Frequency, Style, DPO)
    optimizers/     - Custom optimizers (AdamW FP8, AdamW BF16)
    utils/          - Training utils, hardware detector, block swapper, model hooks, degradation
    snr.py          - Signal-to-noise ratio weighting utilities
    gradient.py     - Gradient clipping utilities
    memory.py       - GPU memory management and module offloading
"""
