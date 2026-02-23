# -*- coding: utf-8 -*-
"""
Model Packages - Convention-based per-model implementations

Each model subdirectory follows the same convention:
    loader.py   - Load model components (transformer, VAE, text encoder, scheduler)
    lora.py     - Model-specific LoRA creation and injection
    sampler.py  - Timestep sampling strategy (anchor, shift, uniform, etc.)
    forward.py  - Forward pass (input format conversion + output handling)
    pipeline.py - Inference pipeline
    train.py    - Training loop (assembles shared + model-specific components)

To add a new model, copy _template/ and implement each file.
"""
