# -*- coding: utf-8 -*-
"""
Model Template Package

Copy this directory to create a new model adapter.
Implement each file following the same convention as zimage/.

Required files:
    loader.py   - Load model components
    lora.py     - LoRA creation and injection
    sampler.py  - Timestep sampling strategy
    forward.py  - Forward pass (input format + output handling)
    pipeline.py - Inference pipeline
    train.py    - Training loop assembly
"""
