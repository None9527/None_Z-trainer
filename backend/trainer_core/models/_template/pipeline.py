# -*- coding: utf-8 -*-
"""
[MODEL_NAME] Inference Pipeline

Wraps model inference for image generation.
"""


class ModelPipeline:
    """Inference pipeline for [MODEL_NAME]."""

    def __init__(self, model_path: str, **kwargs):
        raise NotImplementedError("Implement inference pipeline for your model")

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError("Implement generation for your model")
