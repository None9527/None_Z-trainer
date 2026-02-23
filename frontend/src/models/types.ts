import type { Component } from 'vue'

/**
 * Model capability flags — determines which generic UI sections to show
 */
export interface ModelCapabilities {
    lora: boolean
    finetune: boolean
    controlnet: boolean
    img2img: boolean
    inpainting: boolean
}

/**
 * Cache system configuration — each model has its own cache format/types
 */
export interface CacheDefinition {
    /** File suffix for cache files, e.g. "_zi" → _zi_latent.safetensors */
    fileSuffix: string
    /** Supported cache types, e.g. ["latent", "text", "control", "siglip"] */
    supportedTypes: string[]
    /** Model-specific cache configuration panel component */
    configComponent: Component
}

/**
 * Core model definition — register one per model family
 */
export interface ModelDefinition {
    /** Unique ID matching backend trainer_core/models/<id>/ */
    id: string
    /** Display name */
    name: string
    /** Emoji icon */
    icon: string

    capabilities: ModelCapabilities
    cache: CacheDefinition

    /** Model-specific training parameters panel (collapse section) */
    trainingParamsComponent: Component
    /** Model-specific generation parameters panel */
    generationParamsComponent: Component

    /** Default training config values (merged into global config) */
    defaultTrainingConfig: Record<string, any>
    /** Default generation config values */
    defaultGenerationConfig: Record<string, any>
}
