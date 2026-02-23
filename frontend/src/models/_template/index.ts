/**
 * Model Template — copy this directory to create a new model.
 * 
 * Steps:
 * 1. Copy _template/ to newmodel/
 * 2. Edit index.ts: set id, name, capabilities, default configs
 * 3. Edit TrainingParams.vue: model-specific training parameters
 * 4. Edit GenerationParams.vue: model-specific generation parameters
 * 5. Edit CacheConfig.vue: model-specific cache configuration
 * 6. Add `import './newmodel'` to models/index.ts
 */

import { registry } from '../registry'
import type { ModelDefinition } from '../types'
import TrainingParams from './TrainingParams.vue'
import GenerationParams from './GenerationParams.vue'
import CacheConfig from './CacheConfig.vue'

const templateModel: ModelDefinition = {
    id: 'template',
    name: 'Template Model',
    icon: '📋',

    capabilities: {
        lora: true,
        finetune: false,
        controlnet: false,
        img2img: false,
        inpainting: false
    },

    cache: {
        fileSuffix: '_tpl',
        supportedTypes: ['latent', 'text'],
        configComponent: CacheConfig
    },

    trainingParamsComponent: TrainingParams,
    generationParamsComponent: GenerationParams,

    defaultTrainingConfig: {},
    defaultGenerationConfig: {
        prompt: '',
        steps: 20,
        guidance_scale: 7.5,
        seed: -1,
        width: 1024,
        height: 1024
    }
}

// DO NOT uncomment — this is a template only
// registry.register(templateModel)

export default templateModel
