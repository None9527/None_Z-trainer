import { registry } from '../registry'
import type { ModelDefinition } from '../types'
import TrainingParams from './TrainingParams.vue'
import GenerationParams from './GenerationParams.vue'
import CacheConfig from './CacheConfig.vue'

const zimageModel: ModelDefinition = {
    id: 'zimage',
    name: 'Z-Image',
    icon: '🎨',

    capabilities: {
        lora: true,
        finetune: true,
        controlnet: true,
        img2img: true,
        inpainting: false
    },

    cache: {
        fileSuffix: '_zi',
        supportedTypes: ['latent', 'text', 'control', 'siglip'],
        configComponent: CacheConfig
    },

    trainingParamsComponent: TrainingParams,
    generationParamsComponent: GenerationParams,

    defaultTrainingConfig: {
        timestep: {
            mode: 'uniform',
            shift: 3.0,
            use_dynamic_shift: true,
            base_shift: 0.5,
            max_shift: 1.15,
            logit_mean: 0.0,
            logit_std: 1.0,
            acrf_steps: 10,
            jitter_scale: 0.02,
            latent_jitter_scale: 0.0,
        },
        acrf: {
            snr_gamma: 5.0,
            snr_floor: 0.1,
            raft_mode: false,
            free_stream_ratio: 0.3,
            enable_timestep_aware_loss: false,
            timestep_high_threshold: 0.7,
            timestep_low_threshold: 0.3,
            enable_curvature: false,
            lambda_curvature: 0.05,
            curvature_interval: 10,
            curvature_start_epoch: 0,
            cfg_training: false,
            cfg_scale: 7.0,
            cfg_training_ratio: 0.3
        }
    },

    defaultGenerationConfig: {
        prompt: 'A futuristic city with flying cars, cyberpunk style, highly detailed, 8k',
        negative_prompt: '',
        steps: 9,
        guidance_scale: 1.0,
        seed: -1,
        width: 1024,
        height: 1024,
        lora_path: null,
        lora_scale: 1.0,
        comparison_mode: false,
        model_type: 'zimage',
        transformer_path: null
    }
}

// Auto-register on import
registry.register(zimageModel)

export default zimageModel
