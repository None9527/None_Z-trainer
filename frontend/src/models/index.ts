// Import all model registrations here.
// Each model's index.ts auto-registers itself into the global registry.

import './zimage'

// To add a new model:
// 1. Create src/models/newmodel/ directory
// 2. Add TrainingParams.vue, GenerationParams.vue, CacheConfig.vue
// 3. Add index.ts with ModelDefinition and registry.register()
// 4. Add import below:
// import './newmodel'

export { registry } from './registry'
export type { ModelDefinition, ModelCapabilities, CacheDefinition } from './types'
