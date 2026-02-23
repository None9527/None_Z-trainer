import type { ModelDefinition } from './types'

/**
 * Global model registry — singleton pattern.
 * Models self-register via `registry.register(definition)`.
 */
class ModelRegistry {
    private models = new Map<string, ModelDefinition>()

    register(definition: ModelDefinition): void {
        if (this.models.has(definition.id)) {
            console.warn(`[ModelRegistry] Overwriting model: ${definition.id}`)
        }
        this.models.set(definition.id, definition)
        console.log(`[ModelRegistry] Registered: ${definition.id} (${definition.name})`)
    }

    unregister(id: string): void {
        this.models.delete(id)
    }

    get(id: string): ModelDefinition | undefined {
        return this.models.get(id)
    }

    /**
     * Get model or throw — use when model must exist
     */
    getOrThrow(id: string): ModelDefinition {
        const model = this.models.get(id)
        if (!model) {
            throw new Error(`[ModelRegistry] Model not found: ${id}`)
        }
        return model
    }

    all(): ModelDefinition[] {
        return Array.from(this.models.values())
    }

    ids(): string[] {
        return Array.from(this.models.keys())
    }

    has(id: string): boolean {
        return this.models.has(id)
    }

    /** First registered model as default */
    default(): ModelDefinition | undefined {
        return this.all()[0]
    }
}

/** Singleton instance */
export const registry = new ModelRegistry()
