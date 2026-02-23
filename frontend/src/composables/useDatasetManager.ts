import { ref, computed } from 'vue'
import type { DatasetImage } from '@/stores/dataset'

const MAX_RETRY = 2

/**
 * Composable for managing dataset image loading, retry logic, and selection.
 */
export function useDatasetManager() {
    // Image loading state
    const imageLoadFailed = ref(new Set<string>())
    const imageRetryCount = ref(new Map<string, number>())

    function getImageUrl(image: DatasetImage): string {
        const retry = imageRetryCount.value.get(image.path) || 0
        const cacheBuster = retry > 0 ? `&_t=${Date.now()}&_r=${retry}` : ''
        return `${image.thumbnailUrl}${cacheBuster}`
    }

    function handleImageError(event: Event, image: DatasetImage) {
        const retryCount = imageRetryCount.value.get(image.path) || 0
        if (retryCount < MAX_RETRY) {
            const newRetry = retryCount + 1
            imageRetryCount.value.set(image.path, newRetry)
            const img = event.target as HTMLImageElement
            setTimeout(() => {
                img.src = getImageUrl(image)
            }, 1000 * newRetry)
        } else {
            imageLoadFailed.value.add(image.path)
        }
    }

    function retryLoadImage(image: DatasetImage) {
        imageLoadFailed.value.delete(image.path)
        imageRetryCount.value.delete(image.path)
        imageRetryCount.value = new Map(imageRetryCount.value)
        imageLoadFailed.value = new Set(imageLoadFailed.value)
    }

    // Timer tracking for cleanup
    const activeIntervals: number[] = []
    const activeTimeouts: number[] = []

    function trackInterval(id: number): number {
        activeIntervals.push(id)
        return id
    }

    function trackTimeout(id: number): number {
        activeTimeouts.push(id)
        return id
    }

    function clearTrackedInterval(id: number) {
        clearInterval(id)
        const index = activeIntervals.indexOf(id)
        if (index > -1) activeIntervals.splice(index, 1)
    }

    function clearAllTrackedTimers() {
        activeIntervals.forEach(id => clearInterval(id))
        activeTimeouts.forEach(id => clearTimeout(id))
        activeIntervals.length = 0
        activeTimeouts.length = 0
    }

    // Format helpers
    function formatSize(bytes: number): string {
        if (!bytes) return '0 B'
        const k = 1024
        const sizes = ['B', 'KB', 'MB', 'GB']
        const i = Math.floor(Math.log(bytes) / Math.log(k))
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
    }

    return {
        imageLoadFailed,
        imageRetryCount,
        getImageUrl,
        handleImageError,
        retryLoadImage,
        trackInterval,
        trackTimeout,
        clearTrackedInterval,
        clearAllTrackedTimers,
        formatSize
    }
}
