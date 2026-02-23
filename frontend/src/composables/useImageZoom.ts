import { reactive, computed } from 'vue'

/**
 * Reusable zoom/pan composable for image containers.
 * Used by GenerationPreview and LightboxOverlay.
 */
export function useImageZoom() {
    const state = reactive({
        scale: 1,
        translateX: 0,
        translateY: 0,
        isDragging: false,
        startX: 0,
        startY: 0
    })

    const imageStyle = computed(() => ({
        transform: `translate(${state.translateX}px, ${state.translateY}px) scale(${state.scale})`,
        transition: state.isDragging ? 'none' : 'transform 0.1s ease-out'
    }))

    function handleWheel(event: WheelEvent) {
        const delta = event.deltaY > 0 ? -0.1 : 0.1
        state.scale = Math.max(0.1, Math.min(5, state.scale + delta))
    }

    function startDrag(event: MouseEvent) {
        state.isDragging = true
        state.startX = event.clientX - state.translateX
        state.startY = event.clientY - state.translateY
    }

    function onDrag(event: MouseEvent) {
        if (!state.isDragging) return
        state.translateX = event.clientX - state.startX
        state.translateY = event.clientY - state.startY
    }

    function stopDrag() {
        state.isDragging = false
    }

    function zoomIn() {
        state.scale = Math.min(5, state.scale + 0.2)
    }

    function zoomOut() {
        state.scale = Math.max(0.1, state.scale - 0.2)
    }

    function resetZoom() {
        state.scale = 1
        state.translateX = 0
        state.translateY = 0
    }

    return {
        scale: computed(() => state.scale),
        imageStyle,
        handleWheel,
        startDrag,
        onDrag,
        stopDrag,
        zoomIn,
        zoomOut,
        resetZoom
    }
}
