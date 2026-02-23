import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const STORAGE_KEY_PARAMS = 'generation_params'
const STORAGE_KEY_RESULT = 'generation_result'
const STORAGE_KEY_TASK_ID = 'generation_task_id'

export interface PendingTask {
    prompt: string
    width: number
    height: number
    steps: number
    seed: number
    startTime: number
    interrupted: boolean
    taskId?: string
}

/**
 * SSE-based image generation with:
 * - Per-step progress tracking (diffusers callback)
 * - Task persistence via backend TaskManager
 * - Automatic reconnection on page return (polling fallback)
 */
export function useGenerationSSE() {
    const generating = ref(false)
    const resultImage = ref<string | null>(null)
    const resultSeed = ref<number | null>(null)
    const pendingTask = ref<PendingTask | null>(null)

    // Comparison mode
    const comparisonImages = ref<{ image: string, lora_path: string | null, lora_scale?: number }[]>([])
    const isComparisonResult = ref(false)

    // SSE progress
    const progressStage = ref('准备中...')
    const progressMessage = ref('')
    const progressStep = ref(0)
    const progressTotal = ref(0)

    // ── localStorage helpers ──

    function loadSavedParams() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY_PARAMS)
            return saved ? JSON.parse(saved) : null
        } catch { return null }
    }

    function loadSavedResult() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY_RESULT)
            return saved ? JSON.parse(saved) : null
        } catch { return null }
    }

    function saveParams(params: any) {
        try { localStorage.setItem(STORAGE_KEY_PARAMS, JSON.stringify(params)) } catch { }
    }

    function saveResult() {
        try {
            localStorage.setItem(STORAGE_KEY_RESULT, JSON.stringify({
                image: resultImage.value,
                seed: resultSeed.value
            }))
        } catch { }
    }

    function saveTaskId(taskId: string) {
        try { localStorage.setItem(STORAGE_KEY_TASK_ID, taskId) } catch { }
    }

    function clearTaskId() {
        try { localStorage.removeItem(STORAGE_KEY_TASK_ID) } catch { }
    }

    function getSavedTaskId(): string | null {
        try { return localStorage.getItem(STORAGE_KEY_TASK_ID) } catch { return null }
    }

    // ── Pending task management ──

    function savePendingTask(params: any) {
        const task: PendingTask = {
            prompt: params.prompt,
            width: params.width,
            height: params.height,
            steps: params.num_inference_steps || params.steps || 10,
            seed: params.seed,
            startTime: Date.now(),
            interrupted: false,
        }
        pendingTask.value = task
    }

    function clearPendingTask() {
        pendingTask.value = null
        clearTaskId()
    }

    /**
     * Check for an active backend task on page load.
     * If found, start polling to reconnect to its progress.
     */
    async function checkPendingTask(onComplete?: () => void) {
        // First check if there's a saved task_id
        const savedTaskId = getSavedTaskId()
        if (savedTaskId) {
            try {
                const res = await axios.get(`/api/generation/task/${savedTaskId}`)
                const task = res.data?.data
                if (task) {
                    const state = task.state
                    if (state === 'completed' && task.result) {
                        // Task completed while we were away — show result!
                        handleSuccess(task.result, {})
                        clearPendingTask()
                        onComplete?.()
                        return
                    } else if (state === 'failed') {
                        ElMessage.error('生成失败: ' + (task.error || 'Unknown error'))
                        clearPendingTask()
                        return
                    } else if (['pending', 'loading', 'generating', 'saving'].includes(state)) {
                        // Still running — start polling!
                        generating.value = true
                        progressTotal.value = task.total_steps || 10
                        progressStep.value = task.step || 0
                        updateStageDisplay(state, task.message)
                        pendingTask.value = {
                            prompt: '(reconnected)',
                            width: 0, height: 0, steps: task.total_steps || 10,
                            seed: 0, startTime: Date.now(), interrupted: false,
                            taskId: savedTaskId,
                        }
                        startPolling(savedTaskId, onComplete)
                        return
                    }
                }
            } catch {
                // Task not found — clean up
            }
            clearTaskId()
        }

        // Also check if backend has any active task (e.g., started from another tab)
        try {
            const res = await axios.get('/api/generation/active-task')
            const task = res.data?.data
            if (task && task.task_id) {
                generating.value = true
                progressTotal.value = task.total_steps || 10
                progressStep.value = task.step || 0
                updateStageDisplay(task.state, task.message)
                saveTaskId(task.task_id)
                pendingTask.value = {
                    prompt: '(reconnected)',
                    width: 0, height: 0, steps: task.total_steps || 10,
                    seed: 0, startTime: Date.now(), interrupted: false,
                    taskId: task.task_id,
                }
                startPolling(task.task_id, onComplete)
            }
        } catch {
            // No active task
        }
    }

    // ── Polling for reconnected tasks ──

    function startPolling(taskId: string, onComplete?: () => void) {
        const pollInterval = setInterval(async () => {
            try {
                const res = await axios.get(`/api/generation/task/${taskId}`)
                const task = res.data?.data
                if (!task) {
                    clearInterval(pollInterval)
                    generating.value = false
                    clearPendingTask()
                    return
                }

                progressStep.value = task.step || 0
                progressTotal.value = task.total_steps || progressTotal.value
                updateStageDisplay(task.state, task.message)

                if (task.state === 'completed' && task.result) {
                    clearInterval(pollInterval)
                    handleSuccess(task.result, {})
                    generating.value = false
                    clearPendingTask()
                    onComplete?.()
                } else if (task.state === 'failed') {
                    clearInterval(pollInterval)
                    ElMessage.error('生成失败: ' + (task.error || 'Unknown error'))
                    generating.value = false
                    clearPendingTask()
                }
            } catch {
                clearInterval(pollInterval)
                generating.value = false
                clearPendingTask()
            }
        }, 500) // Poll every 500ms
    }

    function updateStageDisplay(state: string, message: string) {
        switch (state) {
            case 'pending': progressStage.value = '⏳ 排队中...'; break
            case 'loading': progressStage.value = '🔄 加载模型...'; break
            case 'generating': progressStage.value = '🎨 生成中...'; break
            case 'saving': progressStage.value = '💾 保存中...'; break
            case 'completed': progressStage.value = '✅ 完成!'; break
            case 'failed': progressStage.value = '❌ 错误'; break
        }
        progressMessage.value = message || ''
    }

    // ── SSE Generation ──

    async function generate(params: any, onComplete?: () => void) {
        if (!params.prompt) {
            ElMessage.warning('请输入提示词')
            return
        }

        generating.value = true
        isComparisonResult.value = false
        comparisonImages.value = []
        progressStage.value = '准备中...'
        progressMessage.value = ''
        progressStep.value = 0
        progressTotal.value = params.num_inference_steps || params.steps || 10
        savePendingTask(params)

        try {
            const response = await fetch('/api/generate-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            })

            if (!response.ok) throw new Error(`HTTP ${response.status}`)

            const reader = response.body?.getReader()
            const decoder = new TextDecoder()
            if (!reader) throw new Error('Failed to get stream reader')

            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const messages = buffer.split('\n\n')
                buffer = messages.pop() || ''

                for (const message of messages) {
                    for (const line of message.split('\n')) {
                        if (!line.startsWith('data: ')) continue
                        try {
                            const data = JSON.parse(line.slice(6))

                            // Save task_id for recovery
                            if (data.task_id && !getSavedTaskId()) {
                                saveTaskId(data.task_id)
                                if (pendingTask.value) {
                                    pendingTask.value.taskId = data.task_id
                                }
                            }

                            handleSSEData(data, params, onComplete)
                        } catch {
                            // Incomplete JSON chunk
                        }
                    }
                }
            }
        } catch (e: any) {
            // SSE disconnected — but task continues on backend!
            const taskId = getSavedTaskId()
            if (taskId) {
                console.log('SSE disconnected, falling back to polling for task:', taskId)
                startPolling(taskId, onComplete)
                return // Don't clear generating state
            }
            console.error('Generation SSE error:', e)
            ElMessage.error('生成失败: ' + e.message)
        } finally {
            if (!getSavedTaskId()) {
                generating.value = false
                clearPendingTask()
            }
        }
    }

    function handleSSEData(data: any, params: any, onComplete?: () => void) {
        // Progress events
        if (data.stage) {
            updateStageDisplay(data.stage, data.message || '')
            if (data.stage === 'generating') {
                progressStep.value = data.step || 0
                progressTotal.value = data.total || progressTotal.value
            }
            if (data.stage === 'completed') {
                progressStep.value = data.total || progressTotal.value
            }
        }

        // Final result
        if (data.success === true) {
            handleSuccess(data, params)
            generating.value = false
            clearPendingTask()
            onComplete?.()
        } else if (data.success === false) {
            ElMessage.error('生成失败: ' + (data.error || data.message || 'Unknown error'))
            generating.value = false
            clearPendingTask()
        }
    }

    function handleSuccess(data: any, params: any) {
        if (data.comparison_mode && data.images) {
            isComparisonResult.value = true
            comparisonImages.value = data.images.map((img: any) => ({
                image: img.image.startsWith('data:') ? img.image : `data:image/png;base64,${img.image}`,
                lora_path: img.lora_path,
                lora_scale: img.lora_scale || params?.lora_scale,
            }))
            resultImage.value = comparisonImages.value.length > 1
                ? comparisonImages.value[1].image
                : comparisonImages.value[0].image
            resultSeed.value = data.seed
        } else {
            isComparisonResult.value = false
            const imgData = data.image || ''
            resultImage.value = imgData.startsWith('data:') ? imgData : `data:image/png;base64,${imgData}`
            resultSeed.value = data.seed
        }
        saveResult()
        ElMessage.success('生成成功！')
    }

    // ── Restore saved state on init ──
    const savedResult = loadSavedResult()
    if (savedResult) {
        resultImage.value = savedResult.image || null
        resultSeed.value = savedResult.seed || null
    }

    return {
        generating,
        resultImage,
        resultSeed,
        pendingTask,
        comparisonImages,
        isComparisonResult,
        progressStage,
        progressMessage,
        progressStep,
        progressTotal,
        generate,
        checkPendingTask,
        clearPendingTask,
        savePendingTask,
        loadSavedParams,
        saveParams,
    }
}
