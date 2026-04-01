import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useSystemStore, type GpuInfo, type SystemInfo, type ModelStatus, type DownloadStatus } from './system'
import { useTrainingStore } from './training'

/**
 * 缓存状态接口
 */
export interface CacheStatus {
  latent: {
    status: 'idle' | 'running' | 'completed' | 'failed'
    progress?: number
    current?: number
    total?: number
    currentFile?: string
  }
  text: {
    status: 'idle' | 'running' | 'completed' | 'failed'
    progress?: number
    current?: number
    total?: number
    currentFile?: string
  }
}

/**
 * 生成状态接口
 */
export interface GenerationStatus {
  running: boolean
  current_step: number
  total_steps: number
  progress: number
  stage: 'idle' | 'loading' | 'generating' | 'saving' | 'completed' | 'failed'
  message: string
  error: string | null
}

/**
 * WebSocket 消息接口
 */
export interface WebSocketMessage {
  type: string
  timestamp?: string
  gpu?: GpuInfo
  system_info?: SystemInfo
  model_status?: ModelStatus
  download?: DownloadStatus
  training?: any
  training_history?: any
  cache?: CacheStatus
  generation?: GenerationStatus
  logs?: any[]
  message?: string
  progress?: any
  data?: any
  current_step?: number
  total_steps?: number
  stage?: string
}

/**
 * 日志条目接口
 */
interface LogEntry {
  time: string
  message: string
  level: 'info' | 'success' | 'warning' | 'error'
}

/**
 * WebSocket Store
 * 统一管理所有 WebSocket 通讯
 */
export const useWebSocketStore = defineStore('websocket', () => {
  // WebSocket 实例
  const ws = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  const maxReconnectAttempts = 10
  const reconnectDelay = 3000

  // 缓存状态
  const cacheStatus = ref<CacheStatus>({
    latent: { status: 'idle', progress: 0 },
    text: { status: 'idle', progress: 0 }
  })

  // 生成状态
  const generationStatus = ref<GenerationStatus>({
    running: false,
    current_step: 0,
    total_steps: 0,
    progress: 0,
    stage: 'idle',
    message: '',
    error: null
  })

  // 日志
  const logs = ref<LogEntry[]>([])
  const maxLogs = 500

  // 计算属性
  const connectionStatus = computed(() => {
    if (isConnected.value) return 'connected'
    if (reconnectAttempts.value > 0) return 'reconnecting'
    return 'disconnected'
  })

  const hasRunningTask = computed(() => {
    const systemStore = useSystemStore()
    return systemStore.downloadStatus.status === 'running' ||
      cacheStatus.value.latent.status === 'running' ||
      cacheStatus.value.text.status === 'running' ||
      generationStatus.value.running
  })

  const isGenerating = computed(() => generationStatus.value.running)

  /**
   * 获取 WebSocket URL
   */
  function getWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    // 开发模式下走 vite 代理，直接用当前 host
    return `${protocol}//${host}/ws`
  }

  /**
   * 建立连接
   */
  function connect() {
    if (ws.value?.readyState === WebSocket.OPEN) {
      console.log('[WebSocket] Already connected')
      return
    }

    const url = getWebSocketUrl()
    console.log('[WebSocket] Connecting to:', url)

    try {
      ws.value = new WebSocket(url)

      ws.value.onopen = () => {
        console.log('[WebSocket] Connected')
        isConnected.value = true
        reconnectAttempts.value = 0

        // 更新系统状态
        const systemStore = useSystemStore()
        systemStore.setConnected(true)
      }

      ws.value.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          handleMessage(message)
        } catch (e) {
          // 非 JSON 消息（如 pong）
          if (event.data !== 'pong') {
            console.log('[WebSocket] Non-JSON message:', event.data)
          }
        }
      }

      ws.value.onclose = (event) => {
        console.log('[WebSocket] Disconnected:', event.code, event.reason)
        isConnected.value = false
        ws.value = null

        // 更新系统状态
        const systemStore = useSystemStore()
        systemStore.setConnected(false)

        // 自动重连
        if (reconnectAttempts.value < maxReconnectAttempts) {
          reconnectAttempts.value++
          console.log(`[WebSocket] Reconnecting in ${reconnectDelay}ms (attempt ${reconnectAttempts.value})`)
          setTimeout(connect, reconnectDelay)
        }
      }

      ws.value.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
      }

    } catch (error) {
      console.error('[WebSocket] Connection failed:', error)
      isConnected.value = false
    }
  }

  /**
   * 断开连接
   */
  function disconnect() {
    if (ws.value) {
      ws.value.close()
      ws.value = null
    }
    isConnected.value = false
    reconnectAttempts.value = maxReconnectAttempts // 防止自动重连
  }

  /**
   * 发送消息
   */
  function send(data: any) {
    if (ws.value?.readyState === WebSocket.OPEN) {
      ws.value.send(typeof data === 'string' ? data : JSON.stringify(data))
    }
  }

  /**
   * 发送心跳
   */
  function sendPing() {
    send('ping')
  }

  /**
   * 请求完整状态
   */
  function requestFullStatus() {
    send({ action: 'get_status' })
  }

  /**
   * 处理消息
   */
  function handleMessage(message: WebSocketMessage) {
    const systemStore = useSystemStore()
    const trainingStore = useTrainingStore()

    switch (message.type) {
      case 'init':
        // 初始化数据 - 连接时收到的完整状态
        handleInitMessage(message, systemStore, trainingStore)
        break

      case 'status_update':
        // 定时状态更新
        handleStatusUpdate(message, systemStore, trainingStore)
        break

      case 'generation_progress':
        // 生成进度实时更新
        handleGenerationProgress(message)
        break

      case 'training_log':
        // 训练日志
        handleTrainingLog(message, trainingStore)
        break

      case 'download_log':
        // 下载日志
        handleDownloadLog(message)
        break

      case 'cache_latent_log':
        // Latent 缓存日志
        handleCacheLog('latent', message)
        break

      case 'cache_text_log':
        // Text 缓存日志
        handleCacheLog('text', message)
        break

      case 'gpu':
        // GPU 信息响应
        if (message.data) {
          systemStore.updateGpuInfo(message.data)
        }
        break

      case 'full_status':
        // 完整状态响应
        handleFullStatus(message, systemStore, trainingStore)
        break

      case 'logs':
        // 日志响应
        if (message.data) {
          logs.value = message.data
        }
        break

      case 'logs_cleared':
        logs.value = []
        break

      case 'training_reset':
        // 训练开始时重置图表数据
        console.log('[WebSocket] Training reset - clearing chart data')
        if (message.training_history) {
          const h = message.training_history
          trainingStore.setHistory(h.loss_history || [], h.lr_history || [])
          trainingStore.updateProgress({
            currentEpoch: 0,
            totalEpochs: h.total_epochs || 0,
            currentStep: 0,
            totalSteps: h.total_steps || 0,
            learningRate: 0,
            loss: 0,
            elapsedTime: 0,
            estimatedTimeRemaining: 0
          })
        }
        break

      default:
        console.log('[WebSocket] Unknown message type:', message.type)
    }
  }

  /**
   * 处理初始化消息
   */
  function handleInitMessage(
    message: WebSocketMessage,
    systemStore: ReturnType<typeof useSystemStore>,
    trainingStore: ReturnType<typeof useTrainingStore>
  ) {
    if (message.gpu) {
      systemStore.updateGpuInfo(message.gpu)
    }
    if (message.system_info) {
      systemStore.updateSystemInfo(message.system_info)
    }
    if (message.model_status) {
      systemStore.updateModelStatus(message.model_status)
    }
    if (message.download) {
      systemStore.updateDownloadStatus(message.download)
    }
    if (message.training) {
      trainingStore.progress.isRunning = message.training.running
      trainingStore.progress.isLoading = message.training.loading || false
    }
    if (message.cache) {
      cacheStatus.value = message.cache
    }
    if (message.generation) {
      generationStatus.value = message.generation
    }
    if (message.logs) {
      logs.value = message.logs
    }

    // 恢复训练历史（图表数据）
    if (message.training_history) {
      const h = message.training_history
      // 使用 setHistory 设置完整的历史数据
      trainingStore.setHistory(h.loss_history || [], h.lr_history || [])
      // 更新其他进度信息
      trainingStore.updateProgress({
        currentEpoch: h.current_epoch || 0,
        totalEpochs: h.total_epochs || 0,
        currentStep: h.current_step || 0,
        totalSteps: h.total_steps || 0,
        learningRate: h.learning_rate || 0,
        loss: h.loss || 0,
        elapsedTime: h.elapsed_time || 0,
        estimatedTimeRemaining: h.estimated_remaining || 0
      })

      // 恢复训练开始时间戳（解决刷新后时间重置问题）
      if (h.start_timestamp && h.start_timestamp > 0) {
        // 后端时间戳是秒，转换为毫秒
        trainingStartTime = h.start_timestamp * 1000
        console.log('[WebSocket] Restored training start time:', new Date(trainingStartTime).toLocaleTimeString())
      }

      console.log('[WebSocket] Restored training history:', h.loss_history?.length || 0, 'loss points,', h.lr_history?.length || 0, 'lr points')
    }

    console.log('[WebSocket] Initialized with full state')
  }

  /**
   * 处理状态更新
   */
  function handleStatusUpdate(
    message: WebSocketMessage,
    systemStore: ReturnType<typeof useSystemStore>,
    trainingStore: ReturnType<typeof useTrainingStore>
  ) {
    if (message.gpu) {
      systemStore.updateGpuInfo(message.gpu)
    }
    if (message.system_info) {
      systemStore.updateSystemInfo(message.system_info)
    }
    if (message.model_status) {
      systemStore.updateModelStatus(message.model_status)
    }
    if (message.download) {
      const prevStatus = systemStore.downloadStatus.status
      systemStore.updateDownloadStatus(message.download)

      // 下载完成/失败通知
      if (prevStatus === 'running' && message.download.status === 'completed') {
        addLog('模型下载完成！', 'success')
      } else if (prevStatus === 'running' && message.download.status === 'failed') {
        addLog('模型下载失败', 'error')
      }
    }
    if (message.training) {
      const prevRunning = trainingStore.progress.isRunning
      const prevLoading = trainingStore.progress.isLoading
      trainingStore.progress.isRunning = message.training.running
      trainingStore.progress.isLoading = message.training.loading || false

      // Update step/loss/lr/epoch from WebSocket pushed data
      if (message.training.current_step) {
        const parts = message.training.current_step.split('/')
        trainingStore.progress.currentStep = parseInt(parts[0]) || 0
        trainingStore.progress.totalSteps = parseInt(parts[1]) || 0
      }
      if (message.training.current_loss) {
        trainingStore.progress.loss = parseFloat(message.training.current_loss) || 0
      }
      if (message.training.current_lr) {
        trainingStore.progress.learningRate = parseFloat(message.training.current_lr) || 0
      }
      if (message.training.current_epoch) {
        trainingStore.progress.currentEpoch = parseInt(message.training.current_epoch) || 0
      }
      if (message.training.num_train_epochs) {
        trainingStore.progress.totalEpochs = parseInt(message.training.num_train_epochs) || 0
      }

      // step 日志已由 training_log 消息推送，此处仅更新 lastLoggedStep 防止其他地方重复
      if (message.training.running && message.training.current_step) {
        lastLoggedStep = message.training.current_step
      }

      // 从加载中变为训练中时添加日志
      if (prevLoading && !message.training.loading && message.training.running) {
        addLog('模型加载完成，开始训练', 'success')
      }

      // 只在状态首次变化时添加日志（从 running 变为其他状态）
      if (prevRunning && !message.training.running) {
        // 重置训练开始时间
        trainingStartTime = null

        if (message.training.status === 'completed') {
          addLog('训练完成！', 'success')
        } else if (message.training.status === 'failed') {
          const hint = message.training.hint || '未知错误'
          addLog(`训练失败 (退出码 ${message.training.code}): ${hint}`, 'error')
        }
      }
    }
    if (message.cache) {
      handleCacheStatusUpdate(message.cache)
    }
    if (message.generation) {
      handleGenerationStatusUpdate(message.generation)
    }
  }

  /**
   * 处理完整状态
   */
  function handleFullStatus(
    message: WebSocketMessage,
    systemStore: ReturnType<typeof useSystemStore>,
    trainingStore: ReturnType<typeof useTrainingStore>
  ) {
    if (message.gpu) systemStore.updateGpuInfo(message.gpu)
    if (message.system_info) systemStore.updateSystemInfo(message.system_info)
    if (message.model_status) systemStore.updateModelStatus(message.model_status)
    if (message.download) systemStore.updateDownloadStatus(message.download)
    if (message.training) {
      trainingStore.progress.isRunning = message.training.running
      trainingStore.progress.isLoading = message.training.loading || false
    }
    if (message.cache) cacheStatus.value = message.cache
    if (message.generation) generationStatus.value = message.generation
  }

  /**
   * 处理训练日志
   */
  function handleTrainingLog(message: WebSocketMessage, trainingStore: ReturnType<typeof useTrainingStore>) {
    if (message.message) {
      addLog(message.message, 'info')
    }
    if (message.progress) {
      handleTrainingProgress(message.progress, trainingStore)
    }
  }

  /**
   * 处理下载日志
   */
  function handleDownloadLog(message: WebSocketMessage) {
    if (message.message) {
      addLog(`[下载] ${message.message}`, 'info')
    }
    if (message.progress) {
      const systemStore = useSystemStore()
      switch (message.progress.type) {
        case 'percent':
          systemStore.updateDownloadStatus({ progress: message.progress.value })
          break
        case 'file':
          systemStore.updateDownloadStatus({ currentFile: message.progress.name })
          break
        case 'items':
          systemStore.updateDownloadStatus({ progress: message.progress.percent })
          break
      }
    }
  }

  /**
   * 处理缓存日志
   */
  function handleCacheLog(type: 'latent' | 'text', message: WebSocketMessage) {
    if (message.message) {
      addLog(`[${type === 'latent' ? 'Latent' : 'Text'}缓存] ${message.message}`, 'info')
    }
    if (message.progress) {
      handleCacheProgress(type, message.progress)
    }
  }

  // 训练开始时间（用于自动计算时间）
  let trainingStartTime: number | null = null
  // 上次日志的 step（避免重复打印）
  let lastLoggedStep: string | null = null

  /**
   * 处理训练进度
   */
  function handleTrainingProgress(progress: any, trainingStore: ReturnType<typeof useTrainingStore>) {
    // 新格式：一次性更新多个字段
    const update: Partial<typeof trainingStore.progress> = {}

    if (progress.epoch) {
      update.currentEpoch = progress.epoch.current
      update.totalEpochs = progress.epoch.total
    }

    if (progress.step) {
      update.currentStep = progress.step.current
      update.totalSteps = progress.step.total || trainingStore.progress.totalSteps

      // 自动计算时间（当没有 tqdm 时间信息时）
      if (!progress.time && progress.step.current > 0) {
        // 记录训练开始时间（第一次收到 step > 0 时）
        if (!trainingStartTime) {
          trainingStartTime = Date.now()
        }

        const elapsedMs = Date.now() - trainingStartTime
        const elapsedSec = Math.floor(elapsedMs / 1000)
        update.elapsedTime = elapsedSec

        // 基于当前进度估算剩余时间
        const currentStep = progress.step.current
        const totalSteps = progress.step.total || trainingStore.progress.totalSteps
        if (currentStep > 0 && totalSteps > 0) {
          const progressPercent = currentStep / totalSteps
          const estimatedTotalSec = elapsedSec / progressPercent
          const remainingSec = Math.max(0, Math.floor(estimatedTotalSec - elapsedSec))
          update.estimatedTimeRemaining = remainingSec
        }
      }
    }

    if (progress.loss !== undefined) {
      update.loss = progress.loss
    }

    if (progress.learningRate !== undefined) {
      update.learningRate = progress.learningRate
    }

    if (progress.time) {
      // 解析时间字符串 "MM:SS" 或 "H:MM:SS"
      // tqdm 格式: MM:SS (分:秒) 或 H:MM:SS (时:分:秒)
      const parseTimeStr = (str: string): number => {
        const parts = str.split(':').map(Number)
        if (parts.length === 2) {
          // MM:SS 格式 -> 分钟*60 + 秒
          return parts[0] * 60 + parts[1]
        }
        if (parts.length === 3) {
          // H:MM:SS 格式 -> 小时*3600 + 分钟*60 + 秒
          return parts[0] * 3600 + parts[1] * 60 + parts[2]
        }
        return 0
      }
      update.elapsedTime = parseTimeStr(progress.time.elapsed)
      update.estimatedTimeRemaining = parseTimeStr(progress.time.remaining)
    }

    // 使用 EMA loss 作为图表数据（更平滑）
    if (progress.ema_loss !== undefined) {
      update.loss = progress.ema_loss  // 显示 EMA loss
    }

    // Strictly synchronize history updates
    const lossVal = progress.ema_loss ?? progress.loss

    // Only update history if we have a valid loss value (implies a training step)
    if (lossVal !== undefined) {
      if (Object.keys(update).length > 0) {
        trainingStore.updateProgress(update)
      }

      // Ensure specific LR value is recorded for this step to keep arrays aligned
      // If progress doesn't have LR, use current store value to avoid "holes"
      const lrVal = progress.learningRate ?? trainingStore.progress.learningRate

      trainingStore.appendHistory(lossVal, lrVal)
    } else if (Object.keys(update).length > 0) {
      // Just update status (e.g. time, epoch) without adding history points
      trainingStore.updateProgress(update)
    }
  }


  /**
   * 处理缓存进度
   */
  function handleCacheProgress(type: 'latent' | 'text', progress: any) {
    const cache = cacheStatus.value[type]

    switch (progress.type) {
      case 'progress':
        cache.status = 'running'  // 确保设置为 running
        cache.current = progress.current
        cache.total = progress.total
        cache.progress = progress.percent
        break
      case 'percent':
        cache.status = 'running'
        cache.progress = progress.value
        break
      case 'file':
        cache.status = 'running'
        cache.currentFile = progress.name
        break
      case 'completed':
        cache.status = 'completed'
        cache.progress = 100
        addLog(`${type === 'latent' ? 'Latent' : 'Text'} 缓存完成！`, 'success')
        break
    }
  }

  /**
   * 处理缓存状态更新
   */
  function handleCacheStatusUpdate(cache: CacheStatus) {
    const prevLatent = cacheStatus.value.latent.status
    const prevText = cacheStatus.value.text.status

    cacheStatus.value = cache

    // 状态变化通知
    if (prevLatent === 'running' && cache.latent.status === 'completed') {
      addLog('Latent 缓存生成完成！', 'success')
    } else if (prevLatent === 'running' && cache.latent.status === 'failed') {
      addLog('Latent 缓存生成失败', 'error')
    }

    if (prevText === 'running' && cache.text.status === 'completed') {
      addLog('Text 缓存生成完成！', 'success')
    } else if (prevText === 'running' && cache.text.status === 'failed') {
      addLog('Text 缓存生成失败', 'error')
    }
  }

  /**
   * 处理生成进度
   */
  function handleGenerationProgress(message: WebSocketMessage) {
    generationStatus.value = {
      running: message.stage !== 'completed' && message.stage !== 'failed' && message.stage !== 'idle',
      current_step: message.current_step || 0,
      total_steps: message.total_steps || 0,
      progress: message.progress || 0,
      stage: (message.stage as GenerationStatus['stage']) || 'idle',
      message: message.message || '',
      error: message.stage === 'failed' ? message.message || null : null
    }
  }

  /**
   * 处理生成状态更新
   */
  function handleGenerationStatusUpdate(generation: GenerationStatus) {
    const prevStage = generationStatus.value.stage
    generationStatus.value = generation

    // 状态变化通知
    if (prevStage !== 'completed' && generation.stage === 'completed') {
      addLog('图像生成完成！', 'success')
    } else if (prevStage !== 'failed' && generation.stage === 'failed') {
      addLog(`图像生成失败: ${generation.error || '未知错误'}`, 'error')
    }
  }

  /**
   * 重置生成状态
   */
  function resetGenerationStatus() {
    generationStatus.value = {
      running: false,
      current_step: 0,
      total_steps: 0,
      progress: 0,
      stage: 'idle',
      message: '',
      error: null
    }
  }

  /**
   * 添加日志
   */
  function addLog(message: string, level: LogEntry['level'] = 'info') {
    const timestamp = new Date().toLocaleTimeString()
    logs.value.push({ time: timestamp, message, level })

    // 限制日志数量
    if (logs.value.length > maxLogs) {
      logs.value = logs.value.slice(-maxLogs)
    }
  }

  /**
   * 清除日志
   */
  function clearLogs() {
    send({ action: 'clear_logs' })
    logs.value = []
  }

  // 心跳定时器
  let heartbeatInterval: ReturnType<typeof setInterval> | null = null

  /**
   * 启动心跳
   */
  function startHeartbeat() {
    if (heartbeatInterval) return
    heartbeatInterval = setInterval(() => {
      if (isConnected.value) {
        sendPing()
      }
    }, 30000) // 30秒心跳
  }

  /**
   * 停止心跳
   */
  function stopHeartbeat() {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval)
      heartbeatInterval = null
    }
  }

  return {
    // 状态
    ws,
    isConnected,
    connectionStatus,
    cacheStatus,
    generationStatus,
    hasRunningTask,
    isGenerating,
    logs,

    // 方法
    connect,
    disconnect,
    send,
    sendPing,
    requestFullStatus,
    addLog,
    clearLogs,
    startHeartbeat,
    stopHeartbeat,
    resetGenerationStatus
  }
})
