import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export interface DatasetImage {
  path: string
  filename: string
  stem?: string  // 文件名不含扩展名 (用于配对匹配)
  width: number
  height: number
  size: number
  caption?: string
  hasLatentCache: boolean
  hasTextCache: boolean
  hasSiglipCache?: boolean  // Omni 模式 SigLIP 缓存
  thumbnailUrl: string
}

// 通道信息 (来自后端 detect_channels)
export interface ChannelInfo {
  name: string
  role: 'target' | 'source' | 'condition' | 'reference'
  directory: string
  encoder: string
  image_count: number
}

// 统一的图片组 (替代 ImagePair)
export interface ImageGroup {
  id: string                              // 组 ID (文件名 stem)
  caption: string
  target: DatasetImage | null
  channels: Record<string, DatasetImage>  // channel_name -> image
}

// 配对图片 (旧类型，向后兼容)
export interface ImagePair {
  id: string
  source?: DatasetImage
  target: DatasetImage
  conditions?: DatasetImage[]
}

// 数据集类型
export type DatasetType = 'standard' | 'multi_channel' | 'paired' | 'omni'

export interface Pagination {
  page: number
  pageSize: number
  totalPages: number
  totalCount: number
  hasNext: boolean
  hasPrev: boolean
}

export interface DatasetInfo {
  path: string
  name: string
  imageCount: number
  totalSize: number
  images: DatasetImage[]
  pagination?: Pagination
  // 全局缓存统计
  totalLatentCached?: number
  totalTextCached?: number
  // 数据集元信息
  datasetType?: DatasetType
  channels?: ChannelInfo[]
  groups?: ImageGroup[]
  subdirectories?: string[]
  pairs?: ImagePair[]
}

// 本地数据集列表项（轻量级）
export interface LocalDataset {
  name: string
  path: string
  imageCount: number
  type?: string
  channels?: ChannelInfo[]
}

export const useDatasetStore = defineStore('dataset', () => {
  // ============================================================================
  // 状态
  // ============================================================================
  const datasets = ref<DatasetInfo[]>([])
  const currentDataset = ref<DatasetInfo | null>(null)
  const isLoading = ref(false)
  const selectedImages = ref<Set<string>>(new Set())

  // 本地数据集列表（用于替代组件中的 localDatasets）
  const localDatasets = ref<LocalDataset[]>([])
  const datasetsDir = ref('')

  // 分页状态
  const currentPage = ref(1)
  const pageSize = ref(50)
  const pagination = ref<Pagination | null>(null)

  const currentImages = computed(() => currentDataset.value?.images || [])

  // 多通道支持
  const currentGroups = ref<ImageGroup[]>([])
  const currentChannels = ref<ChannelInfo[]>([])
  const datasetType = computed(() => currentDataset.value?.datasetType || 'standard')
  const isMultiChannel = computed(() => datasetType.value === 'multi_channel')

  // 旧接口兼容
  const currentPairs = computed(() => currentDataset.value?.pairs || [])
  const isPairedMode = computed(() => datasetType.value !== 'standard')
  const isOmniMode = computed(() => datasetType.value === 'omni')

  // ============================================================================
  // 数据集列表操作
  // ============================================================================

  /**
   * 获取本地数据集列表
   */
  async function fetchDatasets() {
    try {
      const response = await axios.get('/api/dataset/list')
      const payload = response.data.data || response.data
      localDatasets.value = payload.datasets || []
      datasetsDir.value = payload.datasetsDir || ''
      return response.data
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
      throw error
    }
  }

  /**
   * 清除当前数据集（返回列表视图时调用）
   */
  function clearCurrentDataset() {
    currentDataset.value = null
    clearSelection()
  }

  // ============================================================================
  // 数据集扫描
  // ============================================================================

  async function scanDataset(path: string, page: number = 1, size: number = pageSize.value) {
    isLoading.value = true
    try {
      const response = await axios.post('/api/dataset/scan', {
        path,
        page,
        page_size: size
      })
      // Backend wraps in ApiResponse: {success, message, data: {...datasetInfo}}
      const rawData = response.data?.data ?? response.data
      const datasetInfo: DatasetInfo = rawData

      // 保存分页信息
      const paginationData = rawData.pagination ?? response.data.pagination
      if (paginationData) {
        pagination.value = paginationData
        currentPage.value = paginationData.page
        pageSize.value = paginationData.pageSize
      }

      // 添加或更新数据集列表
      const existingIndex = datasets.value.findIndex((d: DatasetInfo) => d.path === path)
      if (existingIndex >= 0) {
        datasets.value[existingIndex] = datasetInfo
      } else {
        datasets.value.push(datasetInfo)
      }

      // 翻页时保留之前的缓存统计（避免被重置为 null）
      if (currentDataset.value && currentDataset.value.path === path) {
        datasetInfo.totalLatentCached = datasetInfo.totalLatentCached ?? currentDataset.value.totalLatentCached
        datasetInfo.totalTextCached = datasetInfo.totalTextCached ?? currentDataset.value.totalTextCached
      }

      currentDataset.value = datasetInfo

      // Multi-channel: auto-load channel groups if detected
      if (datasetInfo.datasetType === 'multi_channel') {
        currentChannels.value = datasetInfo.channels || []
        // Load groups in background
        scanChannels(path, page, size)
      } else {
        currentGroups.value = []
        currentChannels.value = []
      }

      return datasetInfo
    } catch (error) {
      console.error('Failed to scan dataset:', error)
      throw error
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Scan multi-channel dataset for ImageGroups
   */
  async function scanChannels(path: string, page: number = 1, size: number = pageSize.value) {
    try {
      const response = await axios.post('/api/dataset/scan-channels', {
        path,
        page,
        page_size: size
      })
      const data = response.data
      currentGroups.value = data.groups || []
      currentChannels.value = data.channels || []

      if (data.pagination) {
        pagination.value = data.pagination
      }

      return data
    } catch (error) {
      console.error('Failed to scan channels:', error)
    }
  }

  // 加载指定页（翻页后自动刷新统计）
  async function loadPage(page: number) {
    if (!currentDataset.value) return
    const result = await scanDataset(currentDataset.value.path, page, pageSize.value)
    // 异步刷新缓存统计（不阻塞翻页）
    fetchStats()
    return result
  }

  // 修改每页数量
  async function changePageSize(size: number) {
    pageSize.value = size
    currentPage.value = 1
    if (currentDataset.value) {
      const result = await scanDataset(currentDataset.value.path, 1, size)
      fetchStats()
      return result
    }
  }

  // ============================================================================
  // 缓存统计
  // ============================================================================

  const isLoadingStats = ref(false)

  async function fetchStats(path?: string) {
    const targetPath = path || currentDataset.value?.path
    if (!targetPath) return

    isLoadingStats.value = true
    try {
      const response = await axios.post('/api/dataset/stats', { path: targetPath })
      // Stats API returns ApiResponse: { success, data: { totalLatentCached, ... } }
      const stats = response.data?.data ?? response.data
      if (currentDataset.value && currentDataset.value.path === targetPath) {
        currentDataset.value.totalLatentCached = stats.totalLatentCached
        currentDataset.value.totalTextCached = stats.totalTextCached
      }
      return stats
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    } finally {
      isLoadingStats.value = false
    }
  }

  // ============================================================================
  // 标注操作
  // ============================================================================

  async function loadCaption(imagePath: string) {
    try {
      const response = await axios.get(`/api/dataset/caption?path=${encodeURIComponent(imagePath)}`)
      return response.data.caption
    } catch (error) {
      console.error('Failed to load caption:', error)
      return null
    }
  }

  async function saveCaption(imagePath: string, caption: string) {
    try {
      await axios.post('/api/dataset/caption', { path: imagePath, caption })

      // 更新本地状态
      if (currentDataset.value) {
        const image = currentDataset.value.images.find((img: DatasetImage) => img.path === imagePath)
        if (image) {
          image.caption = caption
        }
      }
      return true
    } catch (error) {
      console.error('Failed to save caption:', error)
      return false
    }
  }

  async function generateCaptions(modelType: 'qwen' | 'blip' = 'qwen') {
    try {
      if (!currentDataset.value) return

      const response = await axios.post('/api/dataset/generate-captions', {
        datasetPath: currentDataset.value.path,
        modelType
      })
      return response.data
    } catch (error) {
      console.error('Failed to generate captions:', error)
      throw error
    }
  }

  // ============================================================================
  // 图片选择
  // ============================================================================

  // 是否当前页全部选中
  const isAllCurrentPageSelected = computed(() => {
    if (!currentDataset.value || currentDataset.value.images.length === 0) {
      return false
    }
    return currentDataset.value.images.every(
      (img: DatasetImage) => selectedImages.value.has(img.path)
    )
  })

  // 当前页选中数量
  const currentPageSelectedCount = computed(() => {
    if (!currentDataset.value) return 0
    return currentDataset.value.images.filter(
      (img: DatasetImage) => selectedImages.value.has(img.path)
    ).length
  })

  function toggleImageSelection(imagePath: string) {
    if (selectedImages.value.has(imagePath)) {
      selectedImages.value.delete(imagePath)
    } else {
      selectedImages.value.add(imagePath)
    }
  }

  /**
   * 选中当前页所有图片
   */
  function selectCurrentPage() {
    if (currentDataset.value) {
      currentDataset.value.images.forEach((img: DatasetImage) => {
        selectedImages.value.add(img.path)
      })
    }
  }

  /**
   * 取消选中当前页所有图片
   */
  function deselectCurrentPage() {
    if (currentDataset.value) {
      currentDataset.value.images.forEach((img: DatasetImage) => {
        selectedImages.value.delete(img.path)
      })
    }
  }

  /**
   * 切换当前页全选状态
   */
  function toggleSelectCurrentPage() {
    if (isAllCurrentPageSelected.value) {
      deselectCurrentPage()
    } else {
      selectCurrentPage()
    }
  }

  // 保留旧 API 兼容
  function selectAll() {
    selectCurrentPage()
  }

  function clearSelection() {
    selectedImages.value.clear()
  }

  // ============================================================================
  // 返回
  // ============================================================================

  return {
    // 状态
    datasets,
    currentDataset,
    currentImages,
    isLoading,
    selectedImages,
    // 多通道支持
    currentGroups,
    currentChannels,
    isMultiChannel,
    // 旧接口兼容
    currentPairs,
    datasetType,
    isPairedMode,
    isOmniMode,
    // 本地数据集列表
    localDatasets,
    datasetsDir,
    // 分页相关
    currentPage,
    pageSize,
    pagination,
    // 选择相关
    isAllCurrentPageSelected,
    currentPageSelectedCount,
    // 缓存统计相关
    isLoadingStats,
    fetchStats,
    // 方法
    fetchDatasets,
    clearCurrentDataset,
    scanDataset,
    scanChannels,
    loadPage,
    changePageSize,
    loadCaption,
    saveCaption,
    generateCaptions,
    toggleImageSelection,
    selectCurrentPage,
    deselectCurrentPage,
    toggleSelectCurrentPage,
    selectAll,
    clearSelection
  }
})
