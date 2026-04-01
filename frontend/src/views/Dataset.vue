<template>
  <div class="dataset-page">
    <!-- Dataset List View -->
    <template v-if="!currentView">
      <div class="page-header">
        <h1 class="gradient-text">数据集管理</h1>
        <p class="subtitle">{{ datasetsDir }}</p>
      </div>

      <!-- Toolbar -->
      <div class="dataset-toolbar glass-card">
        <el-button type="primary" size="large" @click="showCreateDialog = true">
          <el-icon><Plus /></el-icon> 新建数据集
        </el-button>
        <el-button size="large" @click="showCreateFolderDialog = true">
          <el-icon><FolderAdd /></el-icon> 新建文件夹
        </el-button>
        <el-button size="large" @click="refreshCurrentLevel">
          <el-icon><Refresh /></el-icon> 刷新
        </el-button>
        <el-divider direction="vertical" />
        <!-- Open current folder as a single dataset for bulk operations -->
        <el-button v-if="datasetStore.currentSubpath" type="success" size="large" @click="openFolderAsDataset">
          <el-icon><FolderOpened /></el-icon> 打开为数据集
        </el-button>
        <el-divider direction="vertical" />
        <div class="toolbar-section">
          <input type="file" ref="folderInput" webkitdirectory directory hidden @change="handleFolderSelect" />
          <el-button type="primary" size="large" @click="triggerFolderUpload" :loading="isUploadingFolder">
            <el-icon><Upload /></el-icon> 上传文件夹
          </el-button>
        </div>
      </div>

      <!-- Breadcrumb Navigation -->
      <div class="breadcrumb-bar glass-card" v-if="datasetStore.breadcrumb.length > 0">
        <el-breadcrumb separator="/">
          <el-breadcrumb-item @click="navigateTo('')">
            <span class="breadcrumb-link"><el-icon><HomeFilled /></el-icon> 数据集</span>
          </el-breadcrumb-item>
          <el-breadcrumb-item
            v-for="(segment, idx) in datasetStore.breadcrumb" :key="idx"
            @click="navigateTo(datasetStore.breadcrumb.slice(0, idx + 1).join('/'))"
          >
            <span class="breadcrumb-link">{{ segment }}</span>
          </el-breadcrumb-item>
        </el-breadcrumb>
      </div>

      <!-- Combined Folder + Dataset Grid -->
      <div class="folder-grid" v-if="datasetStore.subfolders.length > 0 || localDatasets.length > 0">
        <!-- Subfolders first -->
        <div class="folder-card subfolder-card glass-card" v-for="folder in datasetStore.subfolders" :key="'f-' + folder.subpath" @click="navigateTo(folder.subpath)">
          <div class="folder-icon subfolder-icon"><el-icon :size="48"><FolderOpened /></el-icon></div>
          <div class="folder-info">
            <div class="folder-name">{{ folder.name }}</div>
            <div class="folder-meta">{{ folder.childCount }} 个子项<template v-if="folder.hasImages"> · {{ folder.imageCount }} 张图片</template></div>
          </div>
          <el-button class="delete-btn" type="danger" :icon="Delete" circle size="small" @click.stop="confirmDeleteFolder(folder)" />
        </div>
        <!-- Datasets after -->
        <div class="folder-card glass-card" v-for="ds in localDatasets" :key="ds.name" @click="openDataset(ds)">
          <div class="folder-icon"><el-icon :size="48"><Folder /></el-icon></div>
          <div class="folder-info">
            <div class="folder-name">{{ ds.name }}</div>
            <div class="folder-meta">{{ ds.imageCount }} 张图片</div>
          </div>
          <el-button class="delete-btn" type="danger" :icon="Delete" circle size="small" @click.stop="confirmDeleteDataset(ds)" />
        </div>
      </div>

      <!-- Empty State -->
      <div class="empty-state glass-card" v-if="localDatasets.length === 0 && datasetStore.subfolders.length === 0">
        <el-icon :size="64"><FolderOpened /></el-icon>
        <h3>暂无内容</h3>
        <p>点击「新建数据集」或「新建文件夹」开始</p>
      </div>
    </template>

    <!-- Dataset Detail View -->
    <template v-else>
      <!-- Header -->
      <div class="detail-header glass-card">
        <div class="header-left">
          <el-button @click="goBack" class="back-btn"><el-icon><ArrowLeft /></el-icon></el-button>
          <div class="header-info">
            <h2>{{ currentView.name }}</h2>
            <span class="path-text">{{ currentView.path }}</span>
          </div>
        </div>
        <div class="header-right">
          <!-- 上传通道文件 (多通道模式) -->
          <el-button @click="channelUploadVisible = true">
            <el-icon><FolderOpened /></el-icon> 上传到通道
          </el-button>
          <!-- 上传图片文件 -->
          <el-upload :http-request="customUpload" :multiple="true" :show-file-list="false" :before-upload="beforeUpload" accept="image/*,.txt,.safetensors">
            <el-button type="primary" :loading="isUploading">
              <el-icon><Upload /></el-icon> 上传文件
            </el-button>
          </el-upload>
        </div>
      </div>

      <!-- Stats Bar -->
      <div class="dataset-info glass-card" v-if="datasetStore.currentDataset">
      <div class="info-header">
        <div class="info-stats">
          <div class="stat"><el-icon><Picture /></el-icon><span>{{ datasetStore.currentDataset.imageCount }} 张图片</span></div>
          <div class="stat"><el-icon><Folder /></el-icon><span>{{ dm.formatSize(datasetStore.currentDataset.totalSize) }}</span></div>
          <el-button type="primary" link @click="forceRefreshStats" :loading="datasetStore.isLoadingStats" title="刷新统计"><el-icon><Refresh /></el-icon></el-button>
          <!-- Multi-channel mode inline indicator -->
          <el-tag v-if="datasetStore.isMultiChannel" type="success" size="small" effect="dark" style="margin-left: 4px;">
            {{ datasetStore.currentChannels.map(ch => ch.name).join(' + ') }}
          </el-tag>
          <el-tag v-else-if="datasetStore.isPairedMode" type="primary" size="small" effect="dark" style="margin-left: 4px;">
            {{ datasetStore.isOmniMode ? 'Omni' : 'Img2Img' }}
          </el-tag>
          <div class="stat" :class="{ 'stat-success': latentCachedCount === datasetStore.currentDataset.imageCount }">
            <el-icon><Box /></el-icon>
            <span v-if="datasetStore.isLoadingStats">Latent: <el-icon class="is-loading"><Loading /></el-icon></span>
            <span v-else>Latent: {{ latentCachedCount ?? '...' }} / {{ datasetStore.currentDataset.imageCount }}</span>
          </div>
          <div class="stat" :class="{ 'stat-success': textCachedCount === datasetStore.currentDataset.imageCount }">
            <el-icon><Document /></el-icon>
            <span v-if="datasetStore.isLoadingStats">Text: <el-icon class="is-loading"><Loading /></el-icon></span>
            <span v-else>Text: {{ textCachedCount ?? '...' }} / {{ datasetStore.currentDataset.imageCount }}</span>
          </div>
        </div>
        <div class="info-actions">
          <el-button @click="toggleSelectAll" size="small">{{ isAllSelected ? '取消全选' : '全选' }}</el-button>
          <el-button type="danger" size="small" @click="deleteSelected" :disabled="datasetStore.selectedImages.size === 0">
            <el-icon><Delete /></el-icon> 删除 ({{ datasetStore.selectedImages.size }})
          </el-button>
          <el-button type="primary" size="small" @click="showCacheDialog = true" :loading="isGeneratingCache">
            <el-icon><Box /></el-icon> 一键生成缓存
          </el-button>
          <el-button type="danger" size="small" @click="showClearCacheDialog = true">
            <el-icon><Delete /></el-icon> 清理缓存
          </el-button>
          <el-button type="warning" size="small" @click="showOllamaDialog = true">
            <el-icon><MagicStick /></el-icon> Ollama 标注
          </el-button>
          <el-button type="info" size="small" @click="showResizeDialog = true">
            <el-icon><ScaleToOriginal /></el-icon> 图片缩放
          </el-button>
          <el-button type="danger" size="small" @click="confirmDeleteCaptions" plain>
            <el-icon><Delete /></el-icon> 删除标注
          </el-button>
          <el-button type="success" size="small" @click="showBucketCalculator = true">
            <el-icon><Grid /></el-icon> 分桶计算器
          </el-button>
        </div>
      </div>
      
      <!-- Cache Progress -->
      <div class="cache-progress-section" v-if="cacheStatus.latent.status === 'running' || cacheStatus.text.status === 'running'">
        <div class="cache-progress-item" v-if="cacheStatus.latent.status === 'running'">
          <div class="progress-label">
            <el-icon class="spinning"><Loading /></el-icon><span>Latent 缓存</span>
            <span class="progress-count" v-if="cacheStatus.latent.current && cacheStatus.latent.total">{{ cacheStatus.latent.current }} / {{ cacheStatus.latent.total }}</span>
          </div>
          <el-progress :percentage="cacheStatus.latent.progress || 0" :stroke-width="8" color="#f0b429" />
        </div>
        <div class="cache-progress-item" v-if="cacheStatus.text.status === 'running'">
          <div class="progress-label">
            <el-icon class="spinning"><Loading /></el-icon><span>Text 缓存</span>
            <span class="progress-count" v-if="cacheStatus.text.current && cacheStatus.text.total">{{ cacheStatus.text.current }} / {{ cacheStatus.text.total }}</span>
          </div>
          <el-progress :percentage="cacheStatus.text.progress || 0" :stroke-width="8" color="#67c23a" />
        </div>
        <div class="cache-progress-item queued" v-if="cacheStatus.latent.status === 'running' && cacheStatus.text.status !== 'running' && isGeneratingCache">
          <div class="progress-label"><el-icon><Clock /></el-icon><span>Text 缓存（排队中，等待 Latent 完成）</span></div>
        </div>
      </div>
    </div>

    <!-- Image Grid -->
    <div class="image-grid-container" v-if="hasContent">

      
      <!-- Pagination Top -->
      <div class="pagination-top">
        <el-pagination
          :current-page="currentPage" :page-size="pageSize"
          :total="pagination?.totalCount || datasetStore.currentDataset?.imageCount || 0"
          :page-sizes="[50, 100, 200, 500]" layout="total, sizes, prev, pager, next, jumper" background
          @size-change="handlePageSizeChange" @current-change="handlePageChange"
        />
      </div>
      
      <!-- Multi-Channel Mode (list view) -->
      <div class="mc-list" v-if="datasetStore.isMultiChannel">
        <MultiChannelCard 
          v-for="group in datasetStore.currentGroups" 
          :key="group.id" 
          :group="group"
          :channels="datasetStore.currentChannels"
          :selected="datasetStore.selectedImages.has(group.target?.path || '')"
          @click="previewGroup(group)" 
          @toggle-select="toggleSelection({ path: $event } as DatasetImage)"
          @edit-caption="editGroupCaption(group)"
        />
      </div>

      <!-- Omni Mode (legacy) -->
      <div class="image-grid paired-grid" v-else-if="datasetStore.isOmniMode">
        <OmniImageCard v-for="pair in paginatedPairs" :key="pair.id" :pair="pair"
          :selected="datasetStore.selectedImages.has(pair.target.path)"
          @click="previewPair(pair)" @toggle-select="toggleSelection({ path: $event } as DatasetImage)"
        />
      </div>
      
      <!-- Paired Mode (legacy) -->
      <div class="image-grid paired-grid" v-else-if="datasetStore.isPairedMode">
        <PairedImageCard v-for="pair in paginatedPairs" :key="pair.id" :pair="pair"
          :selected="datasetStore.selectedImages.has(pair.target.path)"
          @click="previewPair(pair)" @toggle-select="toggleSelection({ path: $event } as DatasetImage)"
        />
      </div>
      
      <!-- Standard Mode -->
      <div class="image-grid" v-else>
        <div class="image-card glass-card" v-for="image in paginatedImages" :key="image.path"
          :class="{ selected: datasetStore.selectedImages.has(image.path) }"
        >
          <div class="image-wrapper" @click="previewImage(image)">
            <img :src="dm.getImageUrl(image)" :alt="image.filename" loading="lazy"
              @error="dm.handleImageError($event, image)" :data-retry="dm.imageRetryCount.value.get(image.path) || 0"
            />
            <div class="image-error-overlay" v-if="dm.imageLoadFailed.value.has(image.path)">
              <el-icon><WarningFilled /></el-icon><span>加载失败</span>
              <el-button size="small" @click.stop="dm.retryLoadImage(image)">重试</el-button>
            </div>
            <div class="select-circle" :class="{ checked: datasetStore.selectedImages.has(image.path) }" @click.stop="toggleSelection(image)">
              <el-icon v-if="datasetStore.selectedImages.has(image.path)"><Check /></el-icon>
            </div>
            <div class="cache-tags">
              <div class="cache-tag" :class="{ active: image.hasLatentCache }" title="Latent缓存"><el-icon><Box /></el-icon><span>L</span></div>
              <div class="cache-tag" :class="{ active: image.hasTextCache }" title="Text缓存"><el-icon><Document /></el-icon><span>T</span></div>
            </div>
          </div>
          <div class="image-info">
            <div class="image-name" :title="image.filename">{{ image.filename }}</div>
            <div class="image-meta">{{ image.width && image.height ? `${image.width}×${image.height} · ` : '' }}{{ dm.formatSize(image.size) }}</div>
            <div class="image-caption" :class="{ 'no-caption': !image.caption }">{{ image.caption || '无标注' }}</div>
          </div>
          <div class="image-actions"></div>
        </div>
      </div>
      
      <!-- Pagination Bottom -->
      <div class="pagination-bottom">
        <el-pagination
          :current-page="currentPage" :page-size="pageSize"
          :total="pagination?.totalCount || datasetStore.currentDataset?.imageCount || 0"
          :page-sizes="[50, 100, 200, 500]" layout="total, sizes, prev, pager, next, jumper" background
          @size-change="handlePageSizeChange" @current-change="handlePageChange"
        />
      </div>
    </div>

    <!-- Loading State -->
    <div class="loading-state glass-card" v-else-if="datasetStore.isLoading">
      <el-icon :size="64" class="is-loading"><Loading /></el-icon>
      <h3>正在加载数据集...</h3><p>请稍候，正在扫描图片和缓存信息</p>
    </div>

    <!-- Empty State -->
    <div class="empty-state glass-card" v-else>
      <el-icon :size="64"><FolderOpened /></el-icon>
      <h3>暂无图片</h3><p>上传图片到数据集</p>
    </div>
    </template>

    <!-- ===== DIALOG COMPONENTS ===== -->
    
    <!-- Preview & Edit Dialog -->
    <PreviewDialog v-model="previewDialogVisible" :image="editingImage" @saved="onCaptionSaved" />
    <MultiChannelPreviewDialog 
      v-model="mcPreviewVisible" 
      :group="mcPreviewGroup" 
      :channels="datasetStore.currentChannels"
      @edit-caption="onMcEditCaption"
    />
    <CaptionEditDialog
      v-model="captionEditVisible"
      :group="captionEditGroup"
      @saved="onCaptionSaved"
    />

    <!-- Caption Generation Dialog -->
    <el-dialog v-model="showCaptionDialog" title="批量生成标注" width="500px">
      <el-form>
        <el-form-item label="模型">
          <el-select v-model="captionModel" style="width: 100%">
            <el-option label="Qwen-VL (推荐)" value="qwen" />
            <el-option label="BLIP-2" value="blip" />
          </el-select>
        </el-form-item>
        <el-form-item label="提示词">
          <el-input v-model="captionPrompt" type="textarea" :rows="3" placeholder="描述这张图片..." />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCaptionDialog = false">取消</el-button>
        <el-button type="primary" @click="generateCaptions" :loading="isGenerating">开始生成</el-button>
      </template>
    </el-dialog>

    <!-- Create Dataset Dialog -->
    <el-dialog v-model="showCreateDialog" title="新建数据集" width="400px">
      <el-form>
        <el-form-item v-if="datasetStore.currentSubpath" label="位置">
          <el-tag type="info">{{ datasetStore.currentSubpath }}</el-tag>
        </el-form-item>
        <el-form-item label="数据集名称">
          <el-input v-model="newDatasetName" placeholder="输入数据集名称..." @keyup.enter="createDataset" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="createDataset" :loading="isCreating">创建</el-button>
      </template>
    </el-dialog>

    <!-- Create Folder Dialog -->
    <el-dialog v-model="showCreateFolderDialog" title="新建文件夹" width="400px">
      <el-form>
        <el-form-item v-if="datasetStore.currentSubpath" label="位置">
          <el-tag type="info">{{ datasetStore.currentSubpath }}</el-tag>
        </el-form-item>
        <el-form-item label="文件夹名称">
          <el-input v-model="newFolderName" placeholder="输入文件夹名称..." @keyup.enter="createFolder" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateFolderDialog = false">取消</el-button>
        <el-button type="primary" @click="createFolder" :loading="isCreatingFolder">创建</el-button>
      </template>
    </el-dialog>

    <!-- Cache Generation Dialog -->
    <CacheDialog
      v-model="showCacheDialog"
      :generating="isGeneratingCache"
      @confirm="handleCacheConfirm"
    />

    <!-- Cache Clear Dialog -->
    <el-dialog v-model="showClearCacheDialog" title="清理缓存" width="500px">
      <el-form label-position="top">
        <el-form-item label="模型类型">
          <el-select v-model="cacheModelType" placeholder="请选择模型类型">
            <el-option label="Z-Image" value="zimage" />
          </el-select>
          <div class="cache-model-hint"><span>将清理 _zi 后缀的缓存文件</span></div>
        </el-form-item>
        <el-form-item label="选择清理类型">
          <div class="flex flex-col gap-2">
            <el-checkbox v-model="clearCacheOptions.latent">Latent 缓存</el-checkbox>
            <el-checkbox v-model="clearCacheOptions.text">Text 缓存</el-checkbox>
          </div>
        </el-form-item>
        <el-alert title="清理后需要重新生成才能用于训练" type="warning" :closable="false" show-icon />
      </el-form>
      <template #footer>
        <el-button @click="showClearCacheDialog = false">取消</el-button>
        <el-button type="danger" @click="startClearCache" :loading="isClearingCache" :disabled="!clearCacheOptions.latent && !clearCacheOptions.text">确认清理</el-button>
      </template>
    </el-dialog>

    <!-- Resize Dialog -->
    <ResizeDialog
      v-model="showResizeDialog"
      :resizing="resizing"
      :status="resizeStatus"
      @confirm="handleResizeConfirm"
      @stop="stopResize"
    />

    <!-- Ollama Dialog -->
    <OllamaDialog
      v-model="showOllamaDialog"
      :dataset-path="currentView?.path || ''"
      :tagging="ollamaTagging"
      :status="ollamaStatus"
      @start="handleOllamaStart"
      @stop="stopOllamaTagging"
    />

    <!-- Channel Upload Dialog -->
    <ChannelUploadDialog
      v-model="channelUploadVisible"
      :dataset-name="currentView?.name || ''"
      :channels="datasetStore.currentChannels"
      @uploaded="handleChannelUploaded"
    />

    <!-- Bucket Calculator Dialog -->
    <BucketDialog
      v-model="showBucketCalculator"
      :dataset-path="currentView?.path || ''"
    />

    <!-- Upload Progress Dialog -->
    <el-dialog v-model="showUploadProgress" title="上传文件" width="500px" :close-on-click-modal="false" :close-on-press-escape="false" :show-close="!isUploadingFolder">
      <div class="upload-progress-content">
        <div class="progress-info"><span class="progress-text">{{ uploadProgressText }}</span><span class="progress-percent">{{ uploadProgress }}%</span></div>
        <el-progress :percentage="uploadProgress" :status="uploadStatus" :stroke-width="20" striped striped-flow />
        <div class="upload-stats" v-if="uploadStats.total > 0">
          <span>成功: <strong class="success">{{ uploadStats.success }}</strong></span>
          <span>失败: <strong class="fail">{{ uploadStats.fail }}</strong></span>
          <span>总计: <strong>{{ uploadStats.total }}</strong></span>
        </div>
      </div>
      <template #footer v-if="!isUploadingFolder">
        <el-button type="primary" @click="showUploadProgress = false">完成</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useDatasetStore, type DatasetImage, type LocalDataset, type ImagePair, type ImageGroup } from '@/stores/dataset'
import { useTrainingStore } from '@/stores/training'
import { useWebSocketStore } from '@/stores/websocket'
import { useDatasetManager } from '@/composables/useDatasetManager'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, WarningFilled, MagicStick, ScaleToOriginal, Loading, Clock, Grid,
  Plus, Refresh, Upload, Folder, FolderOpened, ArrowLeft, Picture, Box, Document, Check, HomeFilled, FolderAdd } from '@element-plus/icons-vue'
import axios from 'axios'
import type { FolderItem } from '@/stores/dataset'

// Sub-components
import { PairedImageCard, OmniImageCard } from './dataset/components'
import MultiChannelCard from './dataset/components/MultiChannelCard.vue'
import { PreviewDialog, BucketDialog, CacheDialog, ResizeDialog, OllamaDialog, ChannelUploadDialog, MultiChannelPreviewDialog, CaptionEditDialog } from './dataset/dialogs'

// Stores
const datasetStore = useDatasetStore()
const trainingStore = useTrainingStore()
const wsStore = useWebSocketStore()

// Composable
const dm = useDatasetManager()

// ============================================================================
// View State
// ============================================================================
const currentView = ref<LocalDataset | null>(null)
const datasetPath = ref('')
const localDatasets = computed(() => datasetStore.localDatasets)
const datasetsDir = computed(() => datasetStore.datasetsDir)

// Pagination
const currentPage = computed(() => datasetStore.currentPage)
const pageSize = computed(() => datasetStore.pageSize)
const pagination = computed(() => datasetStore.pagination)
const paginatedImages = computed(() => datasetStore.currentImages)
const paginatedPairs = computed(() => datasetStore.currentPairs)
const hasContent = computed(() => {
  if (datasetStore.isMultiChannel) return datasetStore.currentGroups.length > 0
  if (datasetStore.isPairedMode) return datasetStore.currentPairs.length > 0
  return pagination.value?.totalCount ? pagination.value.totalCount > 0 : datasetStore.currentImages.length > 0
})

// Cache counts
const cacheStatus = computed(() => wsStore.cacheStatus)
const latentCachedCount = computed(() => datasetStore.currentDataset?.totalLatentCached ?? datasetStore.currentImages.filter(img => img.hasLatentCache).length)
const textCachedCount = computed(() => datasetStore.currentDataset?.totalTextCached ?? datasetStore.currentImages.filter(img => img.hasTextCache).length)

// Watch dataset path for stats
watch(() => datasetStore.currentDataset?.path, (newPath) => { if (newPath) datasetStore.fetchStats(newPath) }, { immediate: true })

// ============================================================================
// Dialog State
// ============================================================================
const previewDialogVisible = ref(false)
const mcPreviewVisible = ref(false)
const mcPreviewGroup = ref<ImageGroup | null>(null)
const captionEditVisible = ref(false)
const captionEditGroup = ref<ImageGroup | null>(null)
const showCaptionDialog = ref(false)
const showCreateDialog = ref(false)
const showCacheDialog = ref(false)
const showClearCacheDialog = ref(false)
const showOllamaDialog = ref(false)
const channelUploadVisible = ref(false)
const showResizeDialog = ref(false)
const showBucketCalculator = ref(false)
const showUploadProgress = ref(false)

const editingImage = ref<DatasetImage | null>(null)
const captionModel = ref('qwen')
const captionPrompt = ref('详细描述这张图片的内容、风格和氛围')
const isGenerating = ref(false)
const isGeneratingCache = ref(false)
const isClearingCache = ref(false)
const cacheModelType = ref('zimage')
const clearCacheOptions = ref({ latent: true, text: true })
const newDatasetName = ref('')
const isCreating = ref(false)
const showCreateFolderDialog = ref(false)
const newFolderName = ref('')
const isCreatingFolder = ref(false)

// Resize state
const resizing = ref(false)
const resizeStatus = ref({ running: false, total: 0, completed: 0, current_file: '', errors: [] as string[] })

// Ollama state
const ollamaTagging = ref(false)
const ollamaStatus = ref({ running: false, total: 0, completed: 0, current_file: '', errors: [] as string[] })

// Upload state
const folderInput = ref<HTMLInputElement | null>(null)
const channelFolderInput = ref<HTMLInputElement | null>(null)
const isUploadingFolder = ref(false)
const isUploading = ref(false)
const uploadQueue = ref<File[]>([])
const uploadProgress = ref(0)
const uploadProgressText = ref('准备上传...')
const uploadStatus = ref<'' | 'success' | 'exception'>('')
const uploadStats = ref({ success: 0, fail: 0, total: 0 })

// ============================================================================
// Selection
// ============================================================================
const isAllSelected = computed(() => datasetStore.currentImages.length > 0 && datasetStore.selectedImages.size === datasetStore.currentImages.length)
function toggleSelection(image: DatasetImage) { datasetStore.toggleImageSelection(image.path) }
function toggleSelectAll() { isAllSelected.value ? datasetStore.clearSelection() : datasetStore.selectAll() }

// ============================================================================
// Navigation
// ============================================================================
async function refreshCurrentLevel() { try { await datasetStore.browseFolder(datasetStore.currentSubpath) } catch {} }
async function navigateTo(subpath: string) { try { await datasetStore.browseFolder(subpath) } catch {} }
async function openFolderAsDataset() {
  const subpath = datasetStore.currentSubpath
  if (!subpath) return
  const base = datasetStore.datasetsDir
  const fullPath = `${base}/${subpath}`
  const name = subpath.split('/').pop() || subpath
  const virtualDs: LocalDataset = { name: `📁 ${name}`, path: fullPath, imageCount: 0 }
  currentView.value = virtualDs; datasetPath.value = fullPath; await datasetStore.scanDataset(fullPath)
}
async function openDataset(ds: LocalDataset) { currentView.value = ds; datasetPath.value = ds.path; await datasetStore.scanDataset(ds.path) }
function goBack() { currentView.value = null; datasetStore.clearCurrentDataset(); datasetStore.browseFolder(datasetStore.currentSubpath) }
async function handlePageChange(page: number) { await datasetStore.loadPage(page); window.scrollTo({ top: 0, behavior: 'smooth' }) }
async function handlePageSizeChange(size: number) { await datasetStore.changePageSize(size) }

// ============================================================================
// Preview
// ============================================================================
function previewImage(image: DatasetImage) { editingImage.value = image; previewDialogVisible.value = true }
function previewPair(pair: ImagePair) { editingImage.value = pair.target; previewDialogVisible.value = true }
function previewGroup(group: ImageGroup) { mcPreviewGroup.value = group; mcPreviewVisible.value = true }
function onMcEditCaption(group: ImageGroup) { captionEditGroup.value = group; captionEditVisible.value = true }
function editGroupCaption(group: ImageGroup) { captionEditGroup.value = group; captionEditVisible.value = true }
function onCaptionSaved() { /* PreviewDialog handles save internally via store */ }

// ============================================================================
// Dataset CRUD
// ============================================================================
async function confirmDeleteDataset(ds: LocalDataset) {
  try {
    const deletePath = datasetStore.currentSubpath ? `${datasetStore.currentSubpath}/${ds.name}` : ds.name
    await ElMessageBox.confirm(`确定要删除数据集「${ds.name}」吗？此操作不可恢复！`, '删除确认', { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning', confirmButtonClass: 'el-button--danger' })
    await axios.delete(`/api/dataset/${encodeURIComponent(deletePath)}`)
    ElMessage.success(`数据集「${ds.name}」已删除`)
    await datasetStore.browseFolder(datasetStore.currentSubpath)
  } catch {}
}

async function createDataset() {
  if (!newDatasetName.value.trim()) { ElMessage.warning('请输入数据集名称'); return }
  isCreating.value = true
  try {
    const formData = new FormData(); formData.append('name', newDatasetName.value.trim()); formData.append('parent_path', datasetStore.currentSubpath)
    const response = await axios.post('/api/dataset/create', formData)
    ElMessage.success(`数据集「${response.data.name}」创建成功`)
    await datasetStore.browseFolder(datasetStore.currentSubpath); showCreateDialog.value = false; newDatasetName.value = ''
  } catch (e: any) { ElMessage.error(e.response?.data?.detail || '创建失败') }
  finally { isCreating.value = false }
}

async function createFolder() {
  if (!newFolderName.value.trim()) { ElMessage.warning('请输入文件夹名称'); return }
  isCreatingFolder.value = true
  try {
    await datasetStore.createFolder(newFolderName.value.trim())
    ElMessage.success(`文件夹「${newFolderName.value.trim()}」创建成功`)
    showCreateFolderDialog.value = false; newFolderName.value = ''
  } catch (e: any) { ElMessage.error(e.response?.data?.detail || '创建失败') }
  finally { isCreatingFolder.value = false }
}

async function confirmDeleteFolder(folder: FolderItem) {
  try {
    await ElMessageBox.confirm(`确定要删除文件夹「${folder.name}」吗？其中的所有内容将被删除！`, '删除确认', { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning', confirmButtonClass: 'el-button--danger' })
    await axios.delete(`/api/dataset/${encodeURIComponent(folder.subpath)}`)
    ElMessage.success(`文件夹「${folder.name}」已删除`)
    await datasetStore.browseFolder(datasetStore.currentSubpath)
  } catch {}
}

async function deleteSelected() {
  if (datasetStore.selectedImages.size === 0) return
  try {
    await ElMessageBox.confirm(`确定要删除选中的 ${datasetStore.selectedImages.size} 张图片吗？此操作不可恢复！`, '确认删除', { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' })
    const paths = Array.from(datasetStore.selectedImages)
    const response = await axios.post('/api/dataset/delete-images', { paths })
    if (response.data.deleted > 0) { ElMessage.success(`成功删除 ${response.data.deleted} 张图片`); datasetStore.clearSelection(); if (currentView.value) await datasetStore.scanDataset(currentView.value.path) }
    if (response.data.errors?.length > 0) ElMessage.warning(`${response.data.errors.length} 张图片删除失败`)
  } catch (e: any) { if (e !== 'cancel') ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message || '未知错误')) }
}

async function confirmDeleteCaptions() {
  if (!currentView.value) return
  try {
    await ElMessageBox.confirm('确定要删除该数据集中所有 .txt 标注文件吗？此操作不可撤销！', '删除标注', { confirmButtonText: '确认删除', cancelButtonText: '取消', type: 'warning', confirmButtonClass: 'el-button--danger' })
  } catch { return }
  try {
    const res = await axios.post('/api/dataset/delete-captions', { dataset_path: currentView.value.path })
    if (res.data.deleted > 0) { ElMessage.success(`成功删除 ${res.data.deleted} 个标注文件`); await datasetStore.scanDataset(currentView.value.path) }
    else ElMessage.info('没有标注文件需要删除')
    if (res.data.errors?.length > 0) ElMessage.warning(`${res.data.errors.length} 个文件删除失败`)
  } catch (e: any) { ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message)) }
}

async function generateCaptions() {
  isGenerating.value = true
  try {
    await datasetStore.generateCaptions(captionModel.value as 'qwen' | 'blip')
    ElMessage.success('标注生成完成'); showCaptionDialog.value = false
    await datasetStore.scanDataset(datasetPath.value)
  } catch (e: any) { ElMessage.error(e.message || '生成失败') }
  finally { isGenerating.value = false }
}

// ============================================================================
// Cache Generation (via CacheDialog sub-component)
// ============================================================================
let cacheRefreshIntervalId: number | null = null

async function handleCacheConfirm(config: { modelType: string, trainingMode: string, options: string[], modelPath: string, controlDir?: string, sourceDir?: string, maskDir?: string, conditionDirs?: string, numConditionImages?: number }) {
  if (!currentView.value) return
  isGeneratingCache.value = true
  try {
    await axios.post('/api/cache/generate', {
      datasetPath: currentView.value.path,
      generateLatent: config.options.includes('latent'),
      generateText: config.options.includes('text'),
      generateDino: config.options.includes('dino'),
      modelPath: config.modelPath,
      modelType: config.modelType,
      mode: config.trainingMode,
      controlDir: config.controlDir || undefined,
      sourceDir: config.sourceDir || undefined,
      maskDir: config.maskDir || undefined,
      conditionDirs: config.conditionDirs || undefined,
      numConditionImages: config.numConditionImages || 0,
    })
    ElMessage.success('缓存生成任务已启动'); showCacheDialog.value = false
    if (cacheRefreshIntervalId !== null) dm.clearTrackedInterval(cacheRefreshIntervalId)
    cacheRefreshIntervalId = dm.trackInterval(window.setInterval(async () => {
      if (currentView.value) await datasetStore.scanDataset(currentView.value.path)
    }, 3000))
    dm.trackTimeout(window.setTimeout(() => {
      if (cacheRefreshIntervalId !== null) { dm.clearTrackedInterval(cacheRefreshIntervalId); cacheRefreshIntervalId = null }
      isGeneratingCache.value = false
    }, 30000))
  } catch (e: any) { ElMessage.error('启动失败: ' + (e.response?.data?.detail || e.message)); isGeneratingCache.value = false }
}



async function startClearCache() {
  if (!currentView.value) return
  isClearingCache.value = true
  try {
    const response = await axios.post('/api/cache/clear', { datasetPath: currentView.value.path, clearLatent: clearCacheOptions.value.latent, clearText: clearCacheOptions.value.text, modelType: cacheModelType.value })
    const { deleted, errors } = response.data
    if (errors?.length > 0) ElMessage.warning(`清理完成，但有 ${errors.length} 个文件失败`)
    else ElMessage.success(`成功清理 ${deleted} 个缓存文件`)
    showClearCacheDialog.value = false; await datasetStore.scanDataset(currentView.value.path)
  } catch (e: any) { ElMessage.error('清理失败: ' + (e.response?.data?.detail || e.message)) }
  finally { isClearingCache.value = false }
}

// ============================================================================
// Ollama (via OllamaDialog sub-component)
// ============================================================================
let ollamaPollingInterval: number | null = null

async function handleOllamaStart(config: any) {
  if (!currentView.value) return
  ollamaTagging.value = true; ollamaStatus.value = { running: true, total: 0, completed: 0, current_file: '', errors: [] }
  try {
    const res = await axios.post('/api/dataset/ollama/tag', {
      dataset_path: currentView.value.path, ollama_url: config.url, model: config.model,
      prompt: config.prompt, max_long_edge: config.maxLongEdge, skip_existing: config.skipExisting, trigger_word: config.triggerWord
    })
    if (res.data.total === 0) { ElMessage.info('没有需要标注的图片'); ollamaTagging.value = false; return }
    ElMessage.success(`开始标注 ${res.data.total} 张图片`); ollamaStatus.value.total = res.data.total
    startOllamaStatusPolling()
  } catch (e: any) { ElMessage.error('启动失败: ' + (e.response?.data?.detail || e.message)); ollamaTagging.value = false }
}

function startOllamaStatusPolling() {
  if (ollamaPollingInterval) return
  ollamaPollingInterval = dm.trackInterval(window.setInterval(async () => {
    try {
      const res = await axios.get('/api/dataset/ollama/status'); ollamaStatus.value = res.data
      if (!res.data.running) {
        stopOllamaStatusPolling(); ollamaTagging.value = false
        ElMessage.success(`标注完成！成功: ${res.data.completed}`)
        if (currentView.value) await datasetStore.scanDataset(currentView.value.path)
      }
    } catch { stopOllamaStatusPolling(); ollamaTagging.value = false }
  }, 2000))
}

function stopOllamaStatusPolling() { if (ollamaPollingInterval) { dm.clearTrackedInterval(ollamaPollingInterval); ollamaPollingInterval = null } }

async function stopOllamaTagging() {
  try { await axios.post('/api/dataset/ollama/stop'); ElMessage.info('正在停止...'); stopOllamaStatusPolling(); ollamaTagging.value = false } catch {}
}

async function checkOllamaTaggingStatus() {
  try {
    const res = await axios.get('/api/dataset/ollama/status')
    if (res.data.running) { ollamaTagging.value = true; ollamaStatus.value = res.data; startOllamaStatusPolling() }
    else if (res.data.total > 0 && res.data.completed > 0) ollamaStatus.value = res.data
  } catch {}
}

// ============================================================================
// Resize (via ResizeDialog sub-component)
// ============================================================================
async function handleResizeConfirm(config: { maxLongEdge: number, quality: number, sharpen: number }) {
  if (!currentView.value) return
  try {
    await ElMessageBox.confirm('此操作将直接覆盖原图，不可撤销！确定要继续吗？', '警告', { confirmButtonText: '确认缩放', cancelButtonText: '取消', type: 'warning', confirmButtonClass: 'el-button--danger' })
  } catch { return }
  resizing.value = true; resizeStatus.value = { running: true, total: 0, completed: 0, current_file: '', errors: [] }
  try {
    const res = await axios.post('/api/dataset/resize', { dataset_path: currentView.value.path, max_long_edge: config.maxLongEdge, quality: config.quality, sharpen: config.sharpen })
    if (res.data.total === 0) { ElMessage.info('没有图片需要处理'); resizing.value = false; return }
    ElMessage.success(`开始处理 ${res.data.total} 张图片`)
    let resizePollInterval: number | null = null
    resizePollInterval = dm.trackInterval(window.setInterval(async () => {
      try {
        const statusRes = await axios.get('/api/dataset/resize/status'); resizeStatus.value = statusRes.data
        if (!statusRes.data.running) {
          if (resizePollInterval !== null) { dm.clearTrackedInterval(resizePollInterval); resizePollInterval = null }
          resizing.value = false; ElMessage.success(`处理完成！共 ${statusRes.data.completed} 张`)
          if (currentView.value) await datasetStore.scanDataset(currentView.value.path)
        }
      } catch {}
    }, 500))
  } catch (e: any) { ElMessage.error('启动失败: ' + (e.response?.data?.detail || e.message)); resizing.value = false }
}

async function stopResize() { try { await axios.post('/api/dataset/resize/stop'); ElMessage.info('正在停止...') } catch {} }

// ============================================================================
// Upload
// ============================================================================
function triggerFolderUpload() { folderInput.value?.click() }

/**
 * Handle channel folder upload inside current dataset detail view.
 * Uploads files with preserved subdirectory structure.
 */
async function handleChannelFolderSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  if (!currentView.value) { input.value = ''; return }

  const files = Array.from(input.files)
  const validFiles = files.filter(f => f.type.startsWith('image/') || f.name.endsWith('.txt') || f.name.endsWith('.safetensors'))
  if (validFiles.length === 0) { ElMessage.warning('未找到有效的图片或标注文件'); input.value = ''; return }

  // Count subdirectories
  const subDirs = new Set<string>()
  validFiles.forEach(f => {
    const parts = (f as any).webkitRelativePath?.split('/') || []
    if (parts.length > 2) subDirs.add(parts[1]) // folder/subdir/file -> subdir
  })

  if (subDirs.size === 0) {
    ElMessage.warning('选择的文件夹没有子目录。多通道数据集需要 target/、depth/ 等子文件夹结构')
    input.value = ''
    return
  }

  try {
    await ElMessageBox.confirm(
      `检测到 ${subDirs.size} 个通道子目录：${[...subDirs].join('、')}\n将上传 ${validFiles.length} 个文件到「${currentView.value.name}」`,
      '上传通道文件夹',
      { confirmButtonText: '开始上传', cancelButtonText: '取消', type: 'info' }
    )
    await uploadChannelFiles(validFiles, currentView.value.name)
  } catch { input.value = '' }
}

async function uploadChannelFiles(files: File[], datasetName: string) {
  isUploadingFolder.value = true; showUploadProgress.value = true
  uploadProgress.value = 0; uploadProgressText.value = '准备上传通道文件...'
  uploadStatus.value = ''; uploadStats.value = { success: 0, fail: 0, total: files.length }
  const batchSize = 20; let successCount = 0; let failCount = 0
  try {
    for (let i = 0; i < files.length; i += batchSize) {
      const batch = files.slice(i, i + batchSize)
      const formData = new FormData()
      formData.append('dataset_name', datasetName)
      formData.append('preserve_structure', 'true')
      batch.forEach(f => {
        // Send webkitRelativePath as filename to preserve channel directory structure
        const blob = new File([f], (f as any).webkitRelativePath || f.name, { type: f.type })
        formData.append('files', blob)
      })
      uploadProgressText.value = `正在上传通道 ${i + 1}-${Math.min(i + batchSize, files.length)} / ${files.length}`
      try {
        const res = await axios.post('/api/dataset/upload_batch', formData)
        successCount += res.data.uploaded
        if (res.data.errors) failCount += res.data.errors.length
      } catch { failCount += batch.length }
      uploadProgress.value = Math.round(((i + batchSize) / files.length) * 100)
      uploadStats.value = { success: successCount, fail: failCount, total: files.length }
    }
    uploadProgress.value = 100; uploadProgressText.value = '通道上传完成'
    uploadStatus.value = failCount === 0 ? 'success' : (successCount > 0 ? '' : 'exception')
    uploadStats.value = { success: successCount, fail: failCount, total: files.length }
    ElMessage.success(`成功上传 ${successCount} 个通道文件`)
    // Refresh dataset view
    if (currentView.value) await datasetStore.scanDataset(currentView.value.path)
  } catch (e: any) {
    uploadProgress.value = 100; uploadProgressText.value = '上传出错: ' + e.message
    uploadStatus.value = 'exception'
  } finally {
    isUploadingFolder.value = false
    if (channelFolderInput.value) channelFolderInput.value.value = ''
  }
}

async function handleFolderSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (!input.files || input.files.length === 0) return
  const files = Array.from(input.files)
  const validFiles = files.filter(f => f.type.startsWith('image/') || f.name.endsWith('.txt') || f.name.endsWith('.safetensors'))
  if (validFiles.length === 0) { ElMessage.warning('未找到有效的图片或标注文件'); input.value = ''; return }

  // Detect if folder has subdirectories (multi-channel structure)
  const hasSubDirs = validFiles.some(f => {
    const parts = (f as any).webkitRelativePath?.split('/') || []
    return parts.length > 2  // folder/subfolder/file = 3 parts
  })

  const folderName = validFiles[0].webkitRelativePath?.split('/')[0] || 'New Dataset'
  try {
    const promptMsg = hasSubDirs
      ? '检测到子目录结构（多通道），将保留文件夹层级'
      : '上传文件夹'
    const { value: name } = await ElMessageBox.prompt('请输入数据集名称', promptMsg, {
      confirmButtonText: '开始上传',
      cancelButtonText: '取消',
      inputValue: folderName,
      inputValidator: (val) => !!val.trim() || '名称不能为空'
    }) as unknown as { value: string }
    if (name) await uploadFilesInBatches(validFiles, name, hasSubDirs)
  } catch { input.value = '' }
}

async function uploadFilesInBatches(files: File[], datasetName: string, preserveStructure: boolean = false) {
  isUploadingFolder.value = true; showUploadProgress.value = true; uploadProgress.value = 0; uploadProgressText.value = '准备上传...'; uploadStatus.value = ''; uploadStats.value = { success: 0, fail: 0, total: files.length }
  const batchSize = 20; let successCount = 0; let failCount = 0
  try {
    for (let i = 0; i < files.length; i += batchSize) {
      const batch = files.slice(i, i + batchSize)
      const formData = new FormData()
      formData.append('dataset_name', datasetName)
      if (preserveStructure) formData.append('preserve_structure', 'true')
      batch.forEach(f => {
        // Use webkitRelativePath as filename to preserve directory structure
        const blob = new File([f], preserveStructure ? ((f as any).webkitRelativePath || f.name) : f.name, { type: f.type })
        formData.append('files', blob)
      })
      uploadProgressText.value = `正在上传 ${i + 1}-${Math.min(i + batchSize, files.length)} / ${files.length}`
      try { const res = await axios.post('/api/dataset/upload_batch', formData); successCount += res.data.uploaded; if (res.data.errors) failCount += res.data.errors.length } catch { failCount += batch.length }
      uploadProgress.value = Math.round(((i + batchSize) / files.length) * 100); uploadStats.value = { success: successCount, fail: failCount, total: files.length }
    }
    uploadProgress.value = 100; uploadProgressText.value = '上传完成'; uploadStatus.value = failCount === 0 ? 'success' : (successCount > 0 ? '' : 'exception'); uploadStats.value = { success: successCount, fail: failCount, total: files.length }
    await datasetStore.browseFolder(datasetStore.currentSubpath)
    const newDs = localDatasets.value.find(d => d.name === datasetName); if (newDs) openDataset(newDs)
  } catch (e: any) { uploadProgress.value = 100; uploadProgressText.value = '上传出错: ' + e.message; uploadStatus.value = 'exception' }
  finally { isUploadingFolder.value = false; if (folderInput.value) folderInput.value.value = '' }
}

function beforeUpload(file: File) {
  const isImage = file.type.startsWith('image/'); const isTxt = file.name.endsWith('.txt'); const isSafetensors = file.name.endsWith('.safetensors')
  if (!isImage && !isTxt && !isSafetensors) { ElMessage.error('不支持的文件格式'); return false }
  if (file.size / 1024 / 1024 > 100) { ElMessage.error('文件大小不能超过 100MB'); return false }
  uploadQueue.value.push(file); setTimeout(() => processUploadQueue(), 100); return false
}

async function processUploadQueue() {
  if (uploadQueue.value.length === 0 || isUploading.value || !currentView.value) return
  const files = [...uploadQueue.value]; uploadQueue.value = []; isUploading.value = true
  try {
    const formData = new FormData(); formData.append('dataset', currentView.value.name); files.forEach(file => formData.append('files', file))
    const response = await axios.post('/api/dataset/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    if (response.data.uploaded?.length > 0) { ElMessage.success(`成功上传 ${response.data.uploaded.length} 个文件`); await loadLocalDatasets(); if (response.data.datasetPath) { datasetPath.value = response.data.datasetPath; await datasetStore.scanDataset(response.data.datasetPath) } }
    if (response.data.errors?.length > 0) ElMessage.warning(`${response.data.errors.length} 个文件上传失败`)
  } catch (e: any) { ElMessage.error('上传失败: ' + (e.response?.data?.detail || e.message || '未知错误')) }
  finally { isUploading.value = false }
}

async function handleChannelUploaded() {
  channelUploadVisible.value = false
  if (currentView.value) await datasetStore.scanDataset(currentView.value.path)
}
async function customUpload() { return Promise.resolve() }

// ============================================================================
// Stats / Watchers
// ============================================================================
async function forceRefreshStats() { if (currentView.value) { await datasetStore.fetchStats(currentView.value.path); ElMessage.success('统计已更新') } }

watch(() => cacheStatus.value, async (newStatus, oldStatus) => {
  if (!currentView.value) return
  if (oldStatus?.latent?.status === 'running' && newStatus?.latent?.status === 'completed') { await datasetStore.fetchStats(currentView.value.path); await datasetStore.loadPage(datasetStore.currentPage) }
  if (oldStatus?.text?.status === 'running' && newStatus?.text?.status === 'completed') { await datasetStore.fetchStats(currentView.value.path); await datasetStore.loadPage(datasetStore.currentPage); isGeneratingCache.value = false }
  if (newStatus?.latent?.status !== 'running' && newStatus?.text?.status !== 'running') { if (isGeneratingCache.value) { isGeneratingCache.value = false; datasetStore.fetchStats(currentView.value.path) } }
}, { deep: true })

// ============================================================================
// Lifecycle
// ============================================================================
onMounted(async () => { datasetStore.browseFolder(''); await checkOllamaTaggingStatus() })
onUnmounted(() => { dm.clearAllTrackedTimers(); stopOllamaStatusPolling(); cacheRefreshIntervalId = null })
</script>

<style lang="scss" scoped>
.dataset-page { max-width: 1400px; margin: 0 auto; }

.page-header {
  margin-bottom: var(--space-xl);
  h1 { font-family: var(--font-display); font-size: 2rem; margin-bottom: var(--space-xs); }
  .subtitle { color: var(--text-muted); font-size: 13px; }
}

.detail-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: var(--space-md) var(--space-lg); margin-bottom: var(--space-lg); gap: var(--space-lg);
  .header-left {
    display: flex; align-items: center; gap: var(--space-md); min-width: 0; flex: 1;
    .back-btn { flex-shrink: 0; width: 40px; height: 40px; padding: 0; display: flex; align-items: center; justify-content: center; }
    .header-info {
      min-width: 0;
      h2 { font-size: 18px; font-weight: 600; color: var(--text-primary); margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .path-text { font-size: 12px; color: var(--text-muted); display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    }
  }
  .header-right { display: flex; align-items: center; gap: var(--space-sm); flex-shrink: 0; }
}

.folder-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--space-lg); }

.folder-card {
  padding: var(--space-lg); cursor: pointer; transition: all var(--transition-fast);
  position: relative; display: flex; flex-direction: column; align-items: center; text-align: center;
  &:hover { transform: translateY(-4px); border-color: var(--primary); .delete-btn { opacity: 1; } }
  .folder-icon { color: var(--primary); margin-bottom: var(--space-md); }
  .folder-info {
    .folder-name { font-weight: 600; font-size: 15px; margin-bottom: var(--space-xs); word-break: break-all; }
    .folder-meta { font-size: 13px; color: var(--text-muted); }
  }
  .delete-btn { position: absolute; top: var(--space-sm); right: var(--space-sm); opacity: 0; transition: opacity var(--transition-fast); }
}

.dataset-toolbar {
  padding: var(--space-lg); margin-bottom: var(--space-lg);
  display: flex; align-items: center; gap: var(--space-md); flex-wrap: wrap;
  .el-divider--vertical { height: 32px; margin: 0; }
  .toolbar-section { display: flex; flex-direction: column; gap: var(--space-sm); }
}

.breadcrumb-bar {
  padding: var(--space-sm) var(--space-lg); margin-bottom: var(--space-lg);
  .breadcrumb-link { cursor: pointer; color: var(--primary); &:hover { text-decoration: underline; } }
  .el-breadcrumb__item:last-child .breadcrumb-link { color: var(--text-primary); cursor: default; &:hover { text-decoration: none; } }
}

.subfolder-card {
  border-left: 3px solid var(--el-color-warning);
  .subfolder-icon { color: var(--el-color-warning); }
}

.dataset-info {
  padding: var(--space-md) var(--space-lg); margin-bottom: var(--space-lg);
  .info-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: var(--space-md); }
  .info-stats {
    display: flex; align-items: center; gap: var(--space-lg); flex-wrap: wrap;
    .stat {
      display: flex; align-items: center; gap: var(--space-xs); font-size: 13px; color: var(--text-secondary);
      &.stat-success { color: var(--color-success); font-weight: 600; }
    }
  }
  .info-actions { display: flex; gap: var(--space-sm); flex-wrap: wrap; }
}

.cache-progress-section {
  margin-bottom: var(--space-lg); display: flex; flex-direction: column; gap: var(--space-sm);
  .cache-progress-item {
    padding: var(--space-sm) var(--space-md);
    background: var(--bg-tertiary); border-radius: var(--radius-md);
    .progress-label {
      display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-xs);
      font-size: 13px; font-weight: 500;
      .progress-count { color: var(--text-muted); font-size: 12px; }
    }
    &.queued { opacity: 0.6; }
  }
}

.spinning { animation: spin 1s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.dataset-type-badge {
  display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-md);
  .type-hint { font-size: 12px; color: var(--text-muted); }
}

.pagination-top, .pagination-bottom {
  display: flex; justify-content: center; padding: var(--space-md) 0;
}

.image-grid {
  display: grid; grid-template-columns: repeat(5, 1fr); gap: var(--space-md);
  &.paired-grid { grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
}

.mc-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.image-card {
  overflow: hidden; border-radius: var(--radius-md); transition: all var(--transition-fast);
  &:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
  &.selected { border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary); }

  .image-wrapper {
    position: relative; aspect-ratio: 1; overflow: hidden; cursor: pointer;
    img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s; }
    &:hover img { transform: scale(1.05); }
  }

  .image-error-overlay {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: var(--space-sm); color: #fff;
  }

  .select-circle {
    position: absolute; top: var(--space-sm); left: var(--space-sm);
    width: 24px; height: 24px; border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.7); background: rgba(0,0,0,0.3);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all var(--transition-fast);
    &.checked { background: var(--primary); border-color: var(--primary); }
    &:hover { border-color: var(--primary); }
  }

  .cache-tags {
    position: absolute; bottom: var(--space-xs); right: var(--space-xs); display: flex; gap: 2px;
    .cache-tag {
      display: flex; align-items: center; gap: 2px;
      padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700;
      background: rgba(0,0,0,0.5); color: rgba(255,255,255,0.5);
      &.active { background: rgba(var(--el-color-success-rgb), 0.8); color: #fff; }
    }
  }

  .image-info {
    padding: var(--space-sm) var(--space-md);
    .image-name { font-size: 12px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .image-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
    .image-caption { font-size: 11px; color: var(--text-secondary); margin-top: 4px; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; &.no-caption { color: var(--text-muted); font-style: italic; } }
  }
}

.loading-state, .empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 300px; padding: 64px var(--space-xxl); text-align: center; color: var(--text-muted);
  h3 { margin: var(--space-md) 0 var(--space-xs); }
}

// Cache clear dialog helpers
.cache-model-hint { margin-top: var(--space-xs); font-size: 12px; color: var(--text-muted); }

// Upload progress
.upload-progress-content {
  .progress-info { display: flex; justify-content: space-between; margin-bottom: var(--space-sm); }
  .progress-text { font-size: 14px; }
  .progress-percent { font-weight: bold; color: var(--primary); }
  .upload-stats {
    display: flex; gap: var(--space-lg); margin-top: var(--space-md); font-size: 13px;
    .success { color: var(--color-success); }
    .fail { color: var(--color-danger); }
  }
}
</style>
