<template>
  <div class="generation-container">
    <div class="main-content">
      <div class="generation-grid">
        <!-- Controls Section -->
        <div class="controls-section">
          <el-card class="params-card glass-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span><el-icon><Operation /></el-icon> 生成参数</span>
              </div>
            </template>
            
            <el-form :model="params" size="small" class="params-form">
              <!-- 模型类型选择器 -->
              <div class="param-group">
                <div class="group-label">模型类型 (MODEL)</div>
                <el-radio-group v-model="params.model_type" class="model-selector">
                  <el-radio-button label="zimage">
                    <el-icon><Aim /></el-icon> Z-Image
                  </el-radio-button>
                </el-radio-group>
              </div>

              <!-- Transformer 模型选择 (Finetune) -->
              <div class="param-group">
                <div class="group-label">Transformer 模型 (FINETUNE)</div>
                <el-select v-model="params.transformer_path" placeholder="使用默认模型" clearable filterable style="width: 100%;">
                  <el-option v-for="t in transformerList" :key="t.path" :label="t.name" :value="t.is_default ? null : t.path">
                    <span style="float: left">{{ t.name }}</span>
                    <span v-if="t.size > 0" style="float: right; color: var(--el-text-color-secondary); font-size: 12px">
                      {{ (t.size / 1024 / 1024 / 1024).toFixed(1) }} GB
                    </span>
                  </el-option>
                </el-select>
              </div>

              <!-- Prompt -->
              <div class="param-group">
                <div class="group-label">提示词 (PROMPT)</div>
                <el-input v-model="params.prompt" type="textarea" :rows="4" placeholder="描述你想要生成的图片..." resize="none" class="prompt-input" />
              </div>

              <!-- Negative Prompt -->
              <div class="param-group">
                <div class="group-label">负面提示词 (NEGATIVE)</div>
                <el-input v-model="params.negative_prompt" type="textarea" :rows="2" placeholder="不想要的内容，如：blurry, low quality, watermark..." resize="none" class="prompt-input negative-prompt" />
              </div>

              <!-- Multi-LoRA -->
              <div class="param-group">
                <div class="group-label">
                  LORA 模型
                  <el-tag v-if="params.lora_configs.length > 0" size="small" type="primary" effect="plain" style="margin-left: 8px;">
                    {{ params.lora_configs.length }} 个
                  </el-tag>
                </div>
                
                <!-- LoRA slots -->
                <div v-for="(lora, index) in params.lora_configs" :key="index" class="lora-slot">
                  <div class="lora-slot-header">
                    <el-tag size="small" type="info" effect="plain">#{{ index + 1 }}</el-tag>
                    <el-button type="danger" link size="small" @click="removeLoraSlot(index)">
                      <el-icon><Close /></el-icon>
                    </el-button>
                  </div>
                  <el-select v-model="lora.path" placeholder="选择 LoRA 模型..." clearable filterable style="width: 100%;">
                    <el-option v-for="l in loraList" :key="l.path" :label="l.name" :value="l.path">
                      <span style="float: left">{{ l.name }}</span>
                      <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">
                        {{ (l.size_bytes / 1024 / 1024).toFixed(1) }} MB
                      </span>
                    </el-option>
                  </el-select>
                  <div v-if="lora.path" class="control-row" style="margin-top: 8px;">
                    <span class="label">权重</span>
                    <el-slider v-model="lora.scale" :min="0" :max="2" :step="0.05" :show-tooltip="false" class="slider-flex" />
                    <el-input-number v-model="lora.scale" :min="0" :max="2" :step="0.05" controls-position="right" class="input-fixed" />
                  </div>
                </div>

                <!-- Add LoRA button -->
                <el-button class="add-lora-btn" @click="addLoraSlot" :disabled="params.lora_configs.length >= 5">
                  <el-icon><Plus /></el-icon> 添加 LoRA
                </el-button>

                <!-- Comparison mode (only when at least 1 LoRA) -->
                <div v-if="params.lora_configs.length > 0 && params.lora_configs.some(l => l.path)" class="lora-settings">
                  <div class="control-row">
                    <span class="label">对比</span>
                    <el-switch v-model="params.comparison_mode" active-text="生成原图对比" />
                  </div>
                </div>
              </div>

              <!-- Resolution -->
              <div class="param-group">
                <div class="group-label">分辨率 (RESOLUTION)</div>
                <div class="ratio-grid">
                  <el-button 
                    v-for="ratio in aspectRatios" :key="ratio.label"
                    :type="currentRatio === ratio.label ? 'primary' : 'default'"
                    size="small" class="ratio-btn"
                    @click="setAspectRatio(ratio)"
                  >{{ ratio.label }}</el-button>
                </div>
                <div class="control-row">
                  <span class="label">宽度</span>
                  <el-slider v-model="params.width" :min="256" :max="2048" :step="16" :marks="resolutionMarks" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.width" :min="256" :max="2048" :step="16" controls-position="right" class="input-fixed" />
                </div>
                <div class="control-row">
                  <span class="label">高度</span>
                  <el-slider v-model="params.height" :min="256" :max="2048" :step="16" :marks="resolutionMarks" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.height" :min="256" :max="2048" :step="16" controls-position="right" class="input-fixed" />
                </div>
              </div>

              <!-- Settings -->
              <div class="param-group">
                <div class="group-label">生成设置 (SETTINGS)</div>
                <div class="control-row">
                  <span class="label">步数</span>
                  <el-slider v-model="params.steps" :min="1" :max="50" :step="1" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.steps" :min="1" :max="50" controls-position="right" class="input-fixed" />
                </div>
                <div class="control-row">
                  <span class="label">CFG</span>
                  <el-slider v-model="params.guidance_scale" :min="1" :max="15" :step="0.5" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="params.guidance_scale" :min="1" :max="15" :step="0.5" controls-position="right" class="input-fixed" />
                </div>
                <div class="control-row">
                  <span class="label">Seed</span>
                  <div class="seed-wrapper">
                    <el-input-number v-model="params.seed" :min="-1" controls-position="right" class="seed-input" />
                    <el-button @click="params.seed = -1" icon="Refresh" size="small" class="seed-btn" />
                  </div>
                </div>
              </div>
              
              <el-button 
                type="primary" size="large" class="generate-btn" 
                @click="generateImage" :loading="sse.generating.value" :disabled="sse.generating.value"
              >
                <el-icon><MagicStick /></el-icon>
                {{ sse.generating.value ? '生成中...' : '立即生成' }}
              </el-button>
            </el-form>
          </el-card>
        </div>

        <!-- Preview Section -->
        <div class="preview-section">
          <el-card class="preview-card glass-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span>生成结果</span>
                <div class="header-actions">
                  <span v-if="sse.resultImage.value" class="res-tag">{{ params.width }} x {{ params.height }}</span>
                  <el-button v-if="sse.resultImage.value" type="success" size="small" @click="downloadImage(sse.resultImage.value)">
                    <el-icon><Download /></el-icon> 下载
                  </el-button>
                </div>
              </div>
            </template>
            
            <div class="image-container" 
              @wheel.prevent="mainZoom.handleWheel" @mousedown="mainZoom.startDrag" @mousemove="mainZoom.onDrag"
              @mouseup="mainZoom.stopDrag" @mouseleave="mainZoom.stopDrag" @dblclick="mainZoom.resetZoom"
            >
              <div class="zoom-wrapper" :style="mainZoom.imageStyle.value">
                <!-- 对比模式 -->
                <div v-if="sse.isComparisonResult.value && sse.comparisonImages.value.length === 2" class="comparison-container">
                  <div class="comparison-image-wrapper">
                    <div class="comparison-image-inner">
                      <img :src="sse.comparisonImages.value[0].image" class="comparison-image" draggable="false" />
                      <div class="comparison-label original-label"><el-icon><Picture /></el-icon> 原始模型 (无 LoRA)</div>
                    </div>
                  </div>
                  <div class="comparison-divider"></div>
                  <div class="comparison-image-wrapper">
                    <div class="comparison-image-inner">
                      <img :src="sse.comparisonImages.value[1].image" class="comparison-image" draggable="false" />
                      <div class="comparison-label lora-label">
                        <el-icon><MagicStick /></el-icon>
                        {{ getComparisonLoraLabel(sse.comparisonImages.value[1]) }}
                      </div>
                    </div>
                  </div>
                </div>
                <!-- 普通模式 -->
                <img v-else-if="sse.resultImage.value" :src="sse.resultImage.value" class="generated-image" alt="Generated Image" draggable="false" />
                <div v-else class="placeholder">
                  <el-icon class="placeholder-icon"><Picture /></el-icon>
                  <p>生成的图片将显示在这里</p>
                </div>
              </div>
              
              <!-- 生成中覆盖层 -->
              <Transition name="fade">
                <div v-if="sse.generating.value" class="generation-overlay">
                  <div class="generation-progress-card">
                    <div class="progress-icon"><el-icon class="spinning"><Loading /></el-icon></div>
                    <div class="progress-info">
                      <div class="progress-stage">{{ sse.progressStage.value }}</div>
                      <div class="progress-detail">{{ sse.progressMessage.value }}</div>
                      <el-progress v-if="sse.progressTotal.value > 0" :percentage="Math.round((sse.progressStep.value / sse.progressTotal.value) * 100)" :stroke-width="8" :show-text="true" style="margin-top: 12px; width: 200px;" />
                    </div>
                  </div>
                </div>
              </Transition>
              
              <!-- Zoom Controls -->
              <div class="zoom-controls" v-if="sse.resultImage.value && !sse.generating.value">
                <el-button-group>
                  <el-button size="small" @click="mainZoom.zoomOut()"><el-icon><Minus /></el-icon></el-button>
                  <el-button size="small" @click="mainZoom.resetZoom()">{{ Math.round(mainZoom.scale.value * 100) }}%</el-button>
                  <el-button size="small" @click="mainZoom.zoomIn()"><el-icon><Plus /></el-icon></el-button>
                </el-button-group>
              </div>
            </div>
            
            <div class="result-info" v-if="sse.resultSeed.value !== null">
              <span>Seed: {{ sse.resultSeed.value }}</span>
            </div>
          </el-card>
        </div>
      </div>

      <!-- History Section -->
      <div class="history-section">
        <div class="section-header">
          <h3><el-icon><Clock /></el-icon> 历史记录</h3>
          <el-button link @click="() => fetchHistory()" :loading="loadingHistory">
            <el-icon><Refresh /></el-icon> 刷新
          </el-button>
        </div>
        
        <div class="history-grid" v-loading="loadingHistory">
          <!-- 进行中/已中断任务 -->
          <div v-if="sse.pendingTask.value" class="history-card glass-card pending-task-card" :class="{ 'interrupted': sse.pendingTask.value.interrupted }">
            <div class="history-thumb-wrapper pending-thumb">
              <div class="pending-overlay">
                <template v-if="!sse.pendingTask.value.interrupted">
                  <el-icon class="spinning pending-icon"><Loading /></el-icon>
                  <span class="pending-text">生成中...</span>
                </template>
                <template v-else>
                  <el-icon class="pending-icon interrupted-icon"><WarningFilled /></el-icon>
                  <span class="pending-text">已中断</span>
                </template>
              </div>
            </div>
            <div class="history-info">
              <div class="history-prompt" :title="sse.pendingTask.value.prompt">{{ sse.pendingTask.value.prompt }}</div>
              <div class="history-meta">
                <span>{{ sse.pendingTask.value.width }}x{{ sse.pendingTask.value.height }}</span>
                <el-button v-if="sse.pendingTask.value.interrupted" type="primary" size="small" @click.stop="retryPendingTask">重新生成</el-button>
                <el-button v-if="sse.pendingTask.value.interrupted" type="info" size="small" @click.stop="sse.clearPendingTask()">取消</el-button>
              </div>
            </div>
          </div>
          
          <div v-if="historyList.length === 0 && !sse.pendingTask.value" class="empty-history">暂无历史记录</div>
          <div v-for="item in historyList" :key="getHistoryKey(item)" class="history-card glass-card" :class="{ 'comparison-history': isComparisonEntry(item) }" @click="openLightbox(item)">
            <div class="history-thumb-wrapper">
              <el-image :src="getHistoryThumb(item)" fit="cover" class="history-thumb" lazy />
              <div class="history-overlay"><el-icon class="overlay-icon"><ZoomIn /></el-icon></div>
              <div v-if="isComparisonEntry(item)" class="comparison-badge">VS 对比</div>
            </div>
            <div class="history-info">
              <div class="history-prompt" :title="getHistoryMeta(item).prompt">{{ getHistoryMeta(item).prompt }}</div>
              <div class="history-meta">
                <el-tag size="small" type="primary" effect="plain">Z-Image</el-tag>
                <span>{{ getHistoryMeta(item).width }}x{{ getHistoryMeta(item).height }}</span>
                <span>Seed: {{ getHistoryMeta(item).seed }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Pagination -->
        <div class="history-pagination" v-if="historyTotal > historyPageSize">
          <el-pagination
            v-model:current-page="historyPage"
            :page-size="historyPageSize"
            :total="historyTotal"
            layout="prev, pager, next"
            @current-change="onPageChange"
          />
        </div>
      </div>
    </div>

    <!-- Lightbox Overlay -->
    <div v-if="lightboxVisible" class="lightbox-overlay" @click.self="closeLightbox">
      <div class="lightbox-content">
        <div class="lightbox-header">
          <div class="lightbox-title">历史记录详情</div>
          <div class="lightbox-actions">
            <el-button type="primary" @click="restoreParams((lightboxItem.metadata || lightboxItem))">应用此参数</el-button>
            <el-button type="success" @click="downloadImage(lightboxItem.url || `/api/generation/image/${lightboxItem.timestamp}`)"><el-icon><Download /></el-icon> 下载</el-button>
            <el-button type="danger" @click="deleteHistoryItem(lightboxItem, true)"><el-icon><Delete /></el-icon> 删除</el-button>
            <el-button circle @click="closeLightbox"><el-icon><Close /></el-icon></el-button>
          </div>
        </div>
        
        <div class="lightbox-image-container"
          @wheel.prevent="lbZoom.handleWheel" @mousedown="lbZoom.startDrag" @mousemove="lbZoom.onDrag"
          @mouseup="lbZoom.stopDrag" @mouseleave="lbZoom.stopDrag" @dblclick="lbZoom.resetZoom"
        >
          <div class="zoom-wrapper" :style="lbZoom.imageStyle.value">
            <!-- Comparison lightbox -->
            <div v-if="isComparisonEntry(lightboxItem)" class="comparison-container">
              <div class="comparison-image-wrapper">
                <div class="comparison-image-inner">
                  <img :src="`/api/generation/image/${lightboxItem.comparison_images[0].image_path.split('/').pop().replace('.png','')}`" class="comparison-image" draggable="false" />
                  <div class="comparison-label original-label"><el-icon><Picture /></el-icon> 原始模型 (无 LoRA)</div>
                </div>
              </div>
              <div class="comparison-divider"></div>
              <div class="comparison-image-wrapper">
                <div class="comparison-image-inner">
                  <img :src="`/api/generation/image/${lightboxItem.comparison_images[1].image_path.split('/').pop().replace('.png','')}`" class="comparison-image" draggable="false" />
                  <div class="comparison-label lora-label">
                    <el-icon><MagicStick /></el-icon>
                    {{ getLightboxLoraLabel(lightboxItem) }}
                  </div>
                </div>
              </div>
            </div>
            <!-- Normal lightbox -->
            <img v-else :src="lightboxItem.url || `/api/generation/image/${lightboxItem.timestamp}`" class="lightbox-image" draggable="false" />
          </div>
          <div class="zoom-controls">
            <el-button-group>
              <el-button size="small" @click="lbZoom.zoomOut()"><el-icon><Minus /></el-icon></el-button>
              <el-button size="small" @click="lbZoom.resetZoom()">{{ Math.round(lbZoom.scale.value * 100) }}%</el-button>
              <el-button size="small" @click="lbZoom.zoomIn()"><el-icon><Plus /></el-icon></el-button>
            </el-button-group>
          </div>
        </div>
        
        <div class="lightbox-info">
          <div class="info-item">
            <span class="label">Prompt:</span>
            <span class="text">{{ getHistoryMeta(lightboxItem).prompt }}</span>
          </div>
          <div class="info-row">
            <div class="info-item"><span class="label">Size:</span><span class="text">{{ getHistoryMeta(lightboxItem).width }} x {{ getHistoryMeta(lightboxItem).height }}</span></div>
            <div class="info-item"><span class="label">Steps:</span><span class="text">{{ getHistoryMeta(lightboxItem).steps }}</span></div>
            <div class="info-item"><span class="label">CFG:</span><span class="text">{{ getHistoryMeta(lightboxItem).guidance_scale }}</span></div>
            <div class="info-item"><span class="label">Seed:</span><span class="text">{{ getHistoryMeta(lightboxItem).seed }}</span></div>
            <div class="info-item"><span class="label">Model:</span><el-tag size="small" type="primary">Z-Image</el-tag></div>
            <div class="info-item" v-if="getLoraConfigsFromMeta(getHistoryMeta(lightboxItem)).length > 0">
              <span class="label">LoRA:</span>
              <span class="text">
                <span v-for="(lc, i) in getLoraConfigsFromMeta(getHistoryMeta(lightboxItem))" :key="i">
                  {{ getLoraFileName(lc.path) }} ({{ lc.scale }}){{ i < getLoraConfigsFromMeta(getHistoryMeta(lightboxItem)).length - 1 ? ' + ' : '' }}
                </span>
              </span>
            </div>
            <div class="info-item" v-if="isComparisonEntry(lightboxItem)">
              <el-tag size="small" type="warning">VS 对比模式</el-tag>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { MagicStick, Download, Picture, Refresh, Clock, Plus, Minus, ZoomIn, Close, Delete, Operation, Loading, WarningFilled, Aim } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useImageZoom } from '@/composables/useImageZoom'
import { useGenerationSSE } from '@/composables/useGenerationSSE'

// Types
interface LoraSlot {
  path: string | null
  scale: number
}

// Composables
const sse = useGenerationSSE()
const mainZoom = useImageZoom()
const lbZoom = useImageZoom()

// Params (with localStorage restore)
const defaultParams = {
  prompt: "A futuristic city with flying cars, cyberpunk style, highly detailed, 8k",
  negative_prompt: "",
  steps: 9, guidance_scale: 1.0, seed: -1,
  width: 1024, height: 1024,
  lora_configs: [] as LoraSlot[],
  comparison_mode: false,
  model_type: "zimage",
  transformer_path: null as string | null,
}

const savedParams = sse.loadSavedParams()
const params = ref(migrateParams(savedParams ? { ...defaultParams, ...savedParams } : { ...defaultParams }))

/** Migrate old single-lora params to multi-lora format */
function migrateParams(p: any) {
  if (!p.lora_configs) {
    p.lora_configs = []
  }
  // Migrate legacy lora_path → lora_configs[0]
  if (p.lora_path && p.lora_configs.length === 0) {
    p.lora_configs.push({ path: p.lora_path, scale: p.lora_scale || 1.0 })
  }
  // Clean up legacy fields
  delete p.lora_path
  delete p.lora_scale
  return p
}

// Auto-save params
watch(params, (v) => sse.saveParams(v), { deep: true })

// Multi-LoRA slot management
const addLoraSlot = () => {
  if (params.value.lora_configs.length < 5) {
    params.value.lora_configs.push({ path: null, scale: 1.0 })
  }
}
const removeLoraSlot = (index: number) => {
  params.value.lora_configs.splice(index, 1)
}

// History
const loadingHistory = ref(false)
const historyList = ref<any[]>([])
const historyPage = ref(1)
const historyPageSize = 50
const historyTotal = ref(0)

// LoRA / Transformer lists
const loraList = ref<any[]>([])
const transformerList = ref<any[]>([])

// Lightbox
const lightboxVisible = ref(false)
const lightboxItem = ref<any>(null)

// Aspect Ratio
const aspectRatios = [
  { label: '1:1', w: 1024, h: 1024 },
  { label: '4:3', w: 1024, h: 768 },
  { label: '3:4', w: 768, h: 1024 },
  { label: '16:9', w: 1024, h: 576 },
  { label: '9:16', w: 576, h: 1024 }
]
const currentRatio = computed(() => {
  const match = aspectRatios.find(r => r.w === params.value.width && r.h === params.value.height)
  return match ? match.label : 'Custom'
})
const resolutionMarks = { 512: '', 1024: '', 1536: '', 2048: '' }
const setAspectRatio = (ratio: any) => { params.value.width = ratio.w; params.value.height = ratio.h }

// Helpers
const getLoraFileName = (path: string | null) => {
  if (!path) return 'Unknown'
  return path.split(/[\/\\]/).pop()?.replace('.safetensors', '') || 'Unknown'
}

/** Get lora_configs from a history meta (supports both new and legacy format) */
const getLoraConfigsFromMeta = (meta: any): { path: string, scale: number }[] => {
  if (meta.lora_configs && Array.isArray(meta.lora_configs) && meta.lora_configs.length > 0) {
    return meta.lora_configs
  }
  if (meta.lora_path) {
    return [{ path: meta.lora_path, scale: meta.lora_scale || 1.0 }]
  }
  return []
}

/** Build comparison label for multi-LoRA */
const getComparisonLoraLabel = (imgData: any) => {
  const configs = imgData.lora_configs || []
  if (configs.length > 1) {
    return configs.map((c: any) => `${getLoraFileName(c.path)}@${c.scale?.toFixed(2)}`).join(' + ')
  }
  if (configs.length === 1) {
    return `LoRA: ${getLoraFileName(configs[0].path)} (权重: ${configs[0].scale?.toFixed(2)})`
  }
  // Legacy fallback
  if (imgData.lora_path) {
    return `LoRA: ${getLoraFileName(imgData.lora_path)} (权重: ${imgData.lora_scale?.toFixed(2) || '1.00'})`
  }
  return 'LoRA'
}

const getLightboxLoraLabel = (item: any) => {
  const lcs = getLoraConfigsFromMeta(item)
  if (lcs.length > 1) {
    return lcs.map(c => `${getLoraFileName(c.path)}@${c.scale}`).join(' + ')
  }
  if (lcs.length === 1) {
    return `LoRA: ${getLoraFileName(lcs[0].path)} (权重: ${lcs[0].scale})`
  }
  return 'LoRA'
}

// History helpers for comparison entries
const isComparisonEntry = (item: any) => {
  const meta = item.metadata || item
  return !!meta.comparison_mode
}
const getHistoryMeta = (item: any) => item.metadata || item
const getHistoryKey = (item: any) => {
  const meta = getHistoryMeta(item)
  return meta.timestamp || meta.image_path || Math.random().toString()
}
const getHistoryThumb = (item: any) => {
  const meta = getHistoryMeta(item)
  if (meta.comparison_mode && meta.comparison_images?.length >= 2) {
    // Use LoRA image (second) as thumbnail
    const p = meta.comparison_images[1].image_path
    const ts = p.split('/').pop()?.replace('.png', '') || meta.timestamp
    return `/api/generation/image/${ts}`
  }
  return item.thumbnail || item.url || `/api/generation/image/${meta.timestamp}`
}

// Generate (map frontend field names → backend DTO)
const generateImage = () => {
  const p = params.value
  // Filter out empty LoRA slots
  const validLoras = p.lora_configs.filter((l: LoraSlot) => l.path)
  const payload: any = {
    prompt: p.prompt,
    negative_prompt: p.negative_prompt,
    width: p.width,
    height: p.height,
    num_inference_steps: p.steps,
    guidance_scale: p.guidance_scale,
    seed: p.seed,
    transformer_path: p.transformer_path,
    comparison_mode: p.comparison_mode,
  }
  // Send as lora_configs (multi-LoRA)
  if (validLoras.length > 0) {
    payload.lora_configs = validLoras.map((l: LoraSlot) => ({ path: l.path, scale: l.scale }))
  }
  sse.generate(payload, () => { fetchHistory(1, true); mainZoom.resetZoom() })
}

// Retry pending
const retryPendingTask = () => {
  if (sse.pendingTask.value) {
    const t = sse.pendingTask.value
    params.value.prompt = t.prompt; params.value.width = t.width; params.value.height = t.height
    params.value.steps = t.steps; params.value.seed = t.seed
    sse.clearPendingTask()
    generateImage()
  }
}

// History
const fetchHistory = async (page?: number, silent = false) => {
  if (page !== undefined) historyPage.value = page
  if (!silent) loadingHistory.value = true
  try {
    const res = await axios.get('/api/generation/history', { params: { page: historyPage.value, page_size: historyPageSize } })
    const data = res.data?.data || {}
    historyList.value = Array.isArray(data.items) ? data.items : []
    historyTotal.value = data.total || 0
  } catch { /* history not available yet */ }
  finally { loadingHistory.value = false }
}

const onPageChange = (page: number) => {
  fetchHistory(page)
}

const deleteHistoryItem = async (item: any, fromLightbox = false) => {
  try {
    await ElMessageBox.confirm('确定要删除这张图片吗？此操作不可恢复。', '删除确认', {
      confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning', confirmButtonClass: 'el-button--danger'
    })
    const meta = getHistoryMeta(item)
    const res = await axios.post('/api/history/delete', { timestamps: [meta.timestamp] })
    if (res.data.success) { ElMessage.success('删除成功'); if (fromLightbox) closeLightbox(); fetchHistory() }
    else ElMessage.error('删除失败')
  } catch (e) { if (e !== 'cancel') ElMessage.error('删除请求失败') }
}

// LoRA / Transformer fetching
const fetchLoras = async () => {
  try { const res = await axios.get('/api/loras'); loraList.value = res.data.data || res.data.loras || [] } catch {}
}
const fetchTransformers = async () => {
  try { const res = await axios.get('/api/transformers'); transformerList.value = res.data.transformers || [] } catch {}
}

// Lightbox
const openLightbox = (item: any) => { lightboxItem.value = item; lightboxVisible.value = true; lbZoom.resetZoom(); document.body.style.overflow = 'hidden' }
const closeLightbox = () => { lightboxVisible.value = false; lightboxItem.value = null; document.body.style.overflow = '' }

const restoreParams = (metadata: any) => {
  const meta = metadata.metadata || metadata
  params.value.prompt = meta.prompt; params.value.steps = meta.steps
  params.value.guidance_scale = meta.guidance_scale; params.value.seed = meta.seed
  params.value.width = meta.width; params.value.height = meta.height
  params.value.model_type = meta.model_type || "zimage"
  // Restore multi-LoRA configs
  const loraConfigs = getLoraConfigsFromMeta(meta)
  if (loraConfigs.length > 0) {
    params.value.lora_configs = loraConfigs.map(c => ({ path: c.path, scale: c.scale || 1.0 }))
    params.value.comparison_mode = meta.comparison_mode || false
  } else {
    params.value.lora_configs = []
    params.value.comparison_mode = false
  }
  ElMessage.success('参数已应用'); closeLightbox()
}

// Download
const downloadImage = async (url: string | null) => {
  if (!url) return
  if (sse.isComparisonResult.value && sse.comparisonImages.value.length === 2) {
    try { await downloadComparisonImage(); return } catch {}
  }
  const link = document.createElement('a'); link.href = url
  link.download = `generated_${Date.now()}.png`
  document.body.appendChild(link); link.click(); document.body.removeChild(link)
}

const downloadComparisonImage = async () => {
  const imgs = sse.comparisonImages.value
  const img1 = await loadImg(imgs[0].image); const img2 = await loadImg(imgs[1].image)
  const hdr = 40, gap = 10
  const canvas = document.createElement('canvas')
  canvas.width = img1.width + img2.width + gap; canvas.height = Math.max(img1.height, img2.height) + hdr
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#1e1e1e'; ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.font = 'bold 16px sans-serif'
  ctx.fillStyle = 'rgba(48, 48, 48, 0.9)'; ctx.fillRect(0, 0, img1.width, hdr)
  ctx.fillStyle = '#ffffff'; ctx.fillText('📷 原始模型 (无 LoRA)', 12, 26)
  const loraLabel = getComparisonLoraLabel(imgs[1])
  const grad = ctx.createLinearGradient(img1.width + gap, 0, canvas.width, 0)
  grad.addColorStop(0, 'rgba(64, 158, 255, 0.9)'); grad.addColorStop(1, 'rgba(103, 194, 58, 0.9)')
  ctx.fillStyle = grad; ctx.fillRect(img1.width + gap, 0, img2.width, hdr)
  ctx.fillStyle = '#ffffff'; ctx.fillText(`✨ ${loraLabel}`, img1.width + gap + 12, 26)
  ctx.drawImage(img1, 0, hdr); ctx.drawImage(img2, img1.width + gap, hdr)
  const link = document.createElement('a'); link.href = canvas.toDataURL('image/png')
  link.download = `comparison_${Date.now()}.png`
  document.body.appendChild(link); link.click(); document.body.removeChild(link)
}

const loadImg = (src: string): Promise<HTMLImageElement> => new Promise((resolve, reject) => {
  const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = src
})

onMounted(() => { fetchHistory(); fetchLoras(); fetchTransformers(); sse.checkPendingTask(() => fetchHistory()) })
</script>

<style scoped>
.generation-container { height: 100%; overflow-y: auto; }
.main-content { max-width: 1400px; margin: 0 auto; }
.generation-grid { display: grid; grid-template-columns: 400px 1fr; gap: 24px; margin-bottom: 40px; min-height: 600px; }
.params-card { height: fit-content; position: sticky; top: 0; }
.preview-card { height: 100%; display: flex; flex-direction: column; min-height: 600px; }
.preview-card :deep(.el-card__body) { flex: 1; display: flex; flex-direction: column; padding: 0; overflow: hidden; background: #000; position: relative; }
.card-header { display: flex; justify-content: space-between; align-items: center; font-weight: bold; font-size: 15px; }
.card-header span { display: flex; align-items: center; gap: 8px; }
.params-form { display: flex; flex-direction: column; gap: 20px; }
.param-group { background: var(--el-fill-color-lighter); padding: 16px; border-radius: 8px; border: 1px solid var(--el-border-color-lighter); }
.group-label { font-size: 11px; font-weight: 700; color: var(--el-text-color-secondary); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; }
.prompt-input :deep(.el-textarea__inner) { background: var(--el-bg-color); border-color: var(--el-border-color-light); font-family: inherit; }
.model-selector { width: 100%; display: flex; }
.model-selector :deep(.el-radio-button) { flex: 1; }
.model-selector :deep(.el-radio-button__inner) { width: 100%; display: flex; align-items: center; justify-content: center; gap: 6px; }

/* Multi-LoRA slots */
.lora-slot { background: var(--el-bg-color); border: 1px solid var(--el-border-color-light); border-radius: 6px; padding: 10px; margin-bottom: 8px; transition: border-color 0.2s; }
.lora-slot:hover { border-color: var(--el-color-primary-light-5); }
.lora-slot-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.add-lora-btn { width: 100%; border-style: dashed; margin-top: 4px; }
.add-lora-btn:hover { border-color: var(--el-color-primary); color: var(--el-color-primary); }

.ratio-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-bottom: 16px; }
.ratio-btn { width: 100%; padding: 6px 0; font-size: 11px; }
.control-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.control-row:last-child { margin-bottom: 0; }
.control-row .label { font-size: 12px; color: var(--el-text-color-regular); width: 32px; flex-shrink: 0; }
.slider-flex { flex: 1; margin-right: 8px; }
.input-fixed { width: 80px !important; }
.seed-wrapper { flex: 1; display: flex; gap: 8px; }
.seed-input { flex: 1; }
.seed-btn { width: 32px; padding: 0; }
.lora-settings { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--el-border-color-light); }
.generate-btn { width: 100%; font-weight: bold; letter-spacing: 1px; height: 48px; font-size: 16px; margin-top: 8px; box-shadow: 0 4px 12px rgba(var(--el-color-primary-rgb), 0.3); }

/* Image Preview & Zoom */
.image-container { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; background-image: linear-gradient(45deg, #1a1a1a 25%, transparent 25%), linear-gradient(-45deg, #1a1a1a 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #1a1a1a 75%), linear-gradient(-45deg, transparent 75%, #1a1a1a 75%); background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0px; background-color: #111; cursor: grab; }
.image-container:active { cursor: grabbing; }
.zoom-wrapper { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; }
.generated-image { max-width: 100%; max-height: 100%; object-fit: contain; box-shadow: 0 0 30px rgba(0,0,0,0.5); pointer-events: none; }

.comparison-container { display: flex; align-items: center; justify-content: center; gap: 4px; width: 100%; height: 100%; container-type: size; }
.comparison-image-wrapper { position: relative; }
.comparison-image-inner { position: relative; line-height: 0; }
.comparison-image-inner img { display: block; max-height: 100cqh; max-width: calc(50cqi - 10px); object-fit: contain; box-shadow: 0 0 20px rgba(0,0,0,0.5); border-radius: 4px; }
.comparison-label { position: absolute; top: 8px; left: 8px; background: rgba(0, 0, 0, 0.75); color: #fff; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; z-index: 10; backdrop-filter: blur(4px); display: flex; align-items: center; gap: 6px; white-space: nowrap; max-width: calc(100% - 16px); overflow: hidden; text-overflow: ellipsis; }
.comparison-label.original-label { background: rgba(48, 48, 48, 0.85); }
.comparison-label.lora-label { background: linear-gradient(135deg, rgba(64, 158, 255, 0.9), rgba(103, 194, 58, 0.9)); }
.comparison-divider { width: 2px; align-self: stretch; background: linear-gradient(to bottom, transparent, rgba(255,255,255,0.4), transparent); flex-shrink: 0; }
.zoom-controls { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.6); border-radius: 4px; padding: 4px; backdrop-filter: blur(4px); }

/* Progress Overlay */
.generation-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.85); display: flex; align-items: center; justify-content: center; z-index: 100; backdrop-filter: blur(8px); }
.generation-progress-card { background: linear-gradient(135deg, rgba(30, 30, 40, 0.95), rgba(20, 20, 30, 0.98)); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 32px 48px; min-width: 320px; max-width: 400px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(var(--el-color-primary-rgb), 0.15); text-align: center; }
.progress-icon { font-size: 48px; color: var(--el-color-primary); margin-bottom: 20px; }
.progress-icon .spinning { animation: spin 1.5s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.progress-stage { font-size: 20px; font-weight: bold; color: #fff; margin-bottom: 8px; }
.progress-detail { font-size: 14px; color: rgba(255, 255, 255, 0.7); font-family: 'SF Mono', 'Fira Code', monospace; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
.placeholder { display: flex; flex-direction: column; align-items: center; color: var(--el-text-color-secondary); opacity: 0.5; }
.placeholder-icon { font-size: 80px; margin-bottom: 16px; }
.result-info { padding: 8px 16px; background: var(--el-bg-color-overlay); border-top: 1px solid var(--el-border-color-light); text-align: right; font-family: monospace; color: var(--el-text-color-secondary); font-size: 12px; z-index: 10; }

/* History */
.history-section { margin-top: 40px; }
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.section-header h3 { margin: 0; display: flex; align-items: center; gap: 8px; font-size: 18px; }
.history-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.history-pagination { display: flex; justify-content: center; margin-top: 16px; padding-bottom: 8px; }
.history-card { cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; overflow: hidden; border-radius: 8px; border: 1px solid var(--el-border-color-light); background: var(--el-bg-color); }
.history-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
.history-thumb-wrapper { position: relative; aspect-ratio: 1; overflow: hidden; }
.history-thumb { width: 100%; height: 100%; display: block; }
.history-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity 0.2s; color: white; font-size: 32px; }
.history-card:hover .history-overlay { opacity: 1; }
.history-info { padding: 12px; }
.history-prompt { font-size: 12px; color: var(--el-text-color-primary); display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.4; height: 34px; margin-bottom: 8px; }
.history-meta { display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--el-text-color-secondary); font-family: monospace; gap: 8px; flex-wrap: wrap; }

/* Pending */
.pending-task-card { border: 2px solid var(--el-color-primary); animation: pulse-border 2s infinite; }
.pending-task-card.interrupted { border-color: var(--el-color-warning); animation: none; }
@keyframes pulse-border { 0%, 100% { border-color: var(--el-color-primary); } 50% { border-color: var(--el-color-primary-light-3); } }
.pending-thumb { background: linear-gradient(135deg, var(--el-bg-color-page), var(--el-bg-color)); }
.pending-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; background: rgba(0, 0, 0, 0.3); }
.pending-icon { font-size: 48px; color: var(--el-color-primary); }
.interrupted-icon { color: var(--el-color-warning); }
.pending-text { font-size: 14px; color: var(--el-text-color-primary); font-weight: 500; }
.empty-history { grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--el-text-color-secondary); background: var(--el-fill-color-light); border-radius: 8px; }

/* Comparison History Badge */
.comparison-history { border: 1px solid rgba(230, 162, 60, 0.4); }
.comparison-badge { position: absolute; top: 8px; right: 8px; background: linear-gradient(135deg, rgba(64, 158, 255, 0.9), rgba(103, 194, 58, 0.9)); color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; z-index: 10; backdrop-filter: blur(4px); }

/* Lightbox */
.lightbox-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.9); z-index: 2000; display: flex; align-items: center; justify-content: center; padding: 40px; }
.lightbox-content { width: 100%; height: 100%; max-width: 1400px; display: flex; flex-direction: column; background: #111; border-radius: 8px; overflow: hidden; box-shadow: 0 0 50px rgba(0,0,0,0.5); }
.lightbox-header { padding: 16px 24px; background: #1a1a1a; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; }
.lightbox-title { font-size: 18px; font-weight: bold; color: #fff; }
.lightbox-actions { display: flex; gap: 12px; }
.lightbox-image-container { flex: 1; position: relative; overflow: hidden; background-image: linear-gradient(45deg, #1a1a1a 25%, transparent 25%), linear-gradient(-45deg, #1a1a1a 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #1a1a1a 75%), linear-gradient(-45deg, transparent 75%, #1a1a1a 75%); background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0px; background-color: #000; cursor: grab; }
.lightbox-image-container:active { cursor: grabbing; }
.lightbox-image { max-width: 100%; max-height: 100%; object-fit: contain; pointer-events: none; }
.lightbox-info { padding: 16px 24px; background: #1a1a1a; border-top: 1px solid #333; color: #ccc; max-height: 150px; overflow-y: auto; flex-shrink: 0; }
.info-row { display: flex; gap: 40px; margin-top: 12px; flex-wrap: wrap; }
.info-item { display: flex; gap: 8px; }
.info-item .label { color: #888; font-weight: bold; }
.info-item .text { color: #eee; font-family: monospace; }

@media (max-width: 1024px) {
  .generation-grid { grid-template-columns: 1fr; }
  .params-card { position: static; }
}
</style>
