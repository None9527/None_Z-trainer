<template>
  <el-dialog
    v-model="visible"
    title="生成缓存"
    width="550px"
    @close="handleClose"
  >
    <el-form label-width="auto">
      <el-form-item label="模型类型">
        <el-select v-model="config.modelType" placeholder="请选择模型类型">
          <el-option label="Z-Image" value="zimage" />
        </el-select>
        <div class="cache-model-hint">
          <span>将使用 Z-Image 缓存脚本生成 _zi.safetensors 格式的缓存文件</span>
        </div>
      </el-form-item>
      
      <!-- Model Path (auto-detected from system config) -->
      <el-form-item label="模型路径">
        <div v-if="loadingModel" class="model-loading">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>正在检测模型...</span>
        </div>
        <div v-else-if="modelStatus === 'valid'" class="model-valid">
          <el-icon color="#67c23a"><CircleCheckFilled /></el-icon>
          <span>{{ modelPath }}</span>
        </div>
        <div v-else class="model-invalid">
          <el-icon color="#f56c6c"><WarningFilled /></el-icon>
          <span>模型未就绪: {{ modelPath || '未配置' }}</span>
        </div>
      </el-form-item>

      <!-- GPU Info (auto-detected) -->
      <el-form-item label="GPU 配置">
        <div v-if="loadingGpu" class="model-loading">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>正在检测 GPU...</span>
        </div>
        <div v-else-if="gpuInfo.num_gpus > 0" class="gpu-info-panel">
          <div class="gpu-summary">
            <el-icon color="#67c23a"><CircleCheckFilled /></el-icon>
            <span>{{ gpuInfo.num_gpus }} 张 GPU · {{ gpuInfo.total_vram_gb }}GB 显存</span>
            <el-tag v-if="gpuInfo.num_gpus > 1" type="success" size="small" effect="plain">多卡并行</el-tag>
          </div>
          <div class="gpu-cards" v-if="gpuInfo.gpus.length > 0">
            <div class="gpu-card" v-for="gpu in gpuInfo.gpus" :key="gpu.index">
              <span class="gpu-id">GPU {{ gpu.index }}</span>
              <span class="gpu-name">{{ gpu.name }}</span>
              <span class="gpu-vram">{{ (gpu.vram_free_mb / 1024).toFixed(1) }} / {{ (gpu.vram_total_mb / 1024).toFixed(1) }}GB 可用</span>
            </div>
          </div>
        </div>
        <div v-else class="model-invalid">
          <el-icon color="#f56c6c"><WarningFilled /></el-icon>
          <span>未检测到 GPU</span>
        </div>
      </el-form-item>
      
      <el-form-item label="训练模式">
        <el-select v-model="config.trainingMode" placeholder="请选择训练模式">
          <el-option label="Text2Img (标准)" value="text2img" />
          <el-option label="ControlNet" value="controlnet" />
          <el-option label="Img2Img" value="img2img" />
          <el-option label="Inpaint" value="inpaint" />
          <el-option label="Omni (多图)" value="omni" />
        </el-select>
        <div class="cache-model-hint">
          <span v-if="config.trainingMode === 'text2img'">标准 LoRA 训练，只需 latent 和 text 缓存</span>
          <span v-else-if="config.trainingMode === 'controlnet'">ControlNet 训练，需要额外的条件图缓存</span>
          <span v-else-if="config.trainingMode === 'img2img'">Img2Img 训练，需要源图和目标图 latent 配对</span>
          <span v-else-if="config.trainingMode === 'inpaint'">Inpaint 训练，需要 mask 图和目标图 latent 配对</span>
          <span v-else-if="config.trainingMode === 'omni'">多图条件训练，需要条件图 VAE latent + SigLIP 特征</span>
        </div>
      </el-form-item>
      
      <el-form-item label="选择缓存类型">
        <el-checkbox-group v-model="config.options">
          <el-checkbox label="latent">
            Latent 缓存
            <span class="cache-path-hint" v-if="modelPath">(vae)</span>
          </el-checkbox>
          <el-checkbox label="text">
            Text 缓存
            <span class="cache-path-hint" v-if="modelPath">(text_encoder)</span>
          </el-checkbox>
          
          <!-- ControlNet 模式额外选项 -->
          <el-checkbox label="control" v-if="config.trainingMode === 'controlnet'">
            条件图缓存 (边缘/深度/姿态)
          </el-checkbox>
          
          <!-- Omni 模式额外选项 -->
          <el-checkbox label="siglip" v-if="config.trainingMode === 'omni'">
            SigLIP 特征缓存
          </el-checkbox>
          
          <!-- DINOv3 感知缓存 (所有模式可用) -->
          <el-checkbox label="dino">
            DINOv3 感知缓存
            <el-tooltip content="预缓存 DINOv3 语义特征，训练时可使用 DINOv3 感知 Loss" placement="top">
              <el-icon class="help-icon" style="margin-left: 4px; font-size: 12px; cursor: help;"><QuestionFilled /></el-icon>
            </el-tooltip>
          </el-checkbox>
        </el-checkbox-group>
      </el-form-item>
      
      <!-- DINOv3 模型状态 -->
      <el-form-item label="DINOv3 模型" v-if="config.options.includes('dino')">
        <div v-if="dinoReady" class="model-valid">
          <el-icon color="#67c23a"><CircleCheckFilled /></el-icon>
          <span>{{ dinoPath }}</span>
        </div>
        <div v-else class="model-invalid">
          <el-icon color="#f56c6c"><WarningFilled /></el-icon>
          <span>未配置 DINO_MODEL_PATH (.env)</span>
        </div>
      </el-form-item>
      
      <!-- ControlNet 条件图目录 -->
      <el-form-item label="条件图目录" v-if="config.trainingMode === 'controlnet' && config.options.includes('control')">
        <el-input v-model="config.controlDir" placeholder="条件图像目录路径" />
      </el-form-item>
      
      <!-- Img2Img 源图目录 -->
      <el-form-item label="源图目录" v-if="config.trainingMode === 'img2img'">
        <el-input v-model="config.sourceDir" placeholder="源图像目录路径" />
      </el-form-item>
      
      <!-- Inpaint mask 目录 -->
      <el-form-item label="Mask 目录" v-if="config.trainingMode === 'inpaint'">
        <el-input v-model="config.maskDir" placeholder="掩码图目录路径" />
      </el-form-item>
      
      <!-- Omni 条件图目录 -->
      <el-form-item label="条件图目录" v-if="config.trainingMode === 'omni'">
        <el-input v-model="config.conditionDirs" placeholder="条件图目录 (多个用逗号分隔)" />
        <div class="cache-model-hint">
          <span>每个条件图目录对应一张条件图，文件名必须与目标图一致</span>
        </div>
      </el-form-item>
      <el-form-item label="条件图数量" v-if="config.trainingMode === 'omni'">
        <el-input-number v-model="config.numConditionImages" :min="1" :max="10" />
      </el-form-item>
    </el-form>
    
    <div class="cache-hint" v-if="modelStatus === 'valid'">
      <el-icon><InfoFilled /></el-icon>
      <span>
        缓存文件将保存在数据集目录中
        <template v-if="gpuInfo.num_gpus > 1">
          · 将使用 {{ gpuInfo.num_gpus }} 张 GPU 并行加速
        </template>
      </span>
    </div>
    
    <template #footer>
      <el-button @click="handleClose">取消</el-button>
      <el-button type="primary" @click="handleConfirm" :loading="generating" :disabled="!canGenerate">
        开始生成
      </el-button>
    </template>
  </el-dialog>
</template>


<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { InfoFilled, WarningFilled, CircleCheckFilled, Loading, QuestionFilled } from '@element-plus/icons-vue'
import axios from 'axios'

// Props
const props = defineProps<{
  modelValue: boolean
  generating?: boolean
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm', config: { 
    modelType: string
    trainingMode: string
    options: string[]
    modelPath: string
    dinoModelPath?: string
    controlDir?: string
    sourceDir?: string
    maskDir?: string
    conditionDirs?: string
    numConditionImages?: number
  }): void
}>()

// 本地状态
const visible = ref(props.modelValue)
const loadingModel = ref(false)
const loadingGpu = ref(false)
const modelPath = ref('')
const modelStatus = ref('')
const gpuInfo = ref<{ num_gpus: number; total_vram_gb: number; gpus: any[] }>({
  num_gpus: 0,
  total_vram_gb: 0,
  gpus: [],
})

const config = ref({
  modelType: 'zimage',
  trainingMode: 'text2img',
  options: ['latent', 'text'] as string[],
  controlDir: '',
  sourceDir: '',
  maskDir: '',
  conditionDirs: '',
  numConditionImages: 1,
})

// 同步 visible 与 modelValue
watch(() => props.modelValue, (val) => {
  visible.value = val
  if (val) {
    fetchModelStatus()
    fetchGpuInfo()
    fetchDinoStatus()
  }
})

watch(visible, (val) => {
  emit('update:modelValue', val)
})

// 从系统 API 获取模型路径（与首页一致）
async function fetchModelStatus() {
  loadingModel.value = true
  try {
    const res = await axios.get('/api/system/model/status')
    const data = res.data?.data ?? res.data
    modelPath.value = data.path || ''
    modelStatus.value = data.status || ''
  } catch {
    modelStatus.value = 'error'
    modelPath.value = ''
  } finally {
    loadingModel.value = false
  }
}

// 获取 GPU 信息
async function fetchGpuInfo() {
  loadingGpu.value = true
  try {
    const res = await axios.get('/api/cache/gpu-info')
    const data = res.data?.data ?? res.data
    gpuInfo.value = data
  } catch {
    gpuInfo.value = { num_gpus: 0, total_vram_gb: 0, gpus: [] }
  } finally {
    loadingGpu.value = false
  }
}

// 是否可以生成
const canGenerate = computed(() => {
  return config.value.options.length > 0 && modelStatus.value === 'valid'
})

// DINOv3 模型状态
const dinoReady = ref(false)
const dinoPath = ref('')

async function fetchDinoStatus() {
  try {
    const res = await axios.get('/api/system/dino/status')
    if (res.data.success && res.data.data) {
      const d = res.data.data
      dinoReady.value = d.status === 'ready'
      dinoPath.value = d.path || ''
    }
  } catch { }
}

// 确认生成
function handleConfirm() {
  emit('confirm', { ...config.value, modelPath: modelPath.value, dinoModelPath: dinoPath.value })
}

// 关闭对话框
function handleClose() {
  visible.value = false
}
</script>

<style lang="scss" scoped>
.cache-model-hint {
  margin-top: var(--space-xs);
  font-size: 12px;
  color: var(--text-muted);
}

.cache-path-hint {
  font-size: 12px;
  color: var(--text-muted);
}

.model-loading, .model-valid, .model-invalid {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-size: 13px;
}

.model-valid { color: var(--color-success, #67c23a); }
.model-invalid { color: var(--color-danger, #f56c6c); }

.gpu-info-panel {
  width: 100%;
}

.gpu-summary {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-size: 13px;
  color: var(--color-success, #67c23a);
}

.gpu-cards {
  margin-top: var(--space-sm);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.gpu-card {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  font-size: 12px;
  color: var(--text-secondary);
}

.gpu-id {
  font-weight: 600;
  color: var(--text-primary);
  min-width: 48px;
}

.gpu-name {
  flex: 1;
}

.gpu-vram {
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}

.cache-hint {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  font-size: 13px;
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}
</style>
