<template>
  <el-dialog
    v-model="visible"
    title="Ollama 图片标注"
    width="600px"
    @open="loadModels"
    @close="handleClose"
  >
    <el-form label-width="100px" :disabled="tagging">
      <el-form-item label="Ollama 地址">
        <div class="url-input-row">
          <el-input v-model="config.url" placeholder="http://localhost:11434" />
          <el-button @click="testConnection" :loading="testing">
            测试
          </el-button>
        </div>
      </el-form-item>
      
      <el-form-item label="模型">
        <el-select v-model="config.model" placeholder="选择模型" style="width: 100%">
          <el-option v-for="m in models" :key="m" :label="m" :value="m" />
        </el-select>
        <div class="form-hint" v-if="models.length === 0">
          请先测试连接以获取模型列表
        </div>
      </el-form-item>
      
      <el-form-item label="长边尺寸">
        <el-slider v-model="config.maxLongEdge" :min="512" :max="2048" :step="64" show-input />
        <div class="form-hint">图片将被缩放到此尺寸再发送给 Ollama</div>
      </el-form-item>
      
      <el-form-item label="提示词">
        <el-input
          v-model="config.prompt"
          type="textarea"
          :rows="6"
          placeholder="描述这张图片..."
        />
      </el-form-item>
      
      <el-form-item label="跳过已有">
        <el-switch v-model="config.skipExisting" />
        <span class="switch-label">跳过已有 .txt 标注的图片</span>
      </el-form-item>
      
      <el-form-item label="触发词">
        <el-input 
          v-model="config.triggerWord" 
          placeholder="如: zst_style, my_character"
          clearable
        />
        <div class="form-hint">将此词添加到所有标注开头，用于 LoRA 训练触发</div>
      </el-form-item>
      
    </el-form>
    
    <!-- 进度显示 -->
    <div class="ollama-progress" v-if="tagging || status.completed > 0">
      <el-progress 
        :percentage="progress" 
        :status="status.running ? '' : 'success'"
      />
      <div class="progress-info">
        <span>{{ status.completed }} / {{ status.total }}</span>
        <span v-if="status.current_file">正在处理: {{ status.current_file }}</span>
        <span v-if="status.errors && status.errors.length > 0" class="error-count">
          失败: {{ status.errors.length }}
        </span>
      </div>
    </div>
    
    <template #footer>
      <el-button @click="handleClose" :disabled="tagging">关闭</el-button>
      <el-button v-if="tagging" type="danger" @click="$emit('stop')">
        停止标注
      </el-button>
      <el-button v-else type="primary" @click="handleStart" :disabled="!canStart">
        开始标注
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

interface OllamaStatus {
  running: boolean
  total: number
  completed: number
  current_file: string
  errors: string[]
}

// Props
const props = defineProps<{
  modelValue: boolean
  datasetPath: string
  tagging?: boolean
  status?: OllamaStatus
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'start', config: OllamaConfig): void
  (e: 'stop'): void
}>()

interface OllamaConfig {
  url: string
  model: string
  prompt: string
  maxLongEdge: number
  skipExisting: boolean
  triggerWord: string
}

// 本地状态
const visible = ref(props.modelValue)
const models = ref<string[]>([])
const testing = ref(false)
const config = ref<OllamaConfig>({
  url: 'http://localhost:11434',
  model: '',
  prompt: `你是一位专门为 AI 绘画模型训练服务的打标专家。请为这张图片生成训练标注。

规则：
1. 使用中文短语/Tag 格式，用逗号分隔
2. 描述主体特征：人物、衣着、动作、物品等
3. 不要描述光影、背景、构图、风格
4. 简洁明了，不要写长句

示例输出：1个女孩, 黑发, 齐肩发, 白色连衣裙, 手摸脸, 微笑`,
  maxLongEdge: 1024,
  skipExisting: true,
  triggerWord: ''
})

// 默认状态
const defaultStatus: OllamaStatus = {
  running: false,
  total: 0,
  completed: 0,
  current_file: '',
  errors: []
}

// 同步 visible 与 modelValue
watch(() => props.modelValue, (val) => {
  visible.value = val
})

watch(visible, (val) => {
  emit('update:modelValue', val)
})

// status 安全访问
const status = computed(() => props.status || defaultStatus)

// 进度百分比
const progress = computed(() => {
  if (status.value.total === 0) return 0
  return Math.round((status.value.completed / status.value.total) * 100)
})

// 是否可以开始
const canStart = computed(() => {
  return config.value.url && config.value.model && !props.tagging
})

// 加载模型列表
async function loadModels() {
  if (models.value.length === 0 && config.value.url) {
    await testConnection()
  }
}

// 测试连接
async function testConnection() {
  testing.value = true
  try {
    const response = await axios.get('/api/dataset/ollama/models', {
      params: { ollama_url: config.value.url }
    })
    
    if (response.data.success) {
      models.value = response.data.models
      if (models.value.length > 0 && !config.value.model) {
        config.value.model = models.value[0]
      }
      ElMessage.success(`连接成功，发现 ${models.value.length} 个模型`)
    } else {
      ElMessage.error(response.data.error || '连接失败')
    }
  } catch (error: any) {
    ElMessage.error('连接失败: ' + error.message)
  } finally {
    testing.value = false
  }
}

// 开始标注
function handleStart() {
  emit('start', { ...config.value })
}

// 关闭对话框
function handleClose() {
  visible.value = false
}
</script>

<style lang="scss" scoped>
.url-input-row {
  display: flex;
  gap: var(--space-sm);
}

.form-hint {
  margin-top: var(--space-xs);
  font-size: 12px;
  color: var(--text-muted);
}

.switch-label {
  margin-left: var(--space-sm);
  font-size: 13px;
  color: var(--text-secondary);
}

.ollama-progress {
  margin-top: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  
  .progress-info {
    display: flex;
    gap: var(--space-md);
    margin-top: var(--space-sm);
    font-size: 12px;
    color: var(--text-secondary);
    
    .error-count {
      color: var(--color-danger);
    }
  }
}
</style>
