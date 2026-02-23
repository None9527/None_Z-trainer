<template>
  <el-dialog
    v-model="visible"
    title="批量缩放图片"
    width="600px"
    @close="handleClose"
  >
    <el-form label-width="100px" :disabled="resizing">
      <el-form-item label="长边尺寸">
        <el-slider v-model="config.maxLongEdge" :min="256" :max="2048" :step="64" show-input />
        <div class="form-hint">超过此尺寸的图片将被缩小</div>
      </el-form-item>
      
      <el-form-item label="JPEG 质量">
        <el-slider v-model="config.quality" :min="70" :max="100" show-input />
        <div class="form-hint">输出 JPEG 压缩质量 (95 推荐)</div>
      </el-form-item>
      
      <el-form-item label="锐化强度">
        <el-slider v-model="config.sharpen" :min="0" :max="2" :step="0.1" show-input />
        <div class="form-hint">0 = 不锐化, 1.0 = 适中</div>
      </el-form-item>
    </el-form>
    
    <!-- 进度显示 -->
    <div class="resize-progress" v-if="resizing || status.completed > 0">
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
    
    <el-alert
      title="警告：此操作将覆盖原始图片"
      type="warning"
      :closable="false"
      show-icon
      style="margin-top: 16px"
    />
    
    <template #footer>
      <el-button @click="handleClose" :disabled="resizing">关闭</el-button>
      <el-button v-if="resizing" type="danger" @click="$emit('stop')">
        停止缩放
      </el-button>
      <el-button v-else type="danger" @click="handleConfirm">
        确认缩放
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

interface ResizeStatus {
  running: boolean
  total: number
  completed: number
  current_file: string
  errors: string[]
}

// Props
const props = defineProps<{
  modelValue: boolean
  resizing?: boolean
  status?: ResizeStatus
}>()

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm', config: { maxLongEdge: number, quality: number, sharpen: number }): void
  (e: 'stop'): void
}>()

// 本地状态
const visible = ref(props.modelValue)
const config = ref({
  maxLongEdge: 1536,
  quality: 95,
  sharpen: 0.5
})

// 默认状态
const defaultStatus: ResizeStatus = {
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

// 确认缩放
function handleConfirm() {
  emit('confirm', { ...config.value })
}

// 关闭对话框
function handleClose() {
  visible.value = false
}
</script>

<style lang="scss" scoped>
.form-hint {
  margin-top: var(--space-xs);
  font-size: 12px;
  color: var(--text-muted);
}

.resize-progress {
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
