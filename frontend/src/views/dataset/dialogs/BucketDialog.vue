<template>
  <el-dialog
    v-model="visible"
    title="分桶计算器"
    width="850px"
    class="bucket-dialog"
    @close="handleClose"
  >
    <!-- Config -->
    <div class="bucket-config">
      <el-form :inline="true" label-width="100px">
        <el-form-item label="Batch Size">
          <el-input-number v-model="config.batchSize" :min="1" :max="16" :disabled="applying" />
        </el-form-item>
        <el-form-item label="分辨率限制">
          <el-input-number v-model="config.resolutionLimit" :min="256" :max="2048" :step="64" :disabled="applying" />
        </el-form-item>
        <el-form-item label="填充策略">
          <el-select v-model="config.fillStrategy" style="width: 160px" :disabled="applying">
            <el-option label="不填充 (丢弃余数)" value="none" />
            <el-option label="重复填充" value="repeat" />
            <el-option label="近似裁切" value="crop" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="calculate" :loading="calculating" :disabled="applying">
            计算分桶
          </el-button>
        </el-form-item>
      </el-form>
      <div class="strategy-hint">
        <el-icon><InfoFilled /></el-icon>
        <span v-if="config.fillStrategy === 'none'">不做处理，不满一批的余数图片将被丢弃</span>
        <span v-else-if="config.fillStrategy === 'repeat'">复制桶内图片补齐不完整批次（轻微过采样，零丢弃）</span>
        <span v-else-if="config.fillStrategy === 'crop'">将余数图片裁切到最近邻桶的宽高比（center crop），使其归入邻桶</span>
      </div>
    </div>
    
    <!-- Results -->
    <div class="bucket-results" v-if="results.length > 0">
      <!-- Summary -->
      <div class="bucket-summary">
        <div class="summary-item">
          <span class="label">原始图片</span>
          <span class="value">{{ serverSummary.totalImages }}</span>
        </div>
        <div class="summary-item" v-if="config.fillStrategy !== 'none'">
          <span class="label">有效图片</span>
          <span class="value">{{ serverSummary.totalEffective }}</span>
        </div>
        <div class="summary-item">
          <span class="label">桶数量</span>
          <span class="value">{{ results.length }}</span>
        </div>
        <div class="summary-item">
          <span class="label">总批次</span>
          <span class="value">{{ totalBatches }}</span>
        </div>
        <div class="summary-item" v-if="serverSummary.totalRepeated > 0">
          <span class="label">重复填充</span>
          <span class="value text-info">+{{ serverSummary.totalRepeated }}</span>
        </div>
        <div class="summary-item" v-if="serverSummary.totalCropped > 0">
          <span class="label">裁切合并</span>
          <span class="value text-success">{{ serverSummary.totalCropped }}</span>
        </div>
        <div class="summary-item">
          <span class="label">丢弃图片</span>
          <span class="value" :class="{ 'text-warning': serverSummary.totalDropped > 0, 'text-success': serverSummary.totalDropped === 0 && config.fillStrategy !== 'none' }">
            {{ serverSummary.totalDropped }}
          </span>
        </div>
      </div>
      
      <!-- Table -->
      <el-table :data="results" style="width: 100%" max-height="360" size="small">
        <el-table-column prop="resolution" label="分辨率" width="100">
          <template #default="{ row }">
            {{ row.width }}×{{ row.height }}
          </template>
        </el-table-column>
        <el-table-column prop="aspectRatio" label="宽高比" width="80">
          <template #default="{ row }">
            {{ row.aspectRatio.toFixed(2) }}
          </template>
        </el-table-column>
        <el-table-column label="图片" width="110">
          <template #default="{ row }">
            <span>{{ row.count }}</span>
            <template v-if="row.repeated > 0 || row.croppedIn > 0 || row.croppedOut > 0">
              <span class="count-detail">
                ({{ row.original }}
                <span v-if="row.croppedIn > 0" class="text-success">+{{ row.croppedIn }}裁入</span>
                <span v-if="row.croppedOut > 0" class="text-muted">-{{ row.croppedOut }}裁出</span>
                <span v-if="row.repeated > 0" class="text-info">+{{ row.repeated }}复</span>)
              </span>
            </template>
          </template>
        </el-table-column>
        <el-table-column prop="batches" label="批次" width="55" />
        <el-table-column label="丢弃" width="55">
          <template #default="{ row }">
            <span :class="{ 'text-warning': row.dropped > 0 }">{{ row.dropped }}</span>
          </template>
        </el-table-column>
        <el-table-column label="分布" min-width="150">
          <template #default="{ row }">
            <el-progress 
              :percentage="row.percentage" 
              :stroke-width="12"
              :show-text="false"
              :color="getBucketColor(row.aspectRatio)"
            />
          </template>
        </el-table-column>
      </el-table>

      <!-- Crop remainder hint -->
      <el-alert
        v-if="config.fillStrategy === 'crop' && serverSummary.totalDropped > 0"
        type="info"
        :closable="false"
        show-icon
        style="margin-top: 12px"
      >
        总图片数 {{ serverSummary.totalImages }} 不是 batch_size {{ config.batchSize }} 的倍数，纯裁切最少丢弃 {{ serverSummary.totalImages % config.batchSize }} 张
      </el-alert>

      <!-- Apply action -->
      <div class="apply-section" v-if="config.fillStrategy !== 'none' && canApply">
        <el-divider />
        <div v-if="!applying" class="apply-confirm">
          <el-alert
            :type="config.fillStrategy === 'crop' ? 'warning' : 'info'"
            :closable="false"
            show-icon
          >
            <template #title>
              <span v-if="config.fillStrategy === 'repeat'">
                将复制 <strong>{{ serverSummary.totalRepeated }}</strong> 张图片（含对应标注文件）以补齐批次
              </span>
              <span v-else-if="config.fillStrategy === 'crop'">
                将对 <strong>{{ serverSummary.totalCropped }}</strong> 张图片执行 center crop（不可逆，建议先备份）
              </span>
            </template>
          </el-alert>
          <el-button
            type="warning"
            @click="handleApply"
            style="margin-top: 12px"
          >
            <el-icon><Check /></el-icon>
            确认应用方案
          </el-button>
        </div>

        <!-- Progress -->
        <div v-else class="apply-progress">
          <div class="progress-header">
            <span>正在处理: {{ applyStatus.current_file || '...' }}</span>
            <span>{{ applyStatus.completed }}/{{ applyStatus.total }}</span>
          </div>
          <el-progress
            :percentage="applyProgress"
            :stroke-width="16"
            :status="applyProgressStatus"
          />
          <div v-if="applyStatus.errors.length > 0" class="apply-errors">
            <el-text type="danger" size="small">
              {{ applyStatus.errors.length }} 个错误
            </el-text>
          </div>
          <el-button
            v-if="applyStatus.running"
            type="danger"
            size="small"
            @click="handleStop"
            style="margin-top: 8px"
          >
            停止
          </el-button>
          <div v-if="!applyStatus.running && applyStatus.completed > 0" class="apply-done">
            <el-tag type="success">已完成 {{ applyStatus.completed }} 项处理</el-tag>
          </div>
        </div>
      </div>
    </div>
    
    <div class="bucket-empty" v-else-if="!calculating">
      <el-icon :size="48"><Grid /></el-icon>
      <p>点击「计算分桶」查看数据集的分桶分布</p>
    </div>
    
    <template #footer>
      <el-button @click="handleClose" :disabled="applying && applyStatus.running">关闭</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { Grid, InfoFilled, Check } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

const props = defineProps<{
  modelValue: boolean
  datasetPath: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

const visible = ref(props.modelValue)
const config = ref({
  batchSize: 4,
  resolutionLimit: 1536,
  fillStrategy: 'none' as string,
})
const calculating = ref(false)
const applying = ref(false)

interface BucketInfo {
  width: number; height: number; aspectRatio: number
  count: number; original: number; batches: number; dropped: number
  repeated: number; croppedIn: number; croppedOut: number; percentage: number
}
interface SummaryInfo {
  totalImages: number; totalEffective: number; totalRepeated: number
  totalCropped: number; totalDropped: number; strategy: string
}

const results = ref<BucketInfo[]>([])
const serverSummary = ref<SummaryInfo>({
  totalImages: 0, totalEffective: 0, totalRepeated: 0,
  totalCropped: 0, totalDropped: 0, strategy: 'none',
})
const applyStatus = ref({ running: false, total: 0, completed: 0, current_file: '', errors: [] as string[] })
let pollTimer: ReturnType<typeof setInterval> | null = null

watch(() => props.modelValue, (val) => { visible.value = val })
watch(visible, (val) => { emit('update:modelValue', val) })

const totalBatches = computed(() => results.value.reduce((sum, b) => sum + b.batches, 0))
const canApply = computed(() => {
  if (config.value.fillStrategy === 'repeat') return serverSummary.value.totalRepeated > 0
  if (config.value.fillStrategy === 'crop') return serverSummary.value.totalCropped > 0
  return false
})
const applyProgress = computed(() => {
  if (applyStatus.value.total === 0) return 0
  return Math.round(applyStatus.value.completed / applyStatus.value.total * 100)
})
const applyProgressStatus = computed(() => {
  if (!applyStatus.value.running && applyStatus.value.completed > 0) return 'success'
  if (applyStatus.value.errors.length > 0) return 'exception'
  return undefined
})

function getBucketColor(aspectRatio: number): string {
  if (aspectRatio < 0.8) return '#67c23a'
  if (aspectRatio > 1.2) return '#409eff'
  return '#f0b429'
}

async function calculate() {
  if (!props.datasetPath) return
  calculating.value = true
  results.value = []
  try {
    const response = await axios.post('/api/dataset/calculate-buckets', {
      path: props.datasetPath,
      batch_size: config.value.batchSize,
      resolution_limit: config.value.resolutionLimit,
      fill_strategy: config.value.fillStrategy,
    })
    results.value = response.data.buckets
    if (response.data.summary) serverSummary.value = response.data.summary
  } catch (error: any) {
    ElMessage.error('计算分桶失败: ' + (error.response?.data?.detail || error.message))
  } finally {
    calculating.value = false
  }
}

async function handleApply() {
  const strategyName = config.value.fillStrategy === 'repeat' ? '重复填充' : '近似裁切'
  const count = config.value.fillStrategy === 'repeat'
    ? serverSummary.value.totalRepeated
    : serverSummary.value.totalCropped
  const warning = config.value.fillStrategy === 'crop'
    ? '\n⚠️ 裁切操作不可逆，将直接修改原图文件！'
    : ''

  try {
    await ElMessageBox.confirm(
      `确定执行「${strategyName}」方案？\n将处理 ${count} 张图片。${warning}`,
      '确认应用',
      { confirmButtonText: '确认执行', cancelButtonText: '取消', type: 'warning' }
    )
  } catch {
    return
  }

  applying.value = true
  applyStatus.value = { running: true, total: 0, completed: 0, current_file: '', errors: [] }

  try {
    const res = await axios.post('/api/dataset/buckets/apply', {
      path: props.datasetPath,
      batch_size: config.value.batchSize,
      resolution_limit: config.value.resolutionLimit,
      strategy: config.value.fillStrategy,
    })

    if (!res.data.success) {
      ElMessage.warning(res.data.message || '无法执行')
      applying.value = false
      return
    }

    if (res.data.total === 0) {
      ElMessage.info(res.data.message || '无需处理')
      applying.value = false
      return
    }

    applyStatus.value.total = res.data.total
    startPollStatus()
  } catch (error: any) {
    ElMessage.error('应用失败: ' + (error.response?.data?.detail || error.message))
    applying.value = false
  }
}

function startPollStatus() {
  stopPollStatus()
  pollTimer = setInterval(async () => {
    try {
      const res = await axios.get('/api/dataset/buckets/status')
      applyStatus.value = res.data
      if (!res.data.running) {
        stopPollStatus()
        if (res.data.completed > 0 && res.data.errors.length === 0) {
          ElMessage.success(`处理完成，共 ${res.data.completed} 张图片`)
        }
      }
    } catch { /* ignore */ }
  }, 500)
}

function stopPollStatus() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function handleStop() {
  try {
    await axios.post('/api/dataset/buckets/stop')
    ElMessage.info('已停止')
  } catch { /* ignore */ }
}

function handleClose() {
  if (applying.value && applyStatus.value.running) return
  stopPollStatus()
  applying.value = false
  visible.value = false
}

onBeforeUnmount(() => stopPollStatus())
</script>

<style lang="scss" scoped>
.bucket-config {
  margin-bottom: var(--space-md);
}

.strategy-hint {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  background: var(--bg-tertiary);
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 4px;
}

.bucket-summary {
  display: flex;
  gap: var(--space-lg);
  margin-bottom: var(--space-md);
  padding: var(--space-md);
  background: var(--bg-tertiary);
  border-radius: var(--radius-md);
  flex-wrap: wrap;
  
  .summary-item {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
    
    .label { font-size: 12px; color: var(--text-muted); }
    .value { font-size: 20px; font-weight: bold; }
  }
}

.count-detail {
  font-size: 11px;
  color: var(--text-muted);
  margin-left: 2px;
}

.text-warning { color: var(--color-warning, #e6a23c); }
.text-success { color: var(--color-success, #67c23a); }
.text-info { color: var(--color-primary, #409eff); }
.text-muted { color: var(--text-muted); }

.apply-section {
  margin-top: var(--space-sm);
}

.apply-confirm {
  text-align: center;
}

.apply-progress {
  .progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 13px;
    color: var(--text-secondary);
  }
  .apply-errors { margin-top: 8px; }
  .apply-done { margin-top: 12px; text-align: center; }
}

.bucket-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--space-xl);
  color: var(--text-muted);
  p { margin-top: var(--space-md); }
}
</style>
