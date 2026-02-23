<template>
  <el-dialog
    v-model="visible"
    title="上传到指定通道"
    width="520px"
    @close="handleClose"
  >
    <!-- Step 1: Select channel -->
    <div class="channel-upload-form">
      <div class="form-section">
        <label class="form-label">目标通道</label>
        <div class="channel-selector">
          <!-- Existing channels as quick-pick buttons -->
          <div class="channel-presets" v-if="existingChannels.length > 0">
            <el-tag
              v-for="ch in existingChannels"
              :key="ch.name"
              :type="selectedChannel === ch.name ? 'primary' : 'info'"
              :effect="selectedChannel === ch.name ? 'dark' : 'plain'"
              class="channel-tag"
              @click="selectChannel(ch.name)"
            >
              {{ ch.name }}
              <span class="channel-role">({{ roleLabel(ch.role) }})</span>
            </el-tag>
          </div>
          <!-- Common channel presets -->
          <div class="channel-presets">
            <el-tag
              v-for="preset in commonPresets"
              :key="preset"
              :type="selectedChannel === preset ? 'primary' : 'info'"
              :effect="selectedChannel === preset ? 'dark' : 'plain'"
              class="channel-tag preset"
              @click="selectChannel(preset)"
            >
              + {{ preset }}
            </el-tag>
          </div>
          <!-- Custom channel name input -->
          <el-input
            v-model="customChannel"
            placeholder="或输入自定义通道名..."
            size="small"
            class="custom-input"
            @input="selectedChannel = customChannel"
            @focus="selectedChannel = customChannel"
          >
            <template #prefix>
              <el-icon><FolderOpened /></el-icon>
            </template>
          </el-input>
        </div>
      </div>

      <!-- Step 2: File selection -->
      <div class="form-section">
        <label class="form-label">选择文件 ({{ selectedFiles.length }} 个已选)</label>
        <div
          class="file-drop-zone"
          :class="{ 'drag-over': isDragOver, 'has-files': selectedFiles.length > 0 }"
          @dragover.prevent="isDragOver = true"
          @dragleave="isDragOver = false"
          @drop.prevent="handleDrop"
          @click="fileInputRef?.click()"
        >
          <input
            type="file"
            ref="fileInputRef"
            multiple
            hidden
            accept="image/*,.txt"
            @change="handleFileSelect"
          />
          <div class="drop-content" v-if="selectedFiles.length === 0">
            <el-icon :size="32" color="var(--el-text-color-secondary)"><Upload /></el-icon>
            <p>点击选择图片 或 拖拽文件到此</p>
          </div>
          <div class="file-preview" v-else>
            <div class="file-count">
              <el-icon color="var(--el-color-success)"><CircleCheckFilled /></el-icon>
              <span>{{ selectedFiles.length }} 个文件已选择</span>
              <el-button text type="danger" size="small" @click.stop="clearFiles">清除</el-button>
            </div>
            <div class="file-list">
              <span v-for="f in selectedFiles.slice(0, 5)" :key="f.name" class="file-name">{{ f.name }}</span>
              <span v-if="selectedFiles.length > 5" class="file-more">...+{{ selectedFiles.length - 5 }} 更多</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Upload info -->
      <div class="upload-target-info" v-if="activeChannel">
        <el-icon><InfoFilled /></el-icon>
        <span>文件将上传到: <code>{{ datasetName }}/{{ activeChannel }}/</code></span>
      </div>

      <!-- Upload progress -->
      <div class="upload-progress" v-if="uploading">
        <el-progress :percentage="progress" :status="progressStatus" :stroke-width="16" striped striped-flow />
        <span class="progress-text">{{ progressText }}</span>
      </div>
    </div>

    <template #footer>
      <el-button @click="visible = false">取消</el-button>
      <el-button
        type="primary"
        :disabled="!canUpload"
        :loading="uploading"
        @click="startUpload"
      >
        <el-icon><Upload /></el-icon>
        上传到 {{ activeChannel || '...' }}
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Upload, FolderOpened, CircleCheckFilled, InfoFilled } from '@element-plus/icons-vue'
import axios from 'axios'
import type { ChannelInfo } from '@/stores/dataset'

const props = defineProps<{
  modelValue: boolean
  datasetName: string
  channels: ChannelInfo[]
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'uploaded'): void
}>()

const visible = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v)
})

// Channel selection
const selectedChannel = ref('')
const customChannel = ref('')
const existingChannels = computed(() => props.channels || [])

const commonPresets = computed(() => {
  const existing = new Set(existingChannels.value.map(c => c.name))
  return ['target', 'source', 'depth', 'canny', 'normal', 'pose', 'mask'].filter(p => !existing.has(p))
})

const activeChannel = computed(() => selectedChannel.value.trim())

function selectChannel(name: string) {
  selectedChannel.value = name
  customChannel.value = name
}

function roleLabel(role: string) {
  const map: Record<string, string> = { target: '目标', source: '源图', condition: '条件', reference: '参考' }
  return map[role] || role
}

// File selection
const fileInputRef = ref<HTMLInputElement | null>(null)
const selectedFiles = ref<File[]>([])
const isDragOver = ref(false)

function handleFileSelect(event: Event) {
  const input = event.target as HTMLInputElement
  if (input.files) {
    const files = Array.from(input.files).filter(f => f.type.startsWith('image/') || f.name.endsWith('.txt'))
    selectedFiles.value = [...selectedFiles.value, ...files]
  }
}

function handleDrop(event: DragEvent) {
  isDragOver.value = false
  const files = Array.from(event.dataTransfer?.files || []).filter(f => f.type.startsWith('image/') || f.name.endsWith('.txt'))
  selectedFiles.value = [...selectedFiles.value, ...files]
}

function clearFiles() {
  selectedFiles.value = []
  if (fileInputRef.value) fileInputRef.value.value = ''
}

// Upload logic
const uploading = ref(false)
const progress = ref(0)
const progressText = ref('')
const progressStatus = ref<'' | 'success' | 'exception'>('')
const canUpload = computed(() => activeChannel.value.length > 0 && selectedFiles.value.length > 0 && !uploading.value)

async function startUpload() {
  if (!canUpload.value) return
  uploading.value = true
  progress.value = 0
  progressStatus.value = ''

  const files = selectedFiles.value
  const batchSize = 20
  let successCount = 0
  let failCount = 0

  try {
    for (let i = 0; i < files.length; i += batchSize) {
      const batch = files.slice(i, i + batchSize)
      const formData = new FormData()
      formData.append('dataset_name', props.datasetName)
      formData.append('channel_name', activeChannel.value)
      batch.forEach(f => formData.append('files', f))

      progressText.value = `上传 ${i + 1}-${Math.min(i + batchSize, files.length)} / ${files.length}`

      try {
        const res = await axios.post('/api/dataset/upload-to-channel', formData)
        successCount += res.data.uploaded
        if (res.data.errors?.length) failCount += res.data.errors.length
      } catch {
        failCount += batch.length
      }

      progress.value = Math.round(((i + batchSize) / files.length) * 100)
    }

    progress.value = 100
    progressStatus.value = failCount === 0 ? 'success' : 'exception'
    progressText.value = `完成: ${successCount} 成功, ${failCount} 失败`

    if (successCount > 0) {
      ElMessage.success(`${successCount} 个文件已上传到 ${activeChannel.value}/ 通道`)
      emit('uploaded')
    }
  } catch (e: any) {
    progressStatus.value = 'exception'
    progressText.value = '上传出错: ' + e.message
  } finally {
    uploading.value = false
  }
}

function handleClose() {
  if (!uploading.value) {
    selectedFiles.value = []
    selectedChannel.value = ''
    customChannel.value = ''
    progress.value = 0
    progressText.value = ''
    progressStatus.value = ''
  }
}

watch(visible, (v) => { if (v) handleClose() })
</script>

<style scoped>
.channel-upload-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.form-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.form-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-regular);
}
.channel-selector {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.channel-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.channel-tag {
  cursor: pointer;
  transition: all 0.2s;
}
.channel-tag:hover {
  transform: scale(1.05);
}
.channel-role {
  font-size: 11px;
  opacity: 0.7;
  margin-left: 2px;
}
.channel-tag.preset {
  border-style: dashed;
}
.custom-input {
  margin-top: 4px;
}

.file-drop-zone {
  border: 2px dashed var(--el-border-color);
  border-radius: 8px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.2s;
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.file-drop-zone:hover,
.file-drop-zone.drag-over {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}
.file-drop-zone.has-files {
  border-color: var(--el-color-success);
  border-style: solid;
}
.drop-content {
  text-align: center;
  color: var(--el-text-color-secondary);
}
.drop-content p {
  margin: 8px 0 0;
  font-size: 13px;
}
.file-preview {
  width: 100%;
}
.file-count {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 8px;
}
.file-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.file-name {
  font-size: 11px;
  padding: 2px 6px;
  background: var(--el-fill-color-light);
  border-radius: 4px;
  color: var(--el-text-color-secondary);
}
.file-more {
  font-size: 11px;
  padding: 2px 6px;
  color: var(--el-text-color-placeholder);
}

.upload-target-info {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  background: var(--el-fill-color-lighter);
  padding: 8px 12px;
  border-radius: 6px;
}
.upload-target-info code {
  font-weight: 600;
  color: var(--el-color-primary);
}

.upload-progress {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.progress-text {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  text-align: center;
}
</style>
