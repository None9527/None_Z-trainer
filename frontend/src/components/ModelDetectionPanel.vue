<template>
  <div class="model-detection-panel">
    <el-card shadow="never">
      <template #header>
        <div class="panel-header">
          <span>模型检测</span>
          <el-button type="primary" size="small" @click="detectModels" :loading="loading">
            <el-icon><Search /></el-icon> 检测
          </el-button>
        </div>
      </template>
      <div v-if="!detected" class="empty-state">
        <el-empty description="点击「检测」按钮扫描可用模型" />
      </div>
      <div v-else>
        <el-descriptions :column="2" border>
          <el-descriptions-item label="模型状态">
            <el-tag :type="modelReady ? 'success' : 'danger'">
              {{ modelReady ? '可用' : '不可用' }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="模型路径">
            {{ modelPath || '未检测到' }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Search } from '@element-plus/icons-vue'
import { useSystemStore } from '@/stores/system'

const systemStore = useSystemStore()
const loading = ref(false)
const detected = ref(false)
const modelReady = ref(false)
const modelPath = ref('')

async function detectModels() {
  loading.value = true
  try {
    // Use system store's model status (updated via WebSocket)
    detected.value = true
    modelReady.value = systemStore.modelStatus.downloaded
    modelPath.value = systemStore.modelStatus.path
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.empty-state {
  padding: 20px 0;
}
</style>
