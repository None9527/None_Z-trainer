<template>
  <div class="zimage-cache-config">
    <el-form-item label="训练模式">
      <el-select v-model="trainingMode" placeholder="请选择训练模式">
        <el-option label="Text2Img (标准)" value="text2img" />
        <el-option label="ControlNet" value="controlnet" />
        <el-option label="Img2Img" value="img2img" />
        <el-option label="Inpaint" value="inpaint" />
        <el-option label="Omni (多图)" value="omni" />
      </el-select>
      <div class="cache-model-hint">
        <span v-if="trainingMode === 'text2img'">标准 LoRA 训练，只需 latent 和 text 缓存</span>
        <span v-else-if="trainingMode === 'controlnet'">ControlNet 训练，需要额外的条件图缓存</span>
        <span v-else-if="trainingMode === 'img2img'">剥离训练，需要源图和目标图 latent 配对</span>
        <span v-else-if="trainingMode === 'inpaint'">Inpaint 训练，需要 mask 图和目标图 latent 配对</span>
        <span v-else-if="trainingMode === 'omni'">多图条件训练，需要条件图 VAE latent + SigLIP 特征</span>
      </div>
    </el-form-item>
    
    <el-form-item label="选择缓存类型">
      <el-checkbox-group v-model="cacheTypes">
        <el-checkbox label="latent">
          Latent 缓存
          <span class="cache-path-hint" v-if="vaePath">({{ shortPath(vaePath) }})</span>
          <span class="cache-path-missing" v-else>(未配置VAE)</span>
        </el-checkbox>
        <el-checkbox label="text">
          Text 缓存
          <span class="cache-path-hint" v-if="textEncoderPath">({{ shortPath(textEncoderPath) }})</span>
          <span class="cache-path-missing" v-else>(未配置Text Encoder)</span>
        </el-checkbox>
        <el-checkbox label="control" v-if="trainingMode === 'controlnet'">
          条件图缓存 (边缘/深度/姿态)
        </el-checkbox>
        <el-checkbox label="siglip" v-if="trainingMode === 'omni'">
          SigLIP 特征缓存
        </el-checkbox>
        <el-checkbox label="dino">
          DINOv3 感知缓存
          <el-tooltip content="预缓存 DINOv3 语义特征，训练时可使用 DINOv3 感知 Loss" placement="top">
            <el-icon class="help-icon" style="margin-left: 4px; font-size: 12px; cursor: help;"><QuestionFilled /></el-icon>
          </el-tooltip>
        </el-checkbox>
      </el-checkbox-group>
    </el-form-item>
    
    <!-- DINOv3 模型路径 (只读，由 .env 配置) -->
    <el-form-item label="DINOv3 模型" v-if="cacheTypes.includes('dino')">
      <div class="dino-env-status">
        <el-tag :type="dinoReady ? 'success' : 'warning'" size="small" effect="plain">
          {{ dinoReady ? '就绪' : '未就绪' }}
        </el-tag>
        <code class="dino-path" v-if="dinoPath">{{ dinoPath }}</code>
        <span class="cache-path-missing" v-else>未配置 DINO_MODEL_PATH</span>
      </div>
      <div class="cache-model-hint">
        <span>模型路径由 .env 中 DINO_MODEL_PATH 配置</span>
      </div>
    </el-form-item>
    
    <!-- ControlNet 条件图目录 -->
    <el-form-item label="条件图目录" v-if="trainingMode === 'controlnet' && cacheTypes.includes('control')">
      <el-input v-model="controlDir" placeholder="条件图像目录路径" />
    </el-form-item>
    
    <!-- Img2Img 源图目录 -->
    <el-form-item label="源图目录" v-if="trainingMode === 'img2img'">
      <el-input v-model="sourceDir" placeholder="源图像目录路径" />
    </el-form-item>
    
    <!-- Inpaint mask 目录 -->
    <el-form-item label="Mask 目录" v-if="trainingMode === 'inpaint'">
      <el-input v-model="maskDir" placeholder="掩码图目录路径" />
    </el-form-item>
    
    <!-- Omni 条件图目录 -->
    <el-form-item label="条件图目录" v-if="trainingMode === 'omni'">
      <el-input v-model="conditionDirs" placeholder="条件图目录 (多个用逗号分隔)" />
      <div class="cache-model-hint">
        <span>每个条件图目录对应一张条件图，文件名必须与目标图一致</span>
      </div>
    </el-form-item>
    <el-form-item label="条件图数量" v-if="trainingMode === 'omni'">
      <el-input-number v-model="numConditionImages" :min="1" :max="10" />
    </el-form-item>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const props = defineProps<{
  datasetPath: string
  vaePath?: string
  textEncoderPath?: string
}>()

const trainingMode = defineModel<string>('trainingMode', { default: 'text2img' })
const cacheTypes = defineModel<string[]>('cacheTypes', { default: () => ['latent', 'text'] })
const controlDir = defineModel<string>('controlDir', { default: '' })
const sourceDir = defineModel<string>('sourceDir', { default: '' })
const maskDir = defineModel<string>('maskDir', { default: '' })
const conditionDirs = defineModel<string>('conditionDirs', { default: '' })
const numConditionImages = defineModel<number>('numConditionImages', { default: 1 })
const dinoModelPath = defineModel<string>('dinoModelPath', { default: '' })

import { QuestionFilled } from '@element-plus/icons-vue'

// DINOv3 env status
const dinoReady = ref(false)
const dinoPath = ref('')

async function fetchDinoStatus() {
  try {
    const res = await axios.get('/api/system/dino/status')
    if (res.data.success && res.data.data) {
      const d = res.data.data
      dinoReady.value = d.status === 'ready'
      dinoPath.value = d.path || ''
      // Sync to parent model so cache generation knows the path
      dinoModelPath.value = d.path || ''
    }
  } catch { }
}

onMounted(fetchDinoStatus)

function shortPath(path: string): string {
  return path.split(/[/\\]/).pop() || path
}
</script>

<style scoped>
.dino-env-status {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dino-path {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  background: var(--el-fill-color-light);
  padding: 2px 8px;
  border-radius: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 400px;
}
</style>
