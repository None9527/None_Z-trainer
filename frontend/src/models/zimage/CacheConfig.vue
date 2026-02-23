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
      </el-checkbox-group>
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
import { ref, computed } from 'vue'

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

function shortPath(path: string): string {
  return path.split(/[/\\]/).pop() || path
}
</script>
