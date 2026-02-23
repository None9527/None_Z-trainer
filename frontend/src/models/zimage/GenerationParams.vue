<template>
  <div class="zimage-generation-params">
    <!-- LoRA 选择 -->
    <div class="param-group">
      <div class="group-label">LORA 模型</div>
      <el-select 
        v-model="params.lora_path" 
        placeholder="选择 LoRA 模型..." 
        clearable 
        filterable
        style="width: 100%; margin-bottom: 8px;"
      >
        <el-option
          v-for="lora in loraList"
          :key="lora.path"
          :label="lora.name"
          :value="lora.path"
        >
          <span style="float: left">{{ lora.name }}</span>
          <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">
            {{ (lora.size_bytes / 1024 / 1024).toFixed(1) }} MB
          </span>
        </el-option>
      </el-select>

      <div v-if="params.lora_path" class="lora-settings">
        <div class="control-row">
          <span class="label">权重</span>
          <el-slider v-model="params.lora_scale" :min="0" :max="2" :step="0.05" :show-tooltip="false" class="slider-flex" />
          <el-input-number v-model="params.lora_scale" :min="0" :max="2" :step="0.05" controls-position="right" class="input-fixed" />
        </div>
        <div class="control-row" style="margin-top: 8px;">
          <span class="label">对比</span>
          <el-switch v-model="params.comparison_mode" active-text="生成原图对比" />
        </div>
      </div>
    </div>

    <!-- 分辨率 -->
    <div class="param-group">
      <div class="group-label">分辨率 (RESOLUTION)</div>
      
      <div class="ratio-grid">
        <el-button 
          v-for="ratio in aspectRatios" 
          :key="ratio.label"
          size="small"
          :type="currentRatio === ratio.label ? 'primary' : 'default'"
          @click="setAspectRatio(ratio)"
          class="ratio-btn"
        >
          {{ ratio.label }}
        </el-button>
      </div>

      <div class="control-row">
        <span class="label">宽度</span>
        <el-slider v-model="params.width" :min="256" :max="2048" :step="64" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="params.width" :min="256" :max="2048" :step="64" controls-position="right" class="input-fixed" />
      </div>

      <div class="control-row">
        <span class="label">高度</span>
        <el-slider v-model="params.height" :min="256" :max="2048" :step="64" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="params.height" :min="256" :max="2048" :step="64" controls-position="right" class="input-fixed" />
      </div>
    </div>

    <!-- 生成设置 -->
    <div class="param-group">
      <div class="group-label">生成设置 (SETTINGS)</div>
      
      <div class="control-row">
        <span class="label">步数</span>
        <el-slider v-model="params.steps" :min="1" :max="50" :step="1" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="params.steps" :min="1" :max="50" controls-position="right" class="input-fixed" />
      </div>

      <div class="control-row">
        <span class="label">引导</span>
        <el-slider v-model="params.guidance_scale" :min="1" :max="20" :step="0.5" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="params.guidance_scale" :min="1" :max="20" :step="0.5" controls-position="right" class="input-fixed" />
      </div>

      <div class="control-row">
        <span class="label">Seed</span>
        <div class="seed-wrapper">
          <el-input-number v-model="params.seed" :min="-1" placeholder="随机" controls-position="right" class="seed-input" />
          <el-button @click="params.seed = -1" icon="Refresh" size="small" class="seed-btn" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'

const params = defineModel<any>('modelValue', { required: true })

const loraList = ref<any[]>([])

const aspectRatios = [
  { label: '1:1', w: 1024, h: 1024 },
  { label: '4:3', w: 1024, h: 768 },
  { label: '3:4', w: 768, h: 1024 },
  { label: '16:9', w: 1024, h: 576 },
  { label: '9:16', w: 576, h: 1024 }
]

const currentRatio = computed(() => {
  const { width, height } = params.value
  const match = aspectRatios.find(r => r.w === width && r.h === height)
  return match ? match.label : 'Custom'
})

function setAspectRatio(ratio: any) {
  params.value.width = ratio.w
  params.value.height = ratio.h
}

async function fetchLoraList() {
  try {
    const res = await axios.get('/api/loras')
    loraList.value = res.data.data || res.data.loras || []
  } catch (e) {
    console.error('Failed to fetch LoRA list:', e)
  }
}

onMounted(() => {
  fetchLoraList()
})
</script>
