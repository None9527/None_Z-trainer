<template>
  <el-dialog
    v-model="visible"
    :title="`多通道预览 — ${group?.id || ''}`"
    width="70%"
    top="5vh"
    destroy-on-close
    class="mc-preview-dialog"
    :close-on-click-modal="true"
    @close="$emit('update:modelValue', false)"
  >
    <div class="preview-grid" :style="gridStyle" v-if="group">
      <div 
        class="preview-cell" 
        v-for="ch in allChannels" 
        :key="ch.name"
        :class="{ 'is-target': ch.isTarget }"
      >
        <div class="cell-header">
          <span class="cell-name" :class="{ target: ch.isTarget }">{{ ch.name }}</span>
          <span class="cell-dim" v-if="ch.image?.width">{{ ch.image.width }}×{{ ch.image.height }}</span>
        </div>
        <div class="cell-img">
          <img 
            v-if="ch.image" 
            :src="ch.image.thumbnailUrl" 
            :alt="ch.name" 
            @click="zoomedChannel = zoomedChannel === ch.name ? null : ch.name"
          />
          <div v-else class="cell-empty">
            <el-icon :size="32"><Picture /></el-icon>
            <span>无图片</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Zoomed overlay -->
    <Teleport to="body">
      <div class="zoom-overlay" v-if="zoomedChannel && zoomedImage" @click="zoomedChannel = null">
        <img :src="zoomedImage.thumbnailUrl" :alt="zoomedChannel" />
        <span class="zoom-label">{{ zoomedChannel }}</span>
      </div>
    </Teleport>

    <template #footer>
      <div class="dialog-footer">
        <div class="footer-info">
          <span class="caption-preview" v-if="group?.caption">
            <el-icon><EditPen /></el-icon>
            {{ group.caption }}
          </span>
          <span class="caption-preview none" v-else>无标注</span>
        </div>
        <div class="footer-actions">
          <el-button @click="group && $emit('edit-caption', group)" type="primary" plain size="small">
            <el-icon><EditPen /></el-icon>编辑标注
          </el-button>
          <el-button @click="visible = false" size="small">关闭</el-button>
        </div>
      </div>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { Picture, EditPen } from '@element-plus/icons-vue'
import type { ImageGroup, ChannelInfo, DatasetImage } from '@/stores/dataset'

const props = defineProps<{
  modelValue: boolean
  group: ImageGroup | null
  channels: ChannelInfo[]
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', val: boolean): void
  (e: 'edit-caption', group: ImageGroup): void
}>()

const visible = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v)
})

const zoomedChannel = ref<string | null>(null)

interface DisplayChannel {
  name: string
  isTarget: boolean
  image: DatasetImage | null
}

const allChannels = computed<DisplayChannel[]>(() => {
  if (!props.group) return []
  
  const inputs: DisplayChannel[] = props.channels
    .filter(ch => ch.role !== 'target')
    .map(ch => ({
      name: ch.name,
      isTarget: false,
      image: props.group!.channels[ch.name] || null
    }))

  const target: DisplayChannel = {
    name: 'target',
    isTarget: true,
    image: props.group!.target || null
  }

  return [...inputs, target]
})

const gridStyle = computed(() => {
  const cols = allChannels.value.length
  return {
    gridTemplateColumns: `repeat(${cols}, 1fr)`
  }
})

const zoomedImage = computed(() => {
  if (!zoomedChannel.value || !props.group) return null
  if (zoomedChannel.value === 'target') return props.group.target
  return props.group.channels[zoomedChannel.value] || null
})
</script>

<style lang="scss" scoped>
.mc-preview-dialog {
  :deep(.el-dialog__body) {
    padding: 12px 20px;
  }
}

.preview-grid {
  display: grid;
  gap: 12px;
  width: 100%;
}

.preview-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.cell-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2px;
}

.cell-name {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255,255,255,0.5);

  &.target {
    color: var(--primary, #409eff);
    font-weight: 700;
  }
}

.cell-dim {
  font-size: 10px;
  color: rgba(255,255,255,0.25);
}

.cell-img {
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background: rgba(0,0,0,0.4);
  border: 1px solid rgba(255,255,255,0.06);

  img {
    width: 100%;
    max-height: 55vh;
    object-fit: contain;
    display: block;
    cursor: zoom-in;
    transition: transform 0.2s ease;

    &:hover {
      transform: scale(1.02);
    }
  }
}

.is-target .cell-img {
  border: 2px solid var(--primary, #409eff);
  box-shadow: 0 0 16px rgba(64,158,255,0.1);
}

.cell-empty {
  aspect-ratio: 3/4;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  color: rgba(255,255,255,0.12);
  span { font-size: 11px; }
}

/* Zoom overlay */
.zoom-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  z-index: 9999;
  background: rgba(0,0,0,0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: zoom-out;

  img {
    max-width: 90vw;
    max-height: 90vh;
    object-fit: contain;
    border-radius: 4px;
  }

  .zoom-label {
    position: absolute;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 14px;
    font-weight: 600;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
  }
}

/* Footer */
.dialog-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}
.footer-info {
  flex: 1;
  min-width: 0;
}
.caption-preview {
  font-size: 12px;
  color: rgba(255,255,255,0.5);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 4px;
  &.none { color: rgba(255,255,255,0.2); font-style: italic; }
}
.footer-actions {
  display: flex;
  gap: 8px;
}
</style>
