<template>
  <div 
    class="mc-row"
    :class="{ selected: isSelected }"
    :style="rowStyle"
  >
    <!-- 1. Select + Cache -->
    <div class="col-ctrl">
      <div 
        class="sel"
        :class="{ on: isSelected }"
        @click.stop="$emit('toggle-select', group.target?.path || group.id)"
      >
        <el-icon v-if="isSelected"><Check /></el-icon>
      </div>
      <span class="dot" :class="{ on: group.target?.hasLatentCache }">L</span>
      <span class="dot" :class="{ on: group.target?.hasTextCache }">T</span>
    </div>

    <!-- 2. Name + dimensions (FIRST visible info) -->
    <div class="col-name" @click="$emit('click', group)">
      <span class="name" :title="group.id">{{ group.id }}</span>
      <span class="dim" v-if="group.target?.width">{{ group.target.width }}×{{ group.target.height }}</span>
    </div>

    <!-- 3. Channel thumbnails (smaller, 80px) -->
    <div 
      class="col-thumb" 
      v-for="ch in auxChannels" 
      :key="ch.name"
      @click="$emit('click', group)"
    >
      <div class="thumb">
        <img v-if="ch.image" :src="ch.image.thumbnailUrl" :alt="ch.name" loading="lazy" />
        <div v-else class="thumb-ph"><el-icon><Picture /></el-icon></div>
        <span class="ch-label">{{ ch.name }}</span>
      </div>
    </div>

    <!-- 4. Target -->
    <div class="col-thumb col-target" @click="$emit('click', group)">
      <div class="thumb">
        <img v-if="group.target" :src="group.target.thumbnailUrl" :alt="group.target?.filename" loading="lazy" />
        <div v-else class="thumb-ph"><el-icon><Picture /></el-icon></div>
        <span class="ch-label target-label">target</span>
      </div>
    </div>

    <!-- 5. Caption (clickable to edit) -->
    <div class="col-caption" @click.stop="$emit('edit-caption', group)">
      <span :class="{ none: !group.caption }" :title="group.caption">
        {{ group.caption || '点击编辑标注' }}
      </span>
      <el-icon class="edit-icon"><EditPen /></el-icon>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Check, Picture, EditPen } from '@element-plus/icons-vue'
import type { ImageGroup, ChannelInfo, DatasetImage } from '@/stores/dataset'

const props = defineProps<{
  group: ImageGroup
  channels: ChannelInfo[]
  selected?: boolean
}>()

defineEmits<{
  (e: 'click', group: ImageGroup): void
  (e: 'toggle-select', path: string): void
  (e: 'edit-caption', group: ImageGroup): void
}>()

const isSelected = computed(() => props.selected || false)

interface DisplayChannel {
  name: string
  role: string
  image: DatasetImage | null
}

const auxChannels = computed<DisplayChannel[]>(() => {
  return props.channels
    .filter(ch => ch.role !== 'target')
    .map(ch => ({
      name: ch.name,
      role: ch.role,
      image: props.group.channels[ch.name] || null
    }))
})

const expectedCount = computed(() => props.channels.filter(ch => ch.role !== 'target').length)
const presentCount = computed(() => Object.keys(props.group.channels).length)
const missingCount = computed(() => Math.max(0, expectedCount.value - presentCount.value))
const isComplete = computed(() => missingCount.value === 0)

// Grid: ctrl(60) | name(140) | N×thumbs(1fr each) | target(1fr) | caption(2fr) | status(28)
const totalThumbCols = computed(() => auxChannels.value.length + 1)
const rowStyle = computed(() => {
  const thumbs = Array(totalThumbCols.value).fill('1fr').join(' ')
  return {
    gridTemplateColumns: `56px 140px ${thumbs} minmax(120px, 2fr)`
  }
})
</script>

<style lang="scss" scoped>
.mc-row {
  display: grid;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--bg-card, #1a1a2e);
  border: 1px solid transparent;
  border-radius: 8px;
  cursor: default;
  transition: background 0.15s, border-color 0.15s;

  &:hover {
    background: var(--bg-card-hover, #22223a);
    border-color: rgba(255,255,255,0.06);
  }
  &.selected {
    border-color: var(--primary, #409eff);
    background: rgba(64, 158, 255, 0.05);
  }
}

/* Ctrl col */
.col-ctrl {
  display: flex; align-items: center; gap: 3px;
}
.sel {
  width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.2);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all 0.15s; flex-shrink: 0;
  &:hover { border-color: var(--primary, #409eff); }
  &.on {
    border-color: var(--primary); background: var(--primary);
    .el-icon { color: #fff; font-size: 10px; }
  }
}
.dot {
  width: 14px; height: 14px; border-radius: 3px;
  font-size: 8px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.18);
  &.on { background: var(--success, #22c55e); color: #fff; }
}

/* Name col */
.col-name {
  display: flex; flex-direction: column; gap: 1px;
  overflow: hidden; cursor: pointer;
}
.name {
  font-size: 12px; font-weight: 600;
  color: rgba(255,255,255,0.75);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.dim {
  font-size: 10px; color: rgba(255,255,255,0.3);
}

/* Thumbnail cols */
.col-thumb {
  min-width: 0;
  cursor: pointer;
}
.thumb {
  position: relative;
  width: 100%; height: 80px;
  border-radius: 6px; overflow: hidden;
  background: rgba(0,0,0,0.25);
  border: 1px solid rgba(255,255,255,0.06);
  img { width: 100%; height: 100%; object-fit: contain; display: block; }
}
.thumb-ph {
  width: 100%; height: 100%;
  display: flex; align-items: center; justify-content: center;
  color: rgba(255,255,255,0.1);
  .el-icon { font-size: 16px; }
}
.ch-label {
  position: absolute; bottom: 0; left: 0; right: 0;
  text-align: center; font-size: 8px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.4px;
  color: rgba(255,255,255,0.5);
  background: linear-gradient(transparent, rgba(0,0,0,0.6));
  padding: 6px 2px 2px;
}

/* Target highlight */
.col-target .thumb {
  border: 2px solid var(--primary, #409eff);
  box-shadow: 0 0 8px rgba(64,158,255,0.1);
}
.target-label {
  color: var(--primary, #409eff) !important;
  font-weight: 700 !important;
}

/* Caption col — clickable */
.col-caption {
  min-width: 0; padding: 4px 8px;
  align-self: center; cursor: pointer;
  border-radius: 6px;
  display: flex; align-items: flex-start; gap: 4px;
  transition: background 0.15s;

  &:hover {
    background: rgba(255,255,255,0.04);
    .edit-icon { opacity: 1; }
  }

  span {
    flex: 1; min-width: 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    font-size: 11px; line-height: 1.4;
    color: rgba(255,255,255,0.5);
    word-break: break-all;
    &.none { color: rgba(255,255,255,0.18); font-style: italic; }
  }
  .edit-icon {
    flex-shrink: 0; font-size: 12px;
    color: rgba(255,255,255,0.25);
    opacity: 0; transition: opacity 0.15s;
    margin-top: 1px;
  }
}

</style>
