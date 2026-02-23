<template>
  <el-dialog
    v-model="visible"
    :title="`编辑标注 — ${imageName}`"
    width="560px"
    destroy-on-close
    class="caption-edit-dialog"
    @open="onOpen"
  >
    <el-input
      ref="textareaRef"
      v-model="captionText"
      type="textarea"
      :autosize="{ minRows: 4, maxRows: 12 }"
      placeholder="输入图片描述..."
      resize="vertical"
    />
    <div class="hint">
      <el-icon :size="12"><InfoFilled /></el-icon>
      <span>支持 Ctrl+Enter 快速保存</span>
    </div>

    <template #footer>
      <el-button @click="visible = false">取消</el-button>
      <el-button type="primary" :loading="saving" @click="save">保存标注</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted } from 'vue'
import { InfoFilled } from '@element-plus/icons-vue'
import { useDatasetStore } from '@/stores/dataset'
import type { ImageGroup } from '@/stores/dataset'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  modelValue: boolean
  group: ImageGroup | null
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', val: boolean): void
  (e: 'saved'): void
}>()

const visible = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v)
})

const datasetStore = useDatasetStore()
const captionText = ref('')
const saving = ref(false)
const textareaRef = ref<any>(null)

const imageName = computed(() => props.group?.id || '')

function onOpen() {
  captionText.value = props.group?.caption || ''
  nextTick(() => {
    textareaRef.value?.focus()
  })
}

async function save() {
  if (!props.group?.target) return
  saving.value = true
  try {
    await datasetStore.saveCaption(props.group.target.path, captionText.value)
    // Update group caption locally
    if (props.group) {
      props.group.caption = captionText.value
      if (props.group.target) {
        props.group.target.caption = captionText.value
      }
    }
    ElMessage.success('标注已保存')
    emit('saved')
    visible.value = false
  } catch (err: any) {
    ElMessage.error('保存失败: ' + (err.message || err))
  } finally {
    saving.value = false
  }
}

// Ctrl+Enter keyboard shortcut
function onKeydown(e: KeyboardEvent) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault()
    save()
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
})
</script>

<style lang="scss" scoped>
.caption-edit-dialog {
  :deep(.el-dialog__body) {
    padding: 16px 20px;
  }
  :deep(.el-textarea__inner) {
    font-size: 13px;
    line-height: 1.6;
  }
}

.hint {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 8px;
  font-size: 11px;
  color: rgba(255,255,255,0.3);
}
</style>
