<template>
  <div class="template-cache-config">
    <!-- Add model-specific cache configuration here -->
    <el-form-item label="选择缓存类型">
      <el-checkbox-group v-model="cacheTypes">
        <el-checkbox label="latent">Latent 缓存</el-checkbox>
        <el-checkbox label="text">Text 缓存</el-checkbox>
      </el-checkbox-group>
    </el-form-item>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  datasetPath: string
}>()

const cacheTypes = defineModel<string[]>('cacheTypes', { default: () => ['latent', 'text'] })
</script>
