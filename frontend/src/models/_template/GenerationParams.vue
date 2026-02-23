<template>
  <div class="template-generation-params">
    <!-- Add model-specific generation parameters here -->
    <el-alert type="info" :closable="false" show-icon>
      Template model — no model-specific parameters
    </el-alert>
  </div>
</template>

<script setup lang="ts">
const params = defineModel<any>('modelValue', { required: true })
</script>
