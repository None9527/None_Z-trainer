<template>
  <div class="template-training-params">
    <!-- Add model-specific training parameters here -->
    <el-alert type="info" :closable="false" show-icon>
      Template model — no model-specific parameters
    </el-alert>
  </div>
</template>

<script setup lang="ts">
const config = defineModel<any>('modelValue', { required: true })
</script>
