<template>
  <el-config-provider :locale="zhCn">
    <div class="app-container">
      <!-- 背景动画 -->
      <div class="bg-animation">
        <div class="grid-overlay"></div>
        <div class="glow-orb orb-1"></div>
        <div class="glow-orb orb-2"></div>
        <div class="glow-orb orb-3"></div>
      </div>

      <div class="layout">
        <!-- Sidebar — always expanded -->
        <nav class="sidebar">
          <!-- Logo -->
          <router-link to="/home" class="sidebar-logo">
            <div class="logo-mark">N</div>
            <div class="logo-text">
              <span class="logo-title">None Trainer</span>
              
            </div>
          </router-link>

          <!-- Workflow nav -->
          <div class="nav-section">
            <div class="nav-group-label">工作流</div>
            <router-link
              v-for="item in workflowItems"
              :key="item.path"
              :to="item.path"
              class="nav-item"
              :class="{ active: isActive(item.path) }"
            >
              <el-icon class="nav-icon"><component :is="item.icon" /></el-icon>
              <span class="nav-label">{{ item.label }}</span>
            </router-link>
          </div>

          <!-- Tools nav -->
          <div class="nav-section">
            <div class="nav-group-label">工具</div>
            <router-link
              v-for="item in toolItems"
              :key="item.path"
              :to="item.path"
              class="nav-item"
              :class="{ active: isActive(item.path) }"
            >
              <el-icon class="nav-icon"><component :is="item.icon" /></el-icon>
              <span class="nav-label">{{ item.label }}</span>
            </router-link>
          </div>

          <div class="nav-spacer"></div>

          <!-- GPU Status — always visible -->
          <div class="gpu-status">
            <div class="gpu-header">
              <el-icon><Cpu /></el-icon>
              <span class="gpu-name">{{ gpuInfo.name || 'GPU' }}</span>
            </div>
            <div class="gpu-bar-wrap">
              <div class="gpu-bar">
                <div
                  class="gpu-fill"
                  :class="gpuLevel"
                  :style="{ width: (gpuInfo.memoryPercent || 0) + '%' }"
                ></div>
              </div>
              <span class="gpu-text">{{ gpuInfo.memoryUsed || '0' }} / {{ gpuInfo.memoryTotal || '0' }} GB</span>
            </div>
          </div>
        </nav>

        <!-- Main content -->
        <main class="main-content">
          <router-view v-slot="{ Component }">
            <transition name="fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </main>
      </div>
    </div>
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, markRaw } from 'vue'
import { useRoute } from 'vue-router'
import { useDark } from '@vueuse/core'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import {
  Picture, Setting, VideoPlay, MagicStick,
  DataLine, Files, Cpu
} from '@element-plus/icons-vue'

const route = useRoute()
const isDark = useDark()
const systemStore = useSystemStore()
const wsStore = useWebSocketStore()

const gpuInfo = computed(() => systemStore.gpuInfo)

const gpuLevel = computed(() => {
  const pct = gpuInfo.value.memoryPercent || 0
  if (pct > 85) return 'critical'
  if (pct > 60) return 'warm'
  return 'normal'
})

// Workflow: the core pipeline
const workflowItems = [
  { path: '/dataset', label: '数据集', icon: markRaw(Picture) },
  { path: '/config', label: '训练配置', icon: markRaw(Setting) },
  { path: '/training', label: '开始训练', icon: markRaw(VideoPlay) },
  { path: '/generation', label: '图片生成', icon: markRaw(MagicStick) },
]

// Tools: supporting functions
const toolItems = [
  { path: '/monitor', label: '训练监控', icon: markRaw(DataLine) },
  { path: '/loras', label: '模型管理', icon: markRaw(Files) },
]

function isActive(path: string) {
  return route.path === path
}

onMounted(() => {
  wsStore.connect()
  wsStore.startHeartbeat()
})

onUnmounted(() => {
  wsStore.stopHeartbeat()
  wsStore.disconnect()
})
</script>

<style lang="scss" scoped>
.app-container {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-dark);
  color: var(--text-primary);
}

// 背景动画
.bg-animation {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
}

.grid-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  background-image:
    linear-gradient(rgba(232, 196, 124, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(232, 196, 124, 0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
  0% { transform: translateY(0); }
  100% { transform: translateY(50px); }
}

.glow-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(40px);
  mix-blend-mode: screen;
  animation: float 20s ease-in-out infinite;
}

.orb-1 {
  width: 600px;
  height: 600px;
  background: radial-gradient(circle,
    rgba(200, 170, 90, 0.40) 0%,
    rgba(200, 170, 90, 0.36) 8%,
    rgba(200, 170, 90, 0.30) 16%,
    rgba(200, 170, 90, 0.24) 24%,
    rgba(200, 170, 90, 0.18) 32%,
    rgba(200, 170, 90, 0.13) 40%,
    rgba(200, 170, 90, 0.08) 50%,
    rgba(200, 170, 90, 0.04) 60%,
    rgba(200, 170, 90, 0.01) 75%,
    transparent 100%);
  top: -150px;
  left: -150px;
}

.orb-2 {
  width: 500px;
  height: 500px;
  background: radial-gradient(circle,
    rgba(70, 130, 210, 0.35) 0%,
    rgba(70, 130, 210, 0.31) 8%,
    rgba(70, 130, 210, 0.26) 16%,
    rgba(70, 130, 210, 0.20) 24%,
    rgba(70, 130, 210, 0.15) 32%,
    rgba(70, 130, 210, 0.10) 40%,
    rgba(70, 130, 210, 0.06) 50%,
    rgba(70, 130, 210, 0.03) 60%,
    rgba(70, 130, 210, 0.01) 75%,
    transparent 100%);
  top: 40%;
  right: -120px;
  animation-delay: -7s;
}

.orb-3 {
  width: 550px;
  height: 550px;
  background: radial-gradient(circle,
    rgba(70, 180, 130, 0.30) 0%,
    rgba(70, 180, 130, 0.26) 8%,
    rgba(70, 180, 130, 0.22) 16%,
    rgba(70, 180, 130, 0.17) 24%,
    rgba(70, 180, 130, 0.12) 32%,
    rgba(70, 180, 130, 0.08) 40%,
    rgba(70, 180, 130, 0.05) 50%,
    rgba(70, 180, 130, 0.02) 60%,
    rgba(70, 180, 130, 0.005) 75%,
    transparent 100%);
  bottom: -150px;
  left: 25%;
  animation-delay: -14s;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(30px, 30px) scale(1.1); }
  50% { transform: translate(-20px, 50px) scale(0.9); }
  75% { transform: translate(40px, -20px) scale(1.05); }
}

.layout {
  display: flex;
  height: 100%;
  position: relative;
  z-index: 1;
}

// Sidebar — restored old premium design
.sidebar {
  width: 240px;
  height: 100%;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 0;
  flex-shrink: 0;
  overflow-y: auto;
  overflow-x: hidden;
  backdrop-filter: blur(20px);

  &::-webkit-scrollbar { width: 0; }
}

// Logo
.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  text-decoration: none;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}

.logo-mark {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: linear-gradient(135deg, var(--primary), var(--secondary, var(--primary)));
  color: var(--bg-dark);
  font-family: 'Orbitron', sans-serif;
  font-size: 24px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 0 20px rgba(232, 196, 124, 0.3);
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.logo-title {
  font-family: 'Orbitron', sans-serif;
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--primary), var(--secondary, #88c0ff));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 0.1em;
  line-height: 1.2;
}

.logo-sub {
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.3em;
}

// Nav sections
.nav-section {
  padding: 4px 8px;
}

.nav-group-label {
  padding: 12px 12px 6px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 14px;
  border-radius: 8px;
  text-decoration: none;
  color: var(--text-secondary);
  transition: all var(--transition-fast);
  cursor: pointer;
  margin: 4px 0;
  font-size: 14px;

  &:hover {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
  }

  &.active {
    background: linear-gradient(135deg, var(--primary), var(--secondary, var(--primary)));
    color: var(--bg-dark);

    .nav-icon {
      color: var(--bg-dark);
    }
  }
}

.nav-icon {
  font-size: 18px;
  flex-shrink: 0;
}

.nav-label {
  font-size: 14px;
  font-weight: 500;
}

.nav-spacer {
  flex: 1;
}

// GPU Status
.gpu-status {
  padding: 16px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}

.gpu-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-muted);

  .el-icon {
    color: var(--primary);
    font-size: 14px;
  }
}

.gpu-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gpu-bar-wrap {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.gpu-bar {
  height: 4px;
  background: var(--bg-input);
  border-radius: 2px;
  overflow: hidden;
}

.gpu-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 1s ease;

  &.normal { background: var(--primary); }
  &.warm { background: var(--warning); }
  &.critical { background: var(--error); }
}

.gpu-text {
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}

// Main
.main-content {
  flex: 1;
  overflow-y: auto;
  min-width: 0;
  padding: 28px 32px;

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
  }
}

// Transitions
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>

<!-- Global tooltip styles -->
<style>
.el-popper {
  background: #1e1e2e !important;
  color: #ffffff !important;
  border: 1px solid #3a3a4a !important;
}
.el-popper .el-popper__arrow::before {
  background: #1e1e2e !important;
  border-color: #3a3a4a !important;
}
.el-tooltip__popper {
  background: #1e1e2e !important;
  color: #ffffff !important;
  border: 1px solid #3a3a4a !important;
}
.el-tooltip__popper .el-popper__arrow::before {
  background: #1e1e2e !important;
  border-color: #3a3a4a !important;
}
</style>
