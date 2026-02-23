import { createRouter, createWebHistory } from 'vue-router'
import axios from 'axios'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/home'
    },
    {
      path: '/home',
      name: 'Home',
      component: () => import('@/views/Welcome.vue'),
      meta: { title: 'Home' }
    },
    {
      path: '/welcome',
      name: 'Welcome',
      component: () => import('@/views/Welcome.vue'),
      meta: { title: 'Welcome (Legacy)' }
    },
    {
      path: '/dashboard',
      redirect: '/dataset'
    },
    {
      path: '/dataset',
      name: 'Dataset',
      component: () => import('@/views/Dataset.vue'),
      meta: { title: '数据集管理', requiresModel: true }
    },
    {
      path: '/config',
      name: 'Config',
      component: () => import('@/views/TrainingConfig.vue'),
      meta: { title: '训练配置' }
    },
    {
      path: '/training',
      name: 'Training',
      component: () => import('@/views/Training.vue'),
      meta: { title: '开始训练', requiresModel: true }
    },
    {
      path: '/generation',
      name: 'Generation',
      component: () => import('@/views/Generation.vue'),
      meta: { title: '图片生成', requiresModel: true }
    },
    {
      path: '/monitor',
      name: 'Monitor',
      component: () => import('@/views/Monitor.vue'),
      meta: { title: '训练监控', requiresModel: true }
    },
    {
      path: '/loras',
      name: 'LoraManager',
      component: () => import('@/views/LoraManager.vue'),
      meta: { title: 'LoRA 管理' }
    }
  ]
})

router.beforeEach(async (to, from, next) => {
  // Removed mandatory check to allow sidebar navigation
  next()
})

export default router
