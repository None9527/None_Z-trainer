<template>
  <div class="zimage-training-params">
    <!-- Timestep Sampling Mode Selector -->
    <div class="mode-selector">
      <span class="mode-label">时间步采样模式</span>
      <el-radio-group v-model="config.timestep.mode" size="default">
        <el-radio-button value="uniform">
          <span class="mode-btn-content">
            <span class="mode-icon">📊</span>
            <span>Uniform</span>
          </span>
        </el-radio-button>
        <el-radio-button value="logit_normal">
          <span class="mode-btn-content">
            <span class="mode-icon">🔔</span>
            <span>LogNorm</span>
          </span>
        </el-radio-button>
        <el-radio-button value="acrf">
          <span class="mode-btn-content">
            <span class="mode-icon">⚓</span>
            <span>ACRF</span>
          </span>
        </el-radio-button>
      </el-radio-group>
    </div>

    <!-- Mode Description -->
    <div class="mode-description">
      <template v-if="config.timestep.mode === 'uniform'">
        均匀采样所有时间步，通过 Shift 偏移噪声分布。最通用的训练方式。
      </template>
      <template v-else-if="config.timestep.mode === 'logit_normal'">
        集中采样中间时间步（钟形分布），通过峰值偏移控制重点训练区域。SD3 论文推荐方式。
      </template>
      <template v-else>
        仅在固定锚点时间步上训练（蒸馏/Turbo 模式）。Shift 从模型配置自动读取。
      </template>
    </div>

    <!-- ═══════════ Uniform Params ═══════════ -->
    <template v-if="config.timestep.mode === 'uniform'">
      <div class="subsection-label">Shift 参数 (Noise Schedule)</div>

      <div class="control-row">
        <span class="label">
          启用 Dynamic Shift
          <el-tooltip content="基于分辨率自动计算 Shift 值，大图用大 shift、小图用小 shift" placement="top">
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <el-switch v-model="config.timestep.use_dynamic_shift" />
      </div>

      <template v-if="config.timestep.use_dynamic_shift">
        <div class="control-row">
          <span class="label">
            Base Shift
            <el-tooltip content="动态 Shift 基准值，低分辨率时使用" placement="top">
              <el-icon class="help-icon"><QuestionFilled /></el-icon>
            </el-tooltip>
          </span>
          <el-slider v-model="config.timestep.base_shift" :min="0.1" :max="2.0" :step="0.1" :show-tooltip="false" class="slider-flex" />
          <el-input-number v-model="config.timestep.base_shift" :min="0.1" :max="2.0" :step="0.1" controls-position="right" class="input-fixed" />
        </div>
        <div class="control-row">
          <span class="label">
            Max Shift
            <el-tooltip content="动态 Shift 最大值，高分辨率时使用" placement="top">
              <el-icon class="help-icon"><QuestionFilled /></el-icon>
            </el-tooltip>
          </span>
          <el-slider v-model="config.timestep.max_shift" :min="0.5" :max="3.0" :step="0.05" :show-tooltip="false" class="slider-flex" />
          <el-input-number v-model="config.timestep.max_shift" :min="0.5" :max="3.0" :step="0.05" controls-position="right" class="input-fixed" />
        </div>
      </template>

      <div class="control-row" v-else>
        <span class="label">
          Fixed Shift
          <el-tooltip content="固定时间步偏移。>1 偏向高噪声(构图)，<1 偏向低噪声(细节)" placement="top">
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <el-slider v-model="config.timestep.shift" :min="0.5" :max="5" :step="0.1" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="config.timestep.shift" :min="0.5" :max="5" :step="0.1" controls-position="right" class="input-fixed" />
      </div>
    </template>

    <!-- ═══════════ LogNorm Params ═══════════ -->
    <template v-if="config.timestep.mode === 'logit_normal'">
      <div class="subsection-label">分布参数</div>

      <div class="control-row">
        <span class="label">
          峰值偏移 (mean)
          <el-tooltip placement="top">
            <template #content>
              <div style="max-width: 280px; line-height: 1.6">
                控制训练集中在哪个噪声阶段，<b>效果等同于 Shift</b><br>
                <span style="opacity: 0.7">mean = 0 → 均衡训练（默认）</span><br>
                <span style="opacity: 0.7">mean > 0 → 偏向高噪声（构图/布局）</span><br>
                <span style="opacity: 0.7">mean < 0 → 偏向低噪声（细节/纹理）</span><br>
                推荐值: 0.0 ~ 1.0
              </div>
            </template>
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <el-slider v-model="config.timestep.logit_mean" :min="-2" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="config.timestep.logit_mean" :min="-2" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
      </div>
      <div class="param-hint">等同于 Shift 偏移，无需额外设置 Shift</div>

      <div class="control-row">
        <span class="label">
          集中程度 (std)
          <el-tooltip placement="top">
            <template #content>
              <div style="max-width: 280px; line-height: 1.6">
                值越小，训练越集中在 mean 附近<br>
                值越大，越接近均匀分布<br>
                推荐值: 0.5 ~ 1.5
              </div>
            </template>
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <el-slider v-model="config.timestep.logit_std" :min="0.1" :max="3.0" :step="0.1" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="config.timestep.logit_std" :min="0.1" :max="3.0" :step="0.1" controls-position="right" class="input-fixed" />
      </div>
    </template>

    <!-- ═══════════ ACRF Params ═══════════ -->
    <template v-if="config.timestep.mode === 'acrf'">
      <div class="subsection-label">锚点训练参数</div>

      <div class="control-row">
        <span class="label">
          推理步数
          <el-tooltip content="生成时用多少步推理，锚点就设多少个。必须和推理时的步数一致" placement="top">
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <el-slider v-model="config.timestep.acrf_steps" :min="1" :max="20" :step="1" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="config.timestep.acrf_steps" :min="1" :max="20" :step="1" controls-position="right" class="input-fixed" />
      </div>

      <div class="control-row">
        <span class="label">
          Jitter Scale
          <el-tooltip content="时间步抖动幅度，增加锚点附近的采样多样性" placement="top">
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <el-slider v-model="config.timestep.jitter_scale" :min="0" :max="0.1" :step="0.005" :show-tooltip="false" class="slider-flex" />
        <el-input-number v-model="config.timestep.jitter_scale" :min="0" :max="0.1" :step="0.005" controls-position="right" class="input-fixed" :precision="3" />
      </div>

      <div class="control-row readonly-row">
        <span class="label">
          Shift (自动)
          <el-tooltip content="ACRF 模式下 Shift 决定锚点位置，自动从模型配置读取，训练与推理必须一致" placement="top">
            <el-icon class="help-icon"><QuestionFilled /></el-icon>
          </el-tooltip>
        </span>
        <span class="readonly-value">{{ config.timestep.shift }}</span>
        <el-tag size="small" type="info">只读</el-tag>
      </div>
    </template>

    <!-- ═══════════ Shared: Latent Jitter ═══════════ -->
    <div class="subsection-label" style="margin-top: 20px">共享参数</div>
    <div class="control-row">
      <span class="label">
        Latent Jitter Scale
        <el-tooltip content="在 x_t 上添加空间抠动，垂直于流线，改变构图。推荐 0.03~0.05，0=关闭" placement="top">
          <el-icon class="help-icon"><QuestionFilled /></el-icon>
        </el-tooltip>
      </span>
      <el-slider v-model="config.timestep.latent_jitter_scale" :min="0" :max="0.1" :step="0.01" :show-tooltip="false" class="slider-flex" />
      <el-input-number v-model="config.timestep.latent_jitter_scale" :min="0" :max="0.1" :step="0.01" controls-position="right" class="input-fixed" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { QuestionFilled } from '@element-plus/icons-vue'

const config = defineModel<any>({ required: true })
</script>

<style scoped>
/* Mode selector */
.mode-selector {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 12px;
}
.mode-label {
  font-size: 13px; font-weight: 600; color: var(--el-text-color-primary);
  white-space: nowrap;
}
.mode-btn-content {
  display: flex; align-items: center; gap: 6px;
}
.mode-icon { font-size: 16px; }

.mode-description {
  font-size: 12px; color: var(--el-text-color-secondary);
  padding: 10px 14px; margin-bottom: 16px;
  background: var(--el-fill-color-lighter);
  border-radius: 6px; border-left: 3px solid var(--el-color-primary-light-3);
  line-height: 1.6;
}

/* Subsection labels */
.subsection-label {
  font-size: 11px; font-weight: 700; color: var(--el-text-color-secondary);
  margin: 20px 0 12px 0; text-transform: uppercase; letter-spacing: 1px;
  padding-top: 20px; border-top: 1px solid var(--el-border-color-lighter);
}
.subsection-label:first-child { margin-top: 0; padding-top: 0; border-top: none; }

/* Control rows */
.control-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.control-row:last-child { margin-bottom: 0; }
.control-row .label {
  font-size: 12px; color: var(--el-text-color-regular);
  width: 160px; flex-shrink: 0; display: flex; align-items: center; gap: 4px;
}

.help-icon { color: var(--el-color-primary-light-3); cursor: help; font-size: 14px; opacity: 0.8; }
.help-icon:hover { color: var(--el-color-primary); opacity: 1; }
.slider-flex { flex: 1; margin-right: 8px; }
.input-fixed { width: 100px !important; }

/* Param hint */
.param-hint {
  font-size: 11px; color: var(--el-text-color-placeholder);
  margin: -6px 0 12px 172px; font-style: italic;
}

/* Readonly row */
.readonly-row .readonly-value {
  font-size: 14px; font-weight: 600; color: var(--el-text-color-primary);
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
</style>
