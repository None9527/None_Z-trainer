<template>
  <div class="training-config-page">
    <!-- 顶部配置管理栏 -->
    <div class="config-header glass-card">
      <div class="header-left">
        <h1><el-icon><Setting /></el-icon> 训练配置</h1>
        <div class="config-toolbar">
          <el-select v-model="tc.currentConfigName.value" placeholder="选择配置..." @change="tc.loadSavedConfig()" style="width: 200px">
            <el-option label="默认配置" value="default" />
            <el-option v-for="cfg in tc.savedConfigs.value.filter((c: any) => c.name !== 'default')" :key="cfg.name" :label="cfg.name" :value="cfg.name" />
          </el-select>
          <el-button @click="tc.showNewConfigDialog.value = true" :icon="Plus">新建</el-button>
          <el-button @click="tc.showSaveAsDialog.value = true" :icon="Document">另存为</el-button>
          <el-button type="primary" @click="tc.saveCurrentConfig(activeNames)" :loading="tc.saving.value" :icon="Check">发送训练器</el-button>
          <el-button type="danger" @click="tc.deleteCurrentConfig()" :disabled="tc.currentConfigName.value === 'default'" :icon="Delete">删除</el-button>
        </div>
      </div>
    </div>

    <!-- 新建配置对话框 -->
    <el-dialog v-model="tc.showNewConfigDialog.value" title="新建配置" width="400px">
      <el-form label-width="80px">
        <el-form-item label="配置名称">
          <el-input v-model="tc.newConfigName.value" placeholder="输入配置名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="tc.showNewConfigDialog.value = false">取消</el-button>
        <el-button type="primary" @click="tc.createNewConfig()">创建</el-button>
      </template>
    </el-dialog>

    <!-- 另存为对话框 -->
    <el-dialog v-model="tc.showSaveAsDialog.value" title="另存为" width="400px">
      <el-form label-width="80px">
        <el-form-item label="配置名称">
          <el-input v-model="tc.saveAsName.value" placeholder="输入新配置名称" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="tc.showSaveAsDialog.value = false">取消</el-button>
        <el-button type="primary" @click="tc.saveAsNewConfig()">保存</el-button>
      </template>
    </el-dialog>

    <!-- 配置内容 -->
    <el-card class="config-content-card glass-card" v-loading="tc.loading.value">
      <el-collapse v-model="activeNames" class="config-collapse">

        <!-- 1. 训练类型选择 -->
        <el-collapse-item name="model">
          <template #title>
            <div class="collapse-title">
              <el-icon><Cpu /></el-icon>
              <span>训练类型</span>
              <el-tag :type="tc.trainingTypeTagType.value" size="small" style="margin-left: 10px">{{ tc.trainingTypeDisplayName.value }}</el-tag>
            </div>
          </template>
          <div class="collapse-content">
            <div class="model-type-cards">
              <div 
                v-for="type in trainingTypes" 
                :key="type.value"
                :class="['model-card', { active: tc.config.value.training_type === type.value, disabled: type.disabled }]"
                @click="!type.disabled && (tc.config.value.training_type = type.value)"
              >
                <div class="model-icon">{{ type.icon }}</div>
                <div class="model-info">
                  <div class="model-name">{{ type.label }}</div>
                  <div class="model-desc">{{ type.description }}</div>
                </div>
                <el-tag v-if="type.tag" :type="type.tagType" size="small">{{ type.tag }}</el-tag>
              </div>
            </div>
            
            <!-- 条件模式 -->
            <template v-if="tc.config.value.training_type !== 'controlnet'">
              <div class="subsection-label" style="margin-top: 20px">条件模式</div>
              <div class="model-type-cards training-mode-cards">
                <div 
                  v-for="mode in conditionModes" 
                  :key="mode.value"
                  :class="['model-card', { active: tc.config.value.condition_mode === mode.value, disabled: mode.disabled }]"
                  @click="!mode.disabled && (tc.config.value.condition_mode = mode.value)"
                >
                  <div class="model-icon">{{ mode.icon }}</div>
                  <div class="model-info">
                    <div class="model-name">{{ mode.label }}</div>
                    <div class="model-desc">{{ mode.description }}</div>
                  </div>
                  <el-tag v-if="mode.tag" :type="mode.tagType" size="small">{{ mode.tag }}</el-tag>
                </div>
              </div>
            </template>
            
            <el-alert v-if="tc.config.value.training_type === 'finetune'" type="warning" :closable="false" show-icon style="margin-top: 16px">
              全量微调需要 40GB+ 显存，请确认您的硬件支持
            </el-alert>
            
            <!-- ControlNet 专属配置 -->
            <template v-if="tc.config.value.training_type === 'controlnet'">
              <div class="subsection-label" style="margin-top: 20px">ControlNet 配置</div>
              <div class="control-row">
                <span class="label">
                  训练模式
                  <el-tooltip content="创建新模型从头训练，或加载已有权重继续训练" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-radio-group v-model="tc.config.value.controlnet.resume_training">
                  <el-radio-button :label="false">创建新模型</el-radio-button>
                  <el-radio-button :label="true">继续训练</el-radio-button>
                </el-radio-group>
              </div>
              <template v-if="tc.config.value.controlnet.resume_training">
                <div class="form-row-full">
                  <label>选择 ControlNet 权重</label>
                  <el-select v-model="tc.config.value.controlnet.controlnet_path" placeholder="选择 ControlNet 权重..." filterable clearable style="width: 100%">
                    <el-option v-for="cn in tc.controlnetList.value" :key="cn.path" :label="cn.name" :value="cn.path">
                      <span style="float: left">{{ cn.name }}</span>
                      <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">{{ (cn.size / 1024 / 1024).toFixed(1) }} MB</span>
                    </el-option>
                  </el-select>
                </div>
              </template>
              <div class="control-row">
                <span class="label">
                  条件强度
                  <el-tooltip content="ControlNet 条件的影响程度 (0-1)" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="tc.config.value.controlnet.conditioning_scale" :min="0" :max="1" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.controlnet.conditioning_scale" :min="0" :max="1" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
              <el-alert type="info" :closable="false" show-icon style="margin-top: 16px">
                ControlNet 训练时 Transformer 自动冻结。需准备配对数据：控制图像 (source/) 和目标图像 (target/)
              </el-alert>
            </template>
          </div>
        </el-collapse-item>

        <!-- 2. 模型专属参数 (Registry 动态注入) -->
        <el-collapse-item name="acrf" v-if="tc.config.value.training_type !== 'controlnet' && currentModel">
          <template #title>
            <div class="collapse-title">
              <el-icon><DataAnalysis /></el-icon>
              <span>模型参数 ({{ currentModel.name }})</span>
            </div>
          </template>
          <div class="collapse-content">
            <component :is="currentModel.trainingParamsComponent" v-model="tc.config.value" />
          </div>
        </el-collapse-item>

        <!-- 3. LoRA 配置 -->
        <el-collapse-item name="lora" v-if="tc.config.value.training_type === 'lora'">
          <template #title>
            <div class="collapse-title">
              <el-icon><Grid /></el-icon>
              <span>LoRA 配置</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="control-row">
              <span class="label">
                继续训练已有 LoRA
                <el-tooltip content="加载已有 LoRA 继续训练，Rank/层设置将从 LoRA 文件自动读取" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="tc.config.value.lora.resume_training" />
            </div>
            <template v-if="tc.config.value.lora.resume_training">
              <div class="form-row-full">
                <label>选择 LoRA 文件</label>
                <el-select v-model="tc.config.value.lora.resume_lora_path" placeholder="选择 LoRA 文件..." filterable clearable style="width: 100%">
                  <el-option v-for="lora in tc.loraList.value" :key="lora.path" :label="lora.name" :value="lora.path">
                    <span style="float: left">{{ lora.name }}</span>
                    <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">{{ (lora.size_bytes / 1024 / 1024).toFixed(1) }} MB</span>
                  </el-option>
                </el-select>
              </div>
              <el-alert v-if="tc.config.value.lora.resume_lora_path" type="info" :closable="false" show-icon style="margin-top: 12px">
                Rank 和层设置将从 LoRA 文件自动读取
              </el-alert>
            </template>
            <template v-else>
              <div class="control-row">
                <span class="label">
                  Network Dim (Rank)
                  <el-tooltip content="LoRA 矩阵的秩，越大学习能力越强但文件越大，推荐 4-32" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="tc.config.value.network.dim" :min="4" :max="512" :step="4" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.network.dim" :min="4" :max="512" :step="4" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">
                  Network Alpha
                  <el-tooltip content="缩放因子，通常设为 Dim 的一半，影响学习率效果" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-slider v-model="tc.config.value.network.alpha" :min="1" :max="512" :step="0.5" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.network.alpha" :min="1" :max="512" :step="0.5" controls-position="right" class="input-fixed" />
              </div>
              <div class="subsection-label">高级选项 (LoRA Targets)</div>
              <div class="control-row">
                <span class="label">
                  训练 AdaLN
                  <el-tooltip content="训练 AdaLN 调制层 (激进模式，可能导致过拟合)" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-switch v-model="tc.config.value.lora.train_adaln" />
              </div>
            </template>
          </div>
        </el-collapse-item>

        <!-- 4. 训练设置 -->
        <el-collapse-item name="training">
          <template #title>
            <div class="collapse-title">
              <el-icon><TrendCharts /></el-icon>
              <span>训练设置</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="subsection-label">输出设置 (OUTPUT)</div>
            <div class="form-row-full">
              <label>{{ tc.outputNameLabel.value }}</label>
              <el-input v-model="tc.config.value.training.output_name" :placeholder="tc.outputNamePlaceholder.value" />
            </div>
            
            <div class="subsection-label">训练控制 (TRAINING CONTROL)</div>
            <div class="control-row">
              <span class="label">
                训练轮数
                <el-tooltip content="完整遍历数据集的次数，一般 5-20 轮即可" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="tc.config.value.advanced.num_train_epochs" :min="1" :max="100" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.advanced.num_train_epochs" :min="1" :max="100" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                保存间隔
                <el-tooltip content="每隔几轮保存一次模型，便于挑选最佳效果" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="tc.config.value.advanced.save_every_n_epochs" :min="1" :max="10" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.advanced.save_every_n_epochs" :min="1" :max="10" controls-position="right" class="input-fixed" />
            </div>

            <div class="subsection-label">优化器 (OPTIMIZER)</div>
            <div class="form-row-full">
              <label>
                优化器类型
                <el-tooltip content="AdamW8bit 省显存，Adafactor 更省但可能不稳定，Prodigy 自适应LR" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-select v-model="tc.config.value.optimizer.type" style="width: 100%">
                <el-option label="AdamW" value="AdamW" />
                <el-option label="AdamW8bit (显存优化)" value="AdamW8bit" />
                <el-option label="AdamWFP8 (FP8精度更高)" value="AdamWFP8" />
                <el-option label="AdamWBF16 (BF16平衡)" value="AdamWBF16" />
                <el-option label="Adafactor" value="Adafactor" />
                <el-option label="Prodigy (自适应LR)" value="Prodigy" />
                <el-option label="Lion (显存低)" value="Lion" />
                <el-option label="Lion8bit (显存最低)" value="Lion8bit" />
              </el-select>
            </div>
            <div class="form-row-full" v-if="tc.config.value.optimizer.type === 'Adafactor'">
              <label>
                自适应学习率
                <el-tooltip content="启用后 Adafactor 自动调整学习率，无需手动设置" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-switch v-model="tc.config.value.optimizer.relative_step" />
            </div>
            <div class="form-row-full" v-if="tc.config.value.optimizer.type !== 'Prodigy' && !(tc.config.value.optimizer.type === 'Adafactor' && tc.config.value.optimizer.relative_step)">
              <label>
                学习率
                <el-tooltip content="模型学习的速度，推荐 1e-4" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-input v-model="tc.config.value.training.learning_rate_str" placeholder="1e-4" @blur="tc.parseLearningRate()">
                <template #append>
                  <el-tooltip content="支持科学计数法，如 1e-4, 5e-5" placement="top">
                    <el-icon><InfoFilled /></el-icon>
                  </el-tooltip>
                </template>
              </el-input>
            </div>
            <div class="form-row-full" v-else-if="tc.config.value.optimizer.type === 'Prodigy' || (tc.config.value.optimizer.type === 'Adafactor' && tc.config.value.optimizer.relative_step)">
              <el-alert type="info" :closable="false" show-icon>
                {{ tc.config.value.optimizer.type }} 优化器自动调整学习率，无需手动设置
              </el-alert>
            </div>

            <div class="subsection-label">学习率调度器 (LR SCHEDULER)</div>
            <div class="form-row-full">
              <label>
                调度器类型
                <el-tooltip content="控制学习率变化方式，constant 最简单，cosine 后期更稳定" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-select v-model="tc.config.value.training.lr_scheduler" style="width: 100%">
                <el-option label="constant (固定) ⭐推荐" value="constant" />
                <el-option label="one_cycle (单周期) 🚀FP8优化" value="one_cycle" />
                <el-option label="linear (线性衰减)" value="linear" />
                <el-option label="cosine (余弦退火)" value="cosine" />
                <el-option label="cosine_with_restarts (余弦重启)" value="cosine_with_restarts" />
                <el-option label="constant_with_warmup (带预热)" value="constant_with_warmup" />
              </el-select>
            </div>
            <div class="control-row" v-if="tc.config.value.training.lr_scheduler !== 'constant' && tc.config.value.training.lr_scheduler !== 'one_cycle'">
              <span class="label">Warmup Steps</span>
              <el-slider v-model="tc.config.value.training.lr_warmup_steps" :min="0" :max="500" :step="5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.training.lr_warmup_steps" :min="0" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row" v-if="tc.config.value.training.lr_scheduler === 'cosine_with_restarts'">
              <span class="label">Num Cycles</span>
              <el-slider v-model="tc.config.value.training.lr_num_cycles" :min="1" :max="5" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.training.lr_num_cycles" :min="1" :max="5" controls-position="right" class="input-fixed" />
            </div>
            
            <!-- OneCycleLR -->
            <template v-if="tc.config.value.training.lr_scheduler === 'one_cycle'">
              <div class="control-row">
                <span class="label">Warmup 比例 (pct_start)</span>
                <el-slider v-model="tc.config.value.training.lr_pct_start" :min="0.05" :max="0.5" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lr_pct_start" :min="0.05" :max="0.5" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">初始除数 (div_factor)</span>
                <el-slider v-model="tc.config.value.training.lr_div_factor" :min="5" :max="50" :step="5" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lr_div_factor" :min="5" :max="50" :step="5" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">最终除数 (final_div_factor)</span>
                <el-slider v-model="tc.config.value.training.lr_final_div_factor" :min="10" :max="1000" :step="10" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lr_final_div_factor" :min="10" :max="1000" :step="10" controls-position="right" class="input-fixed" />
              </div>
              <el-alert type="info" :closable="false" show-icon style="margin-top: 8px">
                OneCycleLR: lr 从 {{ (tc.config.value.training.learning_rate / tc.config.value.training.lr_div_factor).toExponential(1) }} 
                → 峰值 {{ tc.config.value.training.learning_rate_str }} 
                → 最终 {{ (tc.config.value.training.learning_rate / tc.config.value.training.lr_final_div_factor).toExponential(1) }}
              </el-alert>
            </template>

            <div class="subsection-label">梯度与内存 (GRADIENT & MEMORY)</div>
            <div class="control-row">
              <span class="label">梯度累积</span>
              <el-slider v-model="tc.config.value.advanced.gradient_accumulation_steps" :min="1" :max="16" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.advanced.gradient_accumulation_steps" :min="1" :max="16" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">梯度检查点</span>
              <el-switch v-model="tc.config.value.advanced.gradient_checkpointing" />
            </div>
            <div class="control-row">
              <span class="label">Blocks to Swap</span>
              <el-input-number v-model="tc.config.value.advanced.blocks_to_swap" :min="0" :max="20" controls-position="right" style="width: 150px" />
            </div>
            <div class="form-row-full">
              <label>混合精度</label>
              <el-select v-model="tc.config.value.advanced.mixed_precision" style="width: 100%">
                <el-option label="bf16 (推荐)" value="bf16" />
                <el-option label="fp16" value="fp16" />
                <el-option label="no (FP32)" value="no" />
              </el-select>
            </div>
            <div class="control-row">
              <span class="label">随机种子</span>
              <el-input-number v-model="tc.config.value.advanced.seed" :min="0" controls-position="right" style="width: 150px" />
            </div>
            
            <div class="subsection-label">GPU 配置 (MULTI-GPU)</div>
            <div class="control-row">
              <span class="label">GPU 数量</span>
              <el-select v-model="tc.config.value.advanced.num_gpus" style="width: 150px">
                <el-option label="1 (单卡)" :value="1" />
                <el-option label="2" :value="2" />
                <el-option label="3" :value="3" />
                <el-option label="4" :value="4" />
                <el-option label="8" :value="8" />
              </el-select>
            </div>
            <div class="control-row">
              <span class="label">GPU ID</span>
              <el-input v-model="tc.config.value.advanced.gpu_ids" placeholder="如: 0,1,2" style="width: 150px" />
            </div>
          </div>
        </el-collapse-item>

        <!-- 5. 数据集配置 -->
        <el-collapse-item name="dataset">
          <template #title>
            <div class="collapse-title">
              <el-icon><Files /></el-icon>
              <span>数据集配置</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="subsection-label">通用设置 (GENERAL)</div>
            <div class="control-row">
              <span class="label">批次大小</span>
              <el-slider v-model="tc.config.value.dataset.batch_size" :min="1" :max="16" :step="1" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.dataset.batch_size" :min="1" :max="16" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">打乱数据</span>
              <el-switch v-model="tc.config.value.dataset.shuffle" />
            </div>
            <div class="control-row">
              <span class="label">启用分桶</span>
              <el-switch v-model="tc.config.value.dataset.enable_bucket" />
            </div>

            <!-- CFG -->
            <template v-if="tc.config.value.training_type !== 'controlnet'">
              <div class="subsection-label">🎯 CFG 训练 (CFG TRAINING)</div>
              <div class="control-row">
                <span class="label">Drop Text 比例</span>
                <el-slider v-model="tc.config.value.dataset.drop_text_ratio" :min="0" :max="0.3" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.dataset.drop_text_ratio" :min="0" :max="0.3" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">启用 CFG 训练</span>
                <el-switch v-model="tc.config.value.acrf.cfg_training" />
              </div>
              <template v-if="tc.config.value.acrf.cfg_training">
                <div class="control-row">
                  <span class="label">CFG Scale</span>
                  <el-slider v-model="tc.config.value.acrf.cfg_scale" :min="1" :max="15" :step="0.5" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="tc.config.value.acrf.cfg_scale" :min="1" :max="15" :step="0.5" controls-position="right" class="input-fixed" />
                </div>
                <div class="control-row">
                  <span class="label">CFG 训练比例</span>
                  <el-slider v-model="tc.config.value.acrf.cfg_training_ratio" :min="0.1" :max="1.0" :step="0.1" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="tc.config.value.acrf.cfg_training_ratio" :min="0.1" :max="1.0" :step="0.1" controls-position="right" class="input-fixed" />
                </div>
              </template>
            </template>

            <!-- 数据集列表 -->
            <div class="subsection-label-with-action">
              <span>数据集列表 (DATASETS)</span>
              <div class="dataset-toolbar">
                <el-select v-model="tc.selectedDataset.value" placeholder="从数据集库选择..." clearable @change="tc.onDatasetSelect()" style="width: 280px">
                  <el-option v-for="ds in tc.cachedDatasets.value" :key="ds.path" :label="ds.name" :value="ds.path">
                    <span style="float: left">{{ ds.name }}</span>
                    <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">{{ ds.files }} 文件</span>
                  </el-option>
                </el-select>
                <el-button size="small" type="primary" @click="tc.addDataset()" :icon="Plus">手动添加</el-button>
              </div>
            </div>
            
            <div v-if="tc.config.value.dataset.datasets.length === 0" class="empty-datasets">
              <el-icon><FolderOpened /></el-icon>
              <p>暂无数据集，点击上方按钮添加</p>
            </div>

            <div v-for="(ds, idx) in tc.config.value.dataset.datasets" :key="idx" class="dataset-item">
              <div class="dataset-header">
                <span class="dataset-index">数据集 {{ idx + 1 }}</span>
                <el-button type="danger" size="small" @click="tc.removeDataset(idx)" :icon="Delete">删除</el-button>
              </div>
              <div class="form-row-full">
                <label>缓存目录路径</label>
                <el-input 
                  v-model="ds.cache_directory" 
                  placeholder="D:/datasets/xxx 或 /datasets/xxx" 
                  @blur="tc.validateDatasetPath(ds)"
                  :class="{'is-valid': ds._validated && ds._valid, 'is-invalid': ds._validated && !ds._valid}"
                >
                  <template #suffix v-if="ds._validating">
                    <el-icon class="is-loading"><Loading /></el-icon>
                  </template>
                  <template #suffix v-else-if="ds._validated">
                    <el-icon v-if="ds._valid" style="color: var(--el-color-success)"><Check /></el-icon>
                    <el-icon v-else style="color: var(--el-color-danger)"><Close /></el-icon>
                  </template>
                </el-input>
                <div v-if="ds._validated && ds._valid" style="font-size: 12px; color: var(--el-color-success); margin-top: 4px;">
                  ✓ {{ ds._imageCount || 0 }} 图片{{ ds._hasCached ? ', 已缓存' : '' }}
                </div>
                <div v-else-if="ds._validated && !ds._valid" style="font-size: 12px; color: var(--el-color-danger); margin-top: 4px;">
                  ✗ {{ ds._error || '路径无效' }}
                </div>
              </div>
              <div class="control-row">
                <span class="label">重复次数</span>
                <el-slider v-model="ds.num_repeats" :min="1" :max="100" :step="1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="ds.num_repeats" :min="1" :max="100" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">分辨率上限</span>
                <el-slider v-model="ds.resolution_limit" :min="256" :max="2048" :step="64" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="ds.resolution_limit" :min="256" :max="2048" :step="64" controls-position="right" class="input-fixed" />
              </div>
            </div>
            
            <!-- 正则数据集 -->
            <template v-if="tc.config.value.training_type !== 'controlnet'">
              <div class="subsection-label" style="margin-top: 20px">正则数据集 (Regularization)</div>
              <div class="control-row">
                <span class="label">启用正则数据集</span>
                <el-switch v-model="tc.config.value.reg_dataset.enabled" />
              </div>
              <template v-if="tc.config.value.reg_dataset.enabled">
                <div class="control-row">
                  <span class="label">混合比例</span>
                  <el-slider v-model="tc.config.value.reg_dataset.ratio" :min="0.1" :max="0.9" :step="0.1" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="tc.config.value.reg_dataset.ratio" :min="0.1" :max="0.9" :step="0.1" controls-position="right" class="input-fixed" :precision="1" />
                </div>
                <div class="control-row">
                  <span class="label">正则损失权重</span>
                  <el-slider v-model="tc.config.value.reg_dataset.weight" :min="0.1" :max="2.0" :step="0.1" :show-tooltip="false" class="slider-flex" />
                  <el-input-number v-model="tc.config.value.reg_dataset.weight" :min="0.1" :max="2.0" :step="0.1" controls-position="right" class="input-fixed" :precision="1" />
                </div>
                <div class="form-row-full">
                  <label>选择正则数据集</label>
                  <div class="dataset-toolbar">
                    <el-select v-model="tc.selectedRegDataset.value" placeholder="从数据集库选择..." clearable @change="tc.onRegDatasetSelect()" style="width: 280px">
                      <el-option v-for="ds in tc.cachedDatasets.value" :key="ds.path" :label="ds.name" :value="ds.path">
                        <span style="float: left">{{ ds.name }}</span>
                        <span style="float: right; color: var(--el-text-color-secondary); font-size: 12px">{{ ds.files }} 文件</span>
                      </el-option>
                    </el-select>
                    <el-button size="small" type="primary" @click="tc.addRegDataset()" :icon="Plus">添加</el-button>
                  </div>
                </div>
                <div v-if="tc.config.value.reg_dataset.datasets.length === 0" class="empty-datasets">
                  <el-icon><FolderOpened /></el-icon>
                  <p>暂无正则数据集</p>
                </div>
                <div v-for="(rds, ridx) in tc.config.value.reg_dataset.datasets" :key="ridx" class="dataset-item reg-dataset-item">
                  <div class="dataset-header">
                    <span class="dataset-index">正则数据集 {{ ridx + 1 }}</span>
                    <el-button type="danger" size="small" @click="tc.removeRegDataset(ridx)" :icon="Delete">删除</el-button>
                  </div>
                  <div class="form-row-full">
                    <label>缓存目录路径</label>
                    <el-input v-model="rds.cache_directory" placeholder="正则数据集缓存路径" />
                  </div>
                  <div class="control-row">
                    <span class="label">重复次数</span>
                    <el-slider v-model="rds.num_repeats" :min="1" :max="50" :step="1" :show-tooltip="false" class="slider-flex" />
                    <el-input-number v-model="rds.num_repeats" :min="1" :max="50" controls-position="right" class="input-fixed" />
                  </div>
                </div>
              </template>
            </template>
          </div>
        </el-collapse-item>

        <!-- 6. 高级选项 (仅 LoRA/Finetune 显示) -->
        <el-collapse-item name="advanced" v-if="tc.config.value.training_type !== 'controlnet'">
          <template #title>
            <div class="collapse-title">
              <el-icon><Tools /></el-icon>
              <span>高级选项</span>
            </div>
          </template>
          <div class="collapse-content">
            <div class="subsection-label">SNR 参数（公用）</div>
            <div class="control-row">
              <span class="label">
                SNR Gamma
                <el-tooltip content="Min-SNR 截断值，平衡不同时间步的 loss 贡献，0=禁用，推荐 5.0" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="tc.config.value.acrf.snr_gamma" :min="0" :max="10" :step="0.5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.acrf.snr_gamma" :min="0" :max="10" :step="0.5" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                SNR Floor
                <el-tooltip content="保底权重，确保高噪区（构图阶段）参与训练。10步模型关键参数，推荐 0.1" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="tc.config.value.acrf.snr_floor" :min="0" :max="0.5" :step="0.01" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.acrf.snr_floor" :min="0" :max="0.5" :step="0.01" controls-position="right" class="input-fixed" />
            </div>

            <!-- 曲率惩罚 -->
            <div class="subsection-label">🎯 曲率惩罚 (Curvature Penalty)</div>
            <div class="control-row">
              <span class="label">
                启用曲率惩罚
                <el-tooltip content="鼓励锚点间匀速直线运动，减少采样步数时的误差" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="tc.config.value.acrf.enable_curvature" />
            </div>
            <template v-if="tc.config.value.acrf.enable_curvature">
              <div class="control-row">
                <span class="label">曲率惩罚权重 (λ)</span>
                <el-slider v-model="tc.config.value.acrf.lambda_curvature" :min="0.01" :max="0.2" :step="0.01" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.acrf.lambda_curvature" :min="0.01" :max="0.2" :step="0.01" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">计算间隔 (N 步)</span>
                <el-slider v-model="tc.config.value.acrf.curvature_interval" :min="1" :max="50" :step="1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.acrf.curvature_interval" :min="1" :max="50" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">延迟启用 (Epoch)</span>
                <el-slider v-model="tc.config.value.acrf.curvature_start_epoch" :min="0" :max="10" :step="1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.acrf.curvature_start_epoch" :min="0" :max="10" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <!-- MSE -->
            <div class="subsection-label">📐 MSE 损失</div>
            <div class="control-row">
              <span class="label">启用 MSE</span>
              <el-switch :model-value="tc.config.value.training.lambda_mse > 0" @update:model-value="tc.config.value.training.lambda_mse = $event ? 1.0 : 0" />
            </div>
            <template v-if="tc.config.value.training.lambda_mse > 0">
              <div class="control-row">
                <span class="label">MSE 权重</span>
                <el-slider v-model="tc.config.value.training.lambda_mse" :min="0.1" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_mse" :min="0.1" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <!-- L1 + Cosine -->
            <div class="subsection-label">📏 L1 损失</div>
            <div class="control-row">
              <span class="label">
                启用 L1
                <el-tooltip content="L1 绝对误差 + Cosine 方向约束" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch :model-value="tc.config.value.training.lambda_l1 > 0" @update:model-value="val => { tc.config.value.training.lambda_l1 = val ? 1.0 : 0; if (!val) tc.config.value.training.lambda_cosine = 0; else if (tc.config.value.training.lambda_cosine === 0) tc.config.value.training.lambda_cosine = 0.1; }" />
            </div>
            <template v-if="tc.config.value.training.lambda_l1 > 0">
              <div class="control-row">
                <span class="label">L1 权重</span>
                <el-slider v-model="tc.config.value.training.lambda_l1" :min="0.1" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_l1" :min="0.1" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ Cosine 权重</span>
                <el-slider v-model="tc.config.value.training.lambda_cosine" :min="0" :max="1" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_cosine" :min="0" :max="1" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <!-- 频域增强 -->
            <div class="subsection-label">🔍 频域增强 (纹理+结构)</div>
            <div class="control-row">
              <span class="label">
                启用频域增强
                <el-tooltip content="分离高频(纹理)和低频(结构)分别监督" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="tc.config.value.training.enable_freq" />
            </div>
            <template v-if="tc.config.value.training.enable_freq">
              <div class="control-row">
                <span class="label">频域总权重 (λ_freq)</span>
                <el-slider v-model="tc.config.value.training.lambda_freq" :min="0.1" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_freq" :min="0.1" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 纹理锐化 (alpha_hf)</span>
                <el-slider v-model="tc.config.value.training.alpha_hf" :min="0" :max="2" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.alpha_hf" :min="0" :max="2" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 结构学习 (beta_lf)</span>
                <el-slider v-model="tc.config.value.training.beta_lf" :min="0" :max="1" :step="0.05" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.beta_lf" :min="0" :max="1" :step="0.05" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <!-- 风格学习 -->
            <div class="subsection-label">🎨 风格学习 (光影+色调)</div>
            <div class="control-row">
              <span class="label">
                启用风格学习
                <el-tooltip content="学习目标图的全局风格统计量（亮度分布、色彩偏好）" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="tc.config.value.training.enable_style" />
            </div>
            <template v-if="tc.config.value.training.enable_style">
              <div class="control-row">
                <span class="label">风格总权重 (λ_style)</span>
                <el-slider v-model="tc.config.value.training.lambda_style" :min="0.1" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_style" :min="0.1" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 光影学习 (λ_light)</span>
                <el-slider v-model="tc.config.value.training.lambda_light" :min="0" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_light" :min="0" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row" style="margin-left: 20px;">
                <span class="label">↳ 色调迁移 (λ_color)</span>
                <el-slider v-model="tc.config.value.training.lambda_color" :min="0" :max="1" :step="0.1" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_color" :min="0" :max="1" :step="0.1" controls-position="right" class="input-fixed" />
              </div>
            </template>

            <!-- DINOv3 感知 Loss -->
            <div class="subsection-label">🧠 DINOv3 感知损失 (语义一致性)</div>
            <div class="control-row">
              <span class="label">
                启用 DINOv3 Loss
                <el-tooltip content="使用 DINOv3 ViT 提取语义特征，约束生成图与目标图的语义一致性。需预缓存 DINOv3 embedding" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="tc.config.value.training.enable_dino" />
            </div>
            <template v-if="tc.config.value.training.enable_dino">
              <div class="control-row">
                <span class="label">DINOv3 权重 (λ_dino)</span>
                <el-slider v-model="tc.config.value.training.lambda_dino" :min="0.01" :max="1" :step="0.01" :show-tooltip="false" class="slider-flex" />
                <el-input-number v-model="tc.config.value.training.lambda_dino" :min="0.01" :max="1" :step="0.01" controls-position="right" class="input-fixed" />
              </div>
              <div class="control-row">
                <span class="label">
                  特征模式
                  <el-tooltip placement="top">
                    <template #content>
                      <div>patch: 逐区域对比（精确还原）</div>
                      <div>cls: 全局语义对比（风格/美学）</div>
                      <div>both: 两者结合</div>
                    </template>
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </span>
                <el-select v-model="tc.config.value.training.dino_feature_mode" style="width: 180px">
                  <el-option label="Patch (逐区域)" value="patch" />
                  <el-option label="CLS (全局美学)" value="cls" />
                  <el-option label="Both (组合)" value="both" />
                </el-select>
              </div>
              <div class="form-row-full">
                <label>
                  DINOv3 模型路径
                  <el-tooltip content="DINOv3 ViT 模型路径，如 ./Dinov3-base" placement="top">
                    <el-icon class="help-icon"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </label>
                <el-input v-model="tc.config.value.training.dino_model" placeholder="./Dinov3-base 或 HuggingFace ID" />
              </div>
              <div class="control-row">
                <span class="label">DINOv3 分辨率</span>
                <el-select v-model="tc.config.value.training.dino_image_size" style="width: 180px">
                  <el-option label="224 (最轻量)" :value="224" />
                  <el-option label="384" :value="384" />
                  <el-option label="512 (推荐)" :value="512" />
                </el-select>
              </div>
            </template>

            <!-- REPA -->
            <div class="subsection-label">时间步感知 (REPA)</div>
            <div class="control-row">
              <span class="label">
                启用时间步感知
                <el-tooltip content="自动根据去噪阶段调整 Freq/Style 权重" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-switch v-model="tc.config.value.acrf.enable_timestep_aware_loss" />
            </div>
            <el-alert 
              v-if="tc.config.value.acrf.enable_timestep_aware_loss && (!tc.config.value.training.enable_freq && !tc.config.value.training.enable_style)" 
              type="warning" :closable="false" show-icon style="margin-top: 8px"
            >
              建议同时启用频域感知或风格结构，时间步感知才能发挥作用
            </el-alert>

            <!-- 其他高级参数 -->
            <div class="subsection-label">其他高级参数</div>
            <div class="control-row">
              <span class="label">
                Max Grad Norm
                <el-tooltip content="梯度裁剪阈值，防止梯度爆炸" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="tc.config.value.advanced.max_grad_norm" :min="0" :max="20" :step="0.5" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.advanced.max_grad_norm" :min="0" :max="20" :step="0.5" controls-position="right" class="input-fixed" />
            </div>
            <div class="control-row">
              <span class="label">
                Weight Decay
                <el-tooltip content="权重衰减，防止过拟合，一般保持0即可" placement="top">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </span>
              <el-slider v-model="tc.config.value.training.weight_decay" :min="0" :max="0.1" :step="0.001" :show-tooltip="false" class="slider-flex" />
              <el-input-number v-model="tc.config.value.training.weight_decay" :min="0" :step="0.001" controls-position="right" class="input-fixed" :precision="3" />
            </div>
          </div>
        </el-collapse-item>

      </el-collapse>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Setting, Check, Close, Loading, FolderOpened, DataAnalysis, Grid, TrendCharts, Files, Plus, Delete, Document, InfoFilled, QuestionFilled, Cpu, Tools } from '@element-plus/icons-vue'
import { useTrainingConfig } from '@/composables/useTrainingConfig'
import { registry } from '@/models'

// Import model registrations (side-effect: auto-registers into registry)
import '@/models/zimage'

const tc = useTrainingConfig()
const activeNames = ref(['model', 'acrf', 'lora', 'training', 'dataset', 'advanced'])

// Current model from registry (default: zimage)
const currentModel = computed(() => registry.get('zimage') || registry.default())

// Training type cards (generic — capabilities-driven)
type TagType = 'primary' | 'success' | 'warning' | 'info' | 'danger'
const trainingTypes = ref([
  { value: 'lora', label: 'LoRA', icon: '🔗', description: '低秩适配器，主模型冻结，显存友好', tag: '推荐', tagType: 'success' as TagType, disabled: false },
  { value: 'finetune', label: 'Finetune', icon: '🔥', description: '解冻主模型全量训练，需40GB+显存', tag: '高级', tagType: 'warning' as TagType, disabled: false },
  { value: 'controlnet', label: 'ControlNet', icon: '🎛️', description: '训练独立控制网络（边缘/深度/姿态）', tag: '', tagType: 'info' as TagType, disabled: false }
])

const conditionModes = ref([
  { value: 'text2img', label: 'Text2Img', icon: '✏️', description: '纯文本到图像生成', tag: '推荐', tagType: 'success' as TagType, disabled: false },
  { value: 'img2img', label: 'Img2Img', icon: '🔄', description: '图像风格转换/修复训练', tag: '', tagType: 'info' as TagType, disabled: false },
  { value: 'omni', label: 'Omni', icon: '🌌', description: 'SigLIP 多图条件训练', tag: '高级', tagType: 'warning' as TagType, disabled: false }
])



onMounted(() => tc.init())
</script>

<style scoped>
.training-config-page {
  padding: 24px;
  height: 100%;
  overflow-y: auto;
}

.config-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 20px 24px;
  margin: 0 auto 24px auto;
  max-width: 1400px;
}

.header-left { flex: 1; }
.header-left h1 { margin: 0 0 8px 0; display: flex; align-items: center; gap: 8px; font-size: 24px; }
.config-toolbar { display: flex; gap: 8px; align-items: center; }
.dataset-toolbar { display: flex; gap: 8px; align-items: center; }
.config-content-card { max-width: 1400px; margin: 0 auto; }
.config-collapse { border: none !important; }
.config-collapse :deep(.el-collapse-item) { margin-bottom: 16px; border: 1px solid var(--el-border-color-lighter); border-radius: 8px; overflow: hidden; background-color: var(--el-bg-color); }
.config-collapse :deep(.el-collapse-item:last-child) { margin-bottom: 0; }
.config-collapse :deep(.el-collapse-item__header) { background-color: var(--el-fill-color-lighter); padding: 16px 20px; font-weight: bold; border-bottom: 1px solid transparent; height: auto; line-height: 1.5; }
.config-collapse :deep(.el-collapse-item.is-active .el-collapse-item__header) { border-bottom-color: var(--el-border-color-lighter); }
.config-collapse :deep(.el-collapse-item__wrap) { border: none; }
.config-collapse :deep(.el-collapse-item:not(.is-active) .el-collapse-item__wrap) { display: none !important; }
.config-collapse :deep(.el-collapse-item__content) { padding: 0 0 16px 0; }

.collapse-title { display: flex; align-items: center; gap: 12px; font-size: 15px; }
.collapse-content { padding: 16px 20px 0 20px; }

.subsection-label { font-size: 11px; font-weight: 700; color: var(--el-text-color-secondary); margin: 20px 0 12px 0; text-transform: uppercase; letter-spacing: 1px; padding-top: 20px; border-top: 1px solid var(--el-border-color-lighter); }
.subsection-label:first-child { margin-top: 16px; padding-top: 0; border-top: none; }
.subsection-label-with-action { display: flex; justify-content: space-between; align-items: center; font-size: 11px; font-weight: 700; color: var(--el-text-color-secondary); margin: 20px 0 12px 0; text-transform: uppercase; letter-spacing: 1px; padding-top: 20px; border-top: 1px solid var(--el-border-color-lighter); }

.form-row-full { margin-bottom: 16px; }
.form-row-full:last-child { margin-bottom: 0; }
.form-row-full label { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--el-text-color-regular); margin-bottom: 6px; }

.control-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.control-row:last-child { margin-bottom: 0; }
.control-row .label { font-size: 12px; color: var(--el-text-color-regular); width: 160px; flex-shrink: 0; display: flex; align-items: center; gap: 4px; }

.help-icon { color: var(--el-color-primary-light-3); cursor: help; font-size: 14px; opacity: 0.8; }
.help-icon:hover { color: var(--el-color-primary); opacity: 1; }
.slider-flex { flex: 1; margin-right: 8px; }
.input-fixed { width: 100px !important; }

.dataset-item { background: var(--el-bg-color); padding: 16px; border-radius: 6px; border: 1px solid var(--el-border-color-light); margin-bottom: 12px; }
.dataset-item:last-child { margin-bottom: 0; }
.dataset-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.dataset-index { font-weight: bold; font-size: 13px; color: var(--el-color-primary); }

.empty-datasets { text-align: center; padding: 40px 20px; color: var(--el-text-color-secondary); }
.empty-datasets .el-icon { font-size: 48px; margin-bottom: 12px; opacity: 0.5; }
.empty-datasets p { margin: 0; font-size: 13px; }

.model-type-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.model-card { display: flex; align-items: center; gap: 12px; padding: 16px; border: 2px solid var(--el-border-color-lighter); border-radius: 12px; cursor: pointer; transition: all 0.2s ease; background: var(--el-bg-color); }
.model-card:hover:not(.disabled) { border-color: var(--el-color-primary-light-5); background: var(--el-color-primary-light-9); }
.model-card.active { border-color: var(--el-color-primary); background: var(--el-color-primary-light-9); box-shadow: 0 0 0 3px var(--el-color-primary-light-7); }
.model-card.disabled { opacity: 0.5; cursor: not-allowed; }
.model-icon { font-size: 32px; flex-shrink: 0; }
.model-info { flex: 1; min-width: 0; }
.model-name { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
.model-desc { font-size: 12px; color: var(--el-text-color-secondary); line-height: 1.4; }
</style>
