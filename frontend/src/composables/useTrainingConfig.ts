import { ref, computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

/**
 * Training configuration management composable.
 * Handles config CRUD, presets, dataset management, LoRA/ControlNet lists.
 */
export function useTrainingConfig() {
    const route = useRoute()

    const loading = ref(false)
    const saving = ref(false)
    const currentConfigName = ref('default')
    const savedConfigs = ref<any[]>([])
    const selectedPreset = ref('')
    const presets = ref<any[]>([])

    // Dialog visibility
    const showNewConfigDialog = ref(false)
    const showSaveAsDialog = ref(false)
    const newConfigName = ref('')
    const saveAsName = ref('')

    // Dataset management
    const cachedDatasets = ref<any[]>([])
    const selectedDataset = ref('')
    const selectedRegDataset = ref('')

    // Model lists
    const loraList = ref<{ name: string, path: string, size: number }[]>([])
    const controlnetList = ref<{ name: string, path: string, size: number }[]>([])

    // System paths
    const systemPaths = ref({ model_path: '', output_base_dir: '' })

    // Default config factory
    function getDefaultConfig() {
        return {
            name: 'default',
            training_type: 'lora',
            condition_mode: 'text2img',
            timestep: {
                mode: 'uniform',
                // Uniform
                shift: 3.0,
                use_dynamic_shift: true,
                base_shift: 0.5,
                max_shift: 1.15,
                // LogNorm
                logit_mean: 0.0,
                logit_std: 1.0,
                // ACRF
                acrf_steps: 10,
                jitter_scale: 0.02,
                // Shared
                latent_jitter_scale: 0.0,
            },
            acrf: {
                snr_gamma: 5.0,
                snr_floor: 0.1,
                raft_mode: false,
                free_stream_ratio: 0.3,
                enable_timestep_aware_loss: false,
                timestep_high_threshold: 0.7,
                timestep_low_threshold: 0.3,
                enable_curvature: false,
                lambda_curvature: 0.05,
                curvature_interval: 10,
                curvature_start_epoch: 0,
                cfg_training: false,
                cfg_scale: 7.0,
                cfg_training_ratio: 0.3
            },
            network: { dim: 8, alpha: 4.0 },
            lora: {
                resume_training: false,
                resume_lora_path: '',
                train_adaln: false,
                train_norm: false,
                train_single_stream: false
            },
            controlnet: {
                resume_training: false,
                controlnet_path: '',
                control_types: ['canny'],
                conditioning_scale: 0.75,
            },
            optimizer: {
                type: 'AdamW8bit',
                learning_rate: '1e-4',
                relative_step: false,
            },
            training: {
                output_name: '',
                learning_rate: 0.0001,
                learning_rate_str: '1e-4',
                weight_decay: 0,
                lr_scheduler: 'constant',
                lr_warmup_steps: 0,
                lr_num_cycles: 1,
                lr_pct_start: 0.1,
                lr_div_factor: 10,
                lr_final_div_factor: 100,
                lambda_mse: 1.0,
                lambda_l1: 1.0,
                lambda_cosine: 0.1,
                enable_freq: false,
                lambda_freq: 0.3,
                alpha_hf: 1.0,
                beta_lf: 0.2,
                enable_style: false,
                lambda_style: 0.3,
                lambda_light: 0.5,
                lambda_color: 0.3,
                enable_dino: false,
                lambda_dino: 0.1,
                dino_model: '',
                dino_image_size: 512,
                dino_feature_mode: 'patch',
            },
            dataset: {
                batch_size: 1,
                shuffle: true,
                enable_bucket: true,
                drop_text_ratio: 0.1,
                datasets: [] as any[]
            },
            reg_dataset: {
                enabled: false,
                weight: 1.0,
                ratio: 0.5,
                datasets: [] as any[]
            },
            advanced: {
                max_grad_norm: 1.0,
                gradient_checkpointing: true,
                blocks_to_swap: 0,
                num_train_epochs: 10,
                save_every_n_epochs: 1,
                gradient_accumulation_steps: 4,
                mixed_precision: 'bf16',
                seed: 42,
                num_gpus: 1,
                gpu_ids: ''
            }
        }
    }

    const config = ref(getDefaultConfig())

    // Computed helpers
    type TagType = 'primary' | 'success' | 'warning' | 'info' | 'danger'

    const trainingTypeDisplayName = computed(() => {
        const types: Record<string, string> = { lora: 'LoRA', finetune: 'Finetune', controlnet: 'ControlNet' }
        return types[config.value.training_type] || 'LoRA'
    })

    const trainingTypeTagType = computed((): TagType => {
        const tags: Record<string, TagType> = { lora: 'success', finetune: 'warning', controlnet: 'info' }
        return tags[config.value.training_type] || 'success'
    })

    const outputNameLabel = computed(() => {
        const labels: Record<string, string> = { lora: 'LoRA 输出名称', finetune: 'Finetune 输出名称', controlnet: 'ControlNet 输出名称' }
        return labels[config.value.training_type] || '输出名称'
    })

    const outputNamePlaceholder = computed(() => {
        const names: Record<string, string> = { lora: 'zimage-lora', finetune: 'zimage-finetune', controlnet: 'zimage-controlnet' }
        return names[config.value.training_type] || 'output'
    })

    const isControlNetMode = computed(() => config.value.training_type === 'controlnet')

    // Basic Loss Toggle (MSE + L1 + Cosine)
    const lastBasicLoss = ref({ mse: 1.0, l1: 1.0, cosine: 0.1 })
    const enableBasicLoss = computed({
        get: () => config.value.training.lambda_mse > 0 || config.value.training.lambda_l1 > 0 || config.value.training.lambda_cosine > 0,
        set: (val: boolean) => {
            if (val) {
                config.value.training.lambda_mse = lastBasicLoss.value.mse > 0 ? lastBasicLoss.value.mse : 1.0
                config.value.training.lambda_l1 = lastBasicLoss.value.l1 > 0 ? lastBasicLoss.value.l1 : 1.0
                config.value.training.lambda_cosine = lastBasicLoss.value.cosine
            } else {
                lastBasicLoss.value = {
                    mse: config.value.training.lambda_mse,
                    l1: config.value.training.lambda_l1,
                    cosine: config.value.training.lambda_cosine
                }
                config.value.training.lambda_mse = 0
                config.value.training.lambda_l1 = 0
                config.value.training.lambda_cosine = 0
            }
        }
    })



    // --- API Functions ---

    async function loadConfigList() {
        try {
            const res = await axios.get('/api/training/configs')
            savedConfigs.value = res.data.configs
        } catch (e) { console.error('Failed to load config list:', e) }
    }

    async function loadConfig(configName: string) {
        loading.value = true
        try {
            const res = await axios.get(`/api/training/config/${configName}`)
            const defaultCfg = getDefaultConfig()
            config.value = {
                ...defaultCfg,
                ...res.data,
                timestep: { ...defaultCfg.timestep, ...res.data.timestep },
                acrf: { ...defaultCfg.acrf, ...res.data.acrf },
                network: { ...defaultCfg.network, ...res.data.network },
                lora: { ...defaultCfg.lora, ...res.data.lora },
                optimizer: { ...defaultCfg.optimizer, ...res.data.optimizer },
                training: { ...defaultCfg.training, ...res.data.training },
                dataset: {
                    ...defaultCfg.dataset, ...res.data.dataset,
                    datasets: res.data.dataset?.datasets || []
                },
                reg_dataset: {
                    ...defaultCfg.reg_dataset, ...res.data.reg_dataset,
                    datasets: res.data.reg_dataset?.datasets || []
                },
                advanced: { ...defaultCfg.advanced, ...res.data.advanced }
            }
            const lr = config.value.training.learning_rate || 0.0001
            config.value.training.learning_rate_str = lr >= 0.001 ? lr.toString() : lr.toExponential()
            currentConfigName.value = configName
        } catch (e: any) {
            if (configName !== 'default') ElMessage.warning(`配置 "${configName}" 加载失败，使用默认配置`)
            config.value = getDefaultConfig()
            currentConfigName.value = 'default'
        } finally { loading.value = false }
    }

    async function loadSavedConfig() {
        if (currentConfigName.value) await loadConfig(currentConfigName.value)
    }

    async function saveCurrentConfig(activeNames: string[]) {
        if (!currentConfigName.value) { ElMessage.warning('请先选择或创建一个配置'); return }

        const outputName = config.value.training.output_name?.trim()
        if (!outputName) {
            ElMessage.warning('请填写输出名称（用于标识训练记录）')
            return
        }

        try {
            const runsRes = await axios.get('/api/training/runs')
            const exists = (runsRes.data.runs || []).some((r: any) => r.name === outputName)
            if (exists) { ElMessage.error(`训练记录 "${outputName}" 已存在，请使用唯一名称`); return }
        } catch { }

        saving.value = true
        try {
            await axios.post('/api/training/config/save', { name: currentConfigName.value, config: config.value })
            ElMessage.success('配置已发送到训练器')
            await loadConfigList()
        } catch (e: any) {
            ElMessage.error('保存失败: ' + (e.response?.data?.detail || e.message))
        } finally { saving.value = false }
    }

    async function createNewConfig() {
        if (!newConfigName.value.trim()) { ElMessage.warning('请输入配置名称'); return }
        try {
            await axios.post('/api/training/config/save', { name: newConfigName.value, config: { ...config.value, name: newConfigName.value } })
            ElMessage.success(`配置 "${newConfigName.value}" 已创建`)
            currentConfigName.value = newConfigName.value
            await loadConfigList()
            showNewConfigDialog.value = false
            newConfigName.value = ''
        } catch (e: any) { ElMessage.error('创建失败: ' + (e.response?.data?.detail || e.message)) }
    }

    async function saveAsNewConfig() {
        if (!saveAsName.value.trim()) { ElMessage.warning('请输入配置名称'); return }
        try {
            await axios.post('/api/training/config/save', { name: saveAsName.value, config: { ...config.value, name: saveAsName.value } })
            ElMessage.success(`已另存为 "${saveAsName.value}"`)
            currentConfigName.value = saveAsName.value
            await loadConfigList()
            showSaveAsDialog.value = false
            saveAsName.value = ''
        } catch (e: any) { ElMessage.error('保存失败: ' + (e.response?.data?.detail || e.message)) }
    }

    async function deleteCurrentConfig() {
        if (currentConfigName.value === 'default') return
        try {
            await ElMessageBox.confirm(`确定要删除配置 "${currentConfigName.value}" 吗？`, '删除确认', { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' })
            await axios.delete(`/api/training/config/${currentConfigName.value}`)
            ElMessage.success('配置已删除')
            currentConfigName.value = 'default'
            await loadConfigList()
            await loadConfig('default')
        } catch (e: any) {
            if (e !== 'cancel') ElMessage.error('删除失败: ' + (e.response?.data?.detail || e.message))
        }
    }

    async function loadPresets() {
        try { const res = await axios.get('/api/training/presets'); presets.value = res.data.presets } catch { }
    }

    function loadPreset() {
        if (!selectedPreset.value) return
        const preset = presets.value.find(p => p.name === selectedPreset.value)
        if (preset) {
            config.value = JSON.parse(JSON.stringify(preset.config))
            ElMessage.success(`已加载预设: ${preset.name}`)
            selectedPreset.value = ''
        }
    }

    async function loadCachedDatasets() {
        try { const res = await axios.get('/api/dataset/cached'); cachedDatasets.value = res.data.datasets } catch { }
    }

    async function loadLoraList() {
        try { const res = await axios.get('/api/loras'); loraList.value = res.data.data || res.data.loras || [] } catch { }
    }

    async function fetchControlnets() {
        try { const res = await axios.get('/api/controlnets'); controlnetList.value = res.data.controlnets || [] } catch { }
    }

    // Dataset helpers
    function onDatasetSelect() {
        if (selectedDataset.value) {
            config.value.dataset.datasets.push({ cache_directory: selectedDataset.value, num_repeats: 1, resolution_limit: 1024 })
            selectedDataset.value = ''
        }
    }

    function addDataset() {
        config.value.dataset.datasets.push({ cache_directory: '', num_repeats: 1, resolution_limit: 1024 })
    }

    function removeDataset(idx: number) { config.value.dataset.datasets.splice(idx, 1) }

    async function validateDatasetPath(ds: any) {
        if (!ds.cache_directory) { ds._validated = false; return }
        ds._validating = true; ds._validated = false
        try {
            const res = await axios.post('/api/dataset/validate', { path: ds.cache_directory })
            ds._validated = true; ds._valid = res.data.valid; ds._error = res.data.error
            ds._imageCount = res.data.imageCount; ds._hasCached = res.data.hasCachedData
            if (res.data.valid && res.data.path) ds.cache_directory = res.data.path
        } catch (e: any) {
            ds._validated = true; ds._valid = false; ds._error = e.response?.data?.detail || '验证失败'
        } finally { ds._validating = false }
    }

    function onRegDatasetSelect() {
        if (selectedRegDataset.value) {
            config.value.reg_dataset.datasets.push({ cache_directory: selectedRegDataset.value, num_repeats: 1 })
            selectedRegDataset.value = ''
        }
    }

    function addRegDataset() { config.value.reg_dataset.datasets.push({ cache_directory: '', num_repeats: 1 }) }
    function removeRegDataset(idx: number) { config.value.reg_dataset.datasets.splice(idx, 1) }

    // Learning rate helpers
    function parseLearningRate() {
        const str = config.value.training.learning_rate_str
        if (!str) return
        const value = parseFloat(str)
        if (!isNaN(value) && value > 0) config.value.training.learning_rate = value
    }

    function formatLearningRate(value: number): string {
        return value >= 0.001 ? value.toString() : value.toExponential().replace('+', '')
    }

    // Init
    async function init() {
        await loadConfigList()
        const editConfig = route.query.edit as string
        if (editConfig && editConfig !== 'default') {
            await loadConfig(editConfig)
            ElMessage.info(`正在编辑配置: ${editConfig}`)
        } else {
            await loadConfig('default')
        }
        await loadPresets()
        await loadCachedDatasets()
        await loadLoraList()
        fetchControlnets()
    }

    return {
        config,
        loading,
        saving,
        currentConfigName,
        savedConfigs,
        selectedPreset,
        presets,
        showNewConfigDialog,
        showSaveAsDialog,
        newConfigName,
        saveAsName,
        cachedDatasets,
        selectedDataset,
        selectedRegDataset,
        loraList,
        controlnetList,
        systemPaths,
        trainingTypeDisplayName,
        trainingTypeTagType,
        outputNameLabel,
        outputNamePlaceholder,
        isControlNetMode,
        enableBasicLoss,
        getDefaultConfig,
        init,
        loadSavedConfig,
        saveCurrentConfig,
        createNewConfig,
        saveAsNewConfig,
        deleteCurrentConfig,
        loadPreset,
        onDatasetSelect,
        addDataset,
        removeDataset,
        validateDatasetPath,
        onRegDatasetSelect,
        addRegDataset,
        removeRegDataset,
        parseLearningRate,
        formatLearningRate
    }
}
