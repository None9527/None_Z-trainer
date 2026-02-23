# Z-Trainer

Z-Image 模型一站式训练 & 推理平台。

## ✨ 功能特性

### 🎨 图片生成
- **Z-Image Pipeline** — 基于 diffusers，自动 CPU offload，低显存推理
- **LoRA 动态加载** — 无需重启 pipeline，实时切换 LoRA 权重
- **LoRA 对比模式** — 同 seed 同 prompt，一键对比原始模型 vs LoRA 效果
- **实时进度** — SSE 流式推送生成进度，支持断线重连 + 后台轮询恢复
- **生成历史** — 自动保存所有生成结果，支持参数回填 & 批量管理

### 🏋️ LoRA 训练
- **可视化配置** — 完整的训练参数面板，支持预设保存/加载
- **多种优化器** — AdamW / FP8 / BF16 / Prodigy / Lion
- **学习率调度** — Cosine / Constant / OneCycle 等多种策略
- **多 Loss** — MSE + L1 + Cosine + 频域感知 + 风格迁移 + Timestep-Aware + Curvature
- **ACRF 时间步** — 锚点耦合整流流采样 + RAFT 模式 + SNR 加权
- **实时监控** — WebSocket 推送训练日志 & 指标，集成 TensorBoard
- **一键开始** — 从数据集标注到训练启动全流程 Web UI

### 📁 数据集管理
- **图片管理** — 上传、预览、批量操作
- **AI 自动标注** — 集成 Ollama 视觉模型，一键生成 caption
- **标签编辑** — 在线编辑/修改标注文本
- **缓存预处理** — Latent + Text Encoder 缓存，训练时零编码开销

### 📦 LoRA 管理
- **模型列表** — 查看所有训练产出的 LoRA 模型
- **快速加载** — 直接从管理页加载到生成 pipeline

---

## 🚀 安装部署

### 1. 环境要求

| 项目 | 已测试版本 |
|------|-----------|
| **操作系统** | Ubuntu 24.04 LTS |
| **Python** | 3.12.3 |
| **Node.js** | 22.22.0 (npm 10.9.4) |
| **GPU** | NVIDIA GeForce RTX 5090 (32GB) |
| **PyTorch** | 2.10.0+cu130 |
| **CUDA** | 13.0 |

### 2. 克隆项目

```bash
git clone <your-repo-url> Z-trainer
cd Z-trainer
```

### 3. 安装 PyTorch（重要！必须先装）

```bash
# CUDA 13.0 (已测试)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

验证安装：
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. 安装 Node.js（前端构建需要）

```bash
# 使用 nvm 安装 Node.js 22 (已测试)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc
nvm install 22

# 验证
node --version   # v22.22.0
npm --version    # 10.9.4
```

### 5. 运行安装脚本

```bash
chmod +x setup.sh start.sh
./setup.sh
```

安装脚本会自动完成：
- ✅ 创建 Python 虚拟环境 (`venv/`)
- ✅ 安装 Python 依赖 (`requirements.txt`)
- ✅ 安装最新 diffusers (git)
- ✅ 生成 `.env` 配置文件
- ✅ 构建前端 (`npm install` + `npm run build`)

### 6. 配置环境变量

编辑 `.env` 文件，**必须设置模型路径**：

```bash
nano .env
```

关键配置项：

```ini
# [必须] Z-Image 模型路径（包含 transformer/ 子目录的文件夹）
MODEL_PATH=/path/to/your/Z-Image

# [可选] 数据集存放目录
DATASET_PATH=./datasets

# [可选] 训练输出目录
OUTPUT_PATH=./model-output

# [可选] Ollama 地址（用于 AI 标注）
OLLAMA_HOST=http://127.0.0.1:11434

# [可选] 服务端口
TRAINER_PORT=28000
```

### 7. 下载模型权重

将 Z-Image 模型放到 `MODEL_PATH` 指定的路径。目录结构应为：

```
Z-Image/
├── transformer/           # Transformer 权重
│   ├── config.json
│   └── *.safetensors
├── vae/                   # VAE
├── text_encoder/          # Text Encoder
├── text_encoder_2/        # Text Encoder 2
└── model_index.json
```

### 8. 启动服务

```bash
./start.sh
```

启动后访问：
- **Web UI**: http://localhost:28000
- **TensorBoard**: http://localhost:6006

#### 启动参数

```bash
# 自定义端口
./start.sh --port 8080

# 开发模式（后端热更新）
./start.sh --dev

# 查看更多选项
./start.sh --help
```

---

## 📖 使用流程

### 训练 LoRA

```
1. 数据集页面 → 上传训练图片
2. 数据集页面 → AI 自动标注 (或手动编辑)
3. 数据集页面 → 生成缓存 (Latent + Text)
4. 训练配置页面 → 设置参数 (LoRA rank、学习率、损失权重等)
5. 开始训练页面 → 一键启动
6. 监控页面 → 实时查看 Loss 曲线
```

### 推理生成

```
1. LoRA 管理页面 → 选择模型加载
2. 图片生成页面 → 输入 Prompt
3. 图片生成页面 → 对比模式 (可选)
4. 生成历史 → 查看/复用参数
```

---

## ⚙️ 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRAINER_PORT` | `28000` | 服务端口 |
| `TRAINER_HOST` | `0.0.0.0` | 监听地址 |
| `MODEL_PATH` | `./Z-Image` | 基础模型路径 |
| `DATASET_PATH` | `./datasets` | 数据集目录 |
| `OUTPUT_PATH` | `./model-output` | 训练输出目录 |
| `GENERATION_OUTPUT_PATH` | `./image-output` | 生成图片目录 |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama 服务地址 |

---

## 📂 项目结构

```
Z-trainer/
├── start.sh / start.bat              # 启动脚本
├── setup.sh / setup.bat              # 首次部署脚本
├── .env                              # 环境配置 (从 env.example 生成)
├── requirements.txt                  # Python 依赖
│
├── backend/                          # 后端 (Python + FastAPI)
│   ├── interface/                    # API 层 (FastAPI Routers)
│   │   ├── main.py                   # 入口: uvicorn backend.interface.main:app
│   │   ├── generation_router.py      # 图片生成 API + SSE
│   │   ├── training_router.py        # 训练管理 API
│   │   ├── dataset_router.py         # 数据集管理 API
│   │   ├── cache_router.py           # 潜变量缓存 API
│   │   └── websocket_manager.py      # WebSocket 训练日志推送
│   ├── domain/                       # 领域层 (实体 + 仓储接口)
│   ├── infrastructure/               # 基础设施层 (实现)
│   └── trainer_core/                 # 训练核心
│       ├── zimage_trainer/           # Z-Image 训练器 + 数据集
│       └── shared/                   # 共享组件 (Loss/优化器/调度器)
│
├── frontend/                         # 前端 (Vue 3 + TypeScript + Element Plus)
│   └── src/views/                    # 页面组件
│
├── Z-Image/                          # 模型权重 (默认路径)
├── datasets/                         # 训练数据集
├── model-output/                     # 训练产出 (LoRA 权重)
├── image-output/                     # 生成图片输出
└── configs/                          # 训练配置预设
```

## 🏗️ 架构

- **DDD 四层架构**: Domain → Application → Infrastructure → Interface
- **前后端分离**: FastAPI (后端) + Vue 3 / Element Plus (前端)
- **实时通信**: SSE (生成进度) + WebSocket (训练日志)
- **CPU Offload**: 模型权重常驻 CPU，推理时逐模块搬到 GPU
- **LoRA 热加载**: 两阶段合并，无需重载 pipeline

## 📝 开发

```bash
# 前端开发模式 (热更新)
cd frontend && npm run dev

# 后端开发模式
./start.sh --dev --log-level debug

# 仅构建前端
cd frontend && npx vite build
```
