# 轻量迁移包：交给目标电脑 Codex 的部署说明

把 ZIP 解压到只包含英文、数字和下划线的目录，然后将下面这段话直接发给目标电脑上的 Codex：

> 请读取项目根目录的 `CODEX_RESTORE_GUIDE.md`、`PORTABLE_DEPLOYMENT_MANIFEST.json`、`LOCAL_DOCKER_SETUP.md` 和 `compose.local.yaml`。检查本机 Windows、NVIDIA 驱动、WSL2、Docker Desktop 和磁盘空间；安装项目 Python 3.12 环境；按 manifest 下载 Qwen3 Embedding 0.6B、Qwen3 4B Instruct 和 Meta DINOv2 ViT-L/14 官方权重；构建 DINOv2 在线推理镜像；启动 Qdrant、Ollama、DINOv2；运行 `initialize_qdrant.ps1 -Recreate` 重建两个 Qdrant 集合；最后启动 Web 并验证文字检索、图片检索和多模态报告。不要把本地 CSV/YEDN 用作运行态搜索后端，运行态必须保持 `WAFER_QDRANT_ONLY=1`。

## 包内包含

- 完整业务代码和前端页面。
- `application/data` 下的 500 条文字记录、YEDN 图片和工业知识 JSON。
- `models/deployed_model` 下的分类头和 Bin Map LoRA 增量权重。
- DINOv2 官方源码、推理 API 和构建文件。
- Windows Python 依赖清单、Docker Compose、初始化和启动脚本。
- 所有训练参数、生成参数、集合名称和模型下载清单。

## 不包含

为保持轻量，不包含 Python 虚拟环境、8.3 GB DINOv2 Docker tar、Docker 数据卷、Qdrant 生成向量和 Ollama 模型 blob。这些内容均可由目标电脑的 Codex根据本包重建。

## 必须下载的模型

1. Ollama `qwen3-embedding:0.6b`。
2. Ollama `qwen3:4b-instruct`。
3. 使用 `deploy/Modelfile.qwen3-report` 从第 2 项创建 `qwen3-wafer-report:latest`，上下文为 8192。
4. Meta 官方 `dinov2_vitl14_pretrain.pth`：
   `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`

下载后的 DINOv2 权重放到：

```text
models/dinov2_base/dinov2_vitl14_pretrain.pth
```

## 推荐部署顺序

```powershell
# 1. 创建 Windows Python 环境
powershell -ExecutionPolicy Bypass -File .\application\scripts\setup_windows_env.ps1

# 2. 下载 DINOv2 官方主干权重
New-Item -ItemType Directory -Force .\models\dinov2_base
Invoke-WebRequest `
  -Uri "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" `
  -OutFile ".\models\dinov2_base\dinov2_vitl14_pretrain.pth"

# 3. 在线构建轻量包对应的 DINOv2 推理镜像
docker build -f .\deploy\Dockerfile.dinov2-online `
  -t dinov2-company-inference:local .

# 4. 拉取并启动基础服务
docker compose -f .\compose.local.yaml pull qdrant qwen
docker compose -f .\compose.local.yaml up -d --no-build qdrant qwen dinov2-api

# 5. 下载两个 Qwen 模型
docker exec wafer-qwen ollama pull qwen3-embedding:0.6b
docker exec wafer-qwen ollama pull qwen3:4b-instruct
docker cp .\deploy\Modelfile.qwen3-report wafer-qwen:/tmp/Modelfile.qwen3-report
docker exec wafer-qwen ollama create qwen3-wafer-report -f /tmp/Modelfile.qwen3-report

# 6. 重新构建业务与知识 Qdrant 集合
powershell -ExecutionPolicy Bypass -File .\application\scripts\initialize_qdrant.ps1 -Recreate

# 7. 启动 Web
powershell -ExecutionPolicy Bypass -File .\application\scripts\start_system.ps1 -SkipDocker
```

打开 `http://127.0.0.1:8002`。验证接口：

- `http://127.0.0.1:6333/collections`
- `http://127.0.0.1:8333/v1/models`
- `http://127.0.0.1:9911/health`
- `http://127.0.0.1:8002/api/report/status`

如果目标电脑显卡显存不足，优先保持 Qwen 4B 的量化版本、`num_ctx=8192`、`OLLAMA_NUM_PARALLEL=1`；不要同时常驻 embedding 和报告模型。
