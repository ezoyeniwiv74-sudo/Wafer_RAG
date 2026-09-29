# 本机 Docker 运行说明

本方案适配当前主机：Windows 10、Docker Desktop WSL2 后端、RTX 3090 24GB。业务 Web 运行在 Windows Python 3.12 中；Qdrant、Qwen3-Embedding-0.6B、Qwen3-4B-Instruct 报告模型和 DINOv2 推理 API 运行在 Docker 中。两个 Qwen 模型由同一个 Ollama GPU 容器提供 OpenAI 兼容接口。报告侧使用 `qwen3-wafer-report` 配置（8K 上下文），检索完成后立即释放嵌入模型，避免两个 Qwen 模型长期同时占用显存。

## 1 创建 Windows Python 环境

在项目根目录的 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\application\scripts\setup_windows_env.ps1
```

脚本使用本机 `D:\anaconda3`，在项目 `.runtime\wafer-rag` 下创建隔离的 Python 3.12 环境。

## 2 准备并启动 Docker 服务

```powershell
powershell -ExecutionPolicy Bypass -File .\application\scripts\setup_local_docker.ps1
```

脚本会依次执行：导入交付包内的 DINOv2 离线基础镜像、构建推理 API、拉取 Qdrant 与 Ollama、启动三个容器，并下载两个 Qwen 模型和创建 8K 上下文报告配置。Qwen3-4B-Instruct 为约 2.5GB 的 Q4 量化模型，报告配置用于控制 KV Cache 的额外显存。

查看状态与日志：

```powershell
docker compose -f .\compose.local.yaml ps
docker compose -f .\compose.local.yaml logs -f qwen
docker compose -f .\compose.local.yaml logs -f dinov2-api
```

三个健康地址：

- Qdrant：`http://127.0.0.1:6333`
- Qwen 向量与报告：`http://127.0.0.1:8333/v1/models`
- DINOv2：`http://127.0.0.1:9911/health`

## 3 首次构建 Qdrant 集合

确认 Qdrant、Qwen 向量和 DINOv2 健康地址均可访问后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\application\scripts\initialize_qdrant.ps1 -Recreate
```

该步骤会读取 `application\data\DN_Description_500.csv` 与 `application\data\YEDN`，调用本地 Qwen 和 DINOv2 生成文本、SEM、Bin Map 向量，并写入 `wafer_dn_records_v2`。首次构建耗时较长；Qdrant 数据保存在 Docker 命名卷中，重启后无需重建。

## 4 启动业务系统

```powershell
powershell -ExecutionPolicy Bypass -File .\application\scripts\start_system.ps1
```

浏览器打开 `http://127.0.0.1:8002`。停止全部服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\application\scripts\stop_system.ps1
```

## 手动等价命令

如需逐步执行 Docker 操作：

```powershell
docker load -i .\docker_images\dinov2_cuda12.1_large.tar
docker compose -f .\compose.local.yaml pull qdrant qwen
docker compose -f .\compose.local.yaml build dinov2-api
docker compose -f .\compose.local.yaml up -d
docker exec wafer-qwen ollama pull qwen3-embedding:0.6b
docker exec wafer-qwen ollama pull qwen3:4b-instruct
docker cp .\deploy\Modelfile.qwen3-report wafer-qwen:/tmp/Modelfile.qwen3-report
docker exec wafer-qwen ollama create qwen3-wafer-report -f /tmp/Modelfile.qwen3-report
docker compose -f .\compose.local.yaml ps
```

Qdrant 使用 Docker 命名卷而不是 Windows 目录绑定，避免 WSL2 共享文件系统导致的存储兼容问题。所有端口只绑定到 `127.0.0.1`，不会直接暴露到局域网。
