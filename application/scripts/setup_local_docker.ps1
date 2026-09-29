param(
    [switch]$SkipPull
)

$ErrorActionPreference = "Stop"
$ApplicationRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ApplicationRoot
$ComposeFile = Join-Path $RepositoryRoot "compose.local.yaml"
$BaseTar = Join-Path $RepositoryRoot "docker_images\dinov2_cuda12.1_large.tar"

docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop is not running or Docker Engine is not accessible." }

$localImages = @(docker image ls --format "{{.Repository}}:{{.Tag}}")
if ($localImages -notcontains "dinov2:cuda12.1-large") {
    if (-not (Test-Path -LiteralPath $BaseTar)) { throw "DINOv2 offline image is missing: $BaseTar" }
    Write-Host "[1/7] Loading the 8.3 GB DINOv2 CUDA 12.1 image..."
    docker load -i $BaseTar
}

if (-not $SkipPull) {
    Write-Host "[2/7] Pulling Qdrant and Ollama images..."
    docker compose -f $ComposeFile pull qdrant qwen
}

Write-Host "[3/7] Building the DINOv2 inference API..."
docker compose -f $ComposeFile build dinov2-api

Write-Host "[4/7] Starting the three local containers..."
docker compose -f $ComposeFile up -d qdrant qwen dinov2-api

Write-Host "[5/7] Downloading Qwen3-Embedding-0.6B (639 MB)..."
docker exec wafer-qwen ollama pull qwen3-embedding:0.6b
if ($LASTEXITCODE -ne 0) { throw "Qwen embedding model download failed." }

Write-Host "[6/7] Downloading Qwen3-4B-Instruct (about 2.5 GB)..."
docker exec wafer-qwen ollama pull qwen3:4b-instruct
if ($LASTEXITCODE -ne 0) { throw "Qwen report model download failed." }

Write-Host "[7/7] Creating the 8K-context low-VRAM report profile..."
docker cp (Join-Path $RepositoryRoot "deploy\Modelfile.qwen3-report") "wafer-qwen:/tmp/Modelfile.qwen3-report"
docker exec wafer-qwen ollama create qwen3-wafer-report -f /tmp/Modelfile.qwen3-report
if ($LASTEXITCODE -ne 0) { throw "Qwen report profile creation failed." }

Write-Host "Containers and both Qwen models are ready. Follow Qwen logs with:"
Write-Host "docker compose -f `"$ComposeFile`" logs -f qwen"
Write-Host "Health URLs: Qdrant http://127.0.0.1:6333; Qwen http://127.0.0.1:8333/v1/models; DINOv2 http://127.0.0.1:9911/health"
