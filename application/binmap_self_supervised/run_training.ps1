param(
    [int]$Epochs = 12,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutputDir = Join-Path $ProjectRoot "models\binmap_ssl_lora"
$DinoRepo = "D:\dinov2_sem_defect_training_review_20260803\training\vendor\dinov2-main"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

docker run --rm `
  --gpus '"device=0"' `
  --shm-size 4g `
  --network none `
  -v "${ProjectRoot}:/workspace" `
  -v "${DinoRepo}:/workspace/vendor/dinov2-main:ro" `
  -w /workspace `
  --entrypoint python `
  dinov2-company-training:latest `
  /workspace/binmap_self_supervised/train_binmap_lora.py `
  --data-root /workspace/data/YEDN `
  --output-dir /workspace/models/binmap_ssl_lora `
  --repo-dir /workspace/vendor/dinov2-main `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --workers 2

if ($LASTEXITCODE -ne 0) {
    throw "Bin Map self-supervised training failed with exit code $LASTEXITCODE"
}
