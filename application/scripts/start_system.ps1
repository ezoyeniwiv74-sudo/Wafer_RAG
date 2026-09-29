param(
    [switch]$NoBrowser,
    [switch]$SkipDocker,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ProjectRoot
$ComposeFile = Join-Path $RepositoryRoot "compose.local.yaml"
if (-not $PythonPath) { $PythonPath = Join-Path $RepositoryRoot ".runtime\wafer-rag\python.exe" }
$Python = $PythonPath
$LogDir = Join-Path $ProjectRoot ".logs"
$PidFile = Join-Path $LogDir "web_8002.pid"
$WebUrl = "http://127.0.0.1:8002/"
$ApiHealth = "http://127.0.0.1:9911/health"

function Wait-LocalEndpoint {
    param(
        [string]$Name,
        [string]$Uri,
        [int]$TimeoutSeconds = 180
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "[OK] $Name is ready: $Uri"
                return
            }
        } catch {}
        Start-Sleep -Seconds 2
    }
    throw "$Name did not become ready within $TimeoutSeconds seconds: $Uri"
}

$env:WAFER_PROFILE = "server"
$env:WAFER_QDRANT_ONLY = "1"
$env:WAFER_QDRANT_URL = "http://127.0.0.1:6333"
$env:WAFER_VLLM_BASE_URL = "http://127.0.0.1:8333/v1"
$env:WAFER_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
$env:WAFER_OLLAMA_EMBED_KEEP_ALIVE = "0"
$env:WAFER_REPORT_LLM_BASE_URL = "http://127.0.0.1:8333/v1"
$env:WAFER_REPORT_LLM_MODEL = "qwen3-wafer-report:latest"
$env:WAFER_IMAGE_SERVER_URL = "http://127.0.0.1:9911/analyze"
$env:WAFER_YEDN_ROOT = Join-Path $ProjectRoot "data\YEDN"
$env:WAFER_WEB_PORT = "8002"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
Remove-Item Env:WAFER_DISABLE_TEXT_EMBEDDING -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force $LogDir | Out-Null
if (-not (Test-Path -LiteralPath $Python)) { throw "Project Python not found: $Python" }

if (-not $SkipDocker) {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw "Docker Desktop is not running." }
    # Runtime startup must never contact an image registry or rebuild an image.
    # All images and model blobs are provisioned locally during installation.
    docker compose -f $ComposeFile up -d --pull never --no-build qdrant qwen dinov2-api
    if ($LASTEXITCODE -ne 0) {
        throw "Offline Docker startup failed. Verify that all local images were provisioned."
    }
}

# Runtime inference is localhost-only.  The unreachable proxy is a second
# guard against accidental external HTTP API use; NO_PROXY keeps local Docker
# services directly reachable.
$env:HTTP_PROXY = "http://127.0.0.1:9"
$env:HTTPS_PROXY = "http://127.0.0.1:9"
$env:ALL_PROXY = "http://127.0.0.1:9"

Wait-LocalEndpoint -Name "Qdrant" -Uri "http://127.0.0.1:6333/collections"
Wait-LocalEndpoint -Name "Qwen/Ollama" -Uri "http://127.0.0.1:8333/v1/models"
Wait-LocalEndpoint -Name "DINOv2" -Uri $ApiHealth

$webReady = $false
try {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8002/api/health" -TimeoutSec 2
    $webReady = ($response.StatusCode -eq 200)
} catch {}
if (-not $webReady) {
    $process = Start-Process -FilePath $Python -ArgumentList "api.py" -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput (Join-Path $LogDir "web_8002.out.log") `
        -RedirectStandardError (Join-Path $LogDir "web_8002.err.log") `
        -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ASCII
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 400
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8002/api/health" -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $webReady = $true; break }
        } catch {}
    }
}
if (-not $webReady) { throw "Web startup failed. See $LogDir\web_8002.err.log" }

Write-Host "[OK] Web: $WebUrl"
try {
    Invoke-RestMethod -Uri $ApiHealth -TimeoutSec 3 | Out-Null
    Write-Host "[OK] Image API: $ApiHealth"
} catch { Write-Host "[NOTICE] Web is ready, but image API is unavailable." }
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8333/v1/models" -TimeoutSec 3 | Out-Null
    Write-Host "[OK] Qwen embedding/report service: http://127.0.0.1:8333/v1"
} catch { Write-Host "[NOTICE] Qwen service is unavailable or still loading." }
if (-not $NoBrowser) { Start-Process $WebUrl }
