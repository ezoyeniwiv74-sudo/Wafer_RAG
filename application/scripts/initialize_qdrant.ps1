param(
    [string]$PythonPath = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ApplicationRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ApplicationRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $RepositoryRoot ".runtime\wafer-rag\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Project Python was not found: $PythonPath. Run setup_windows_env.ps1 first."
}

$env:WAFER_PROFILE = "server"
$env:WAFER_QDRANT_ONLY = "1"
$env:WAFER_QDRANT_URL = "http://127.0.0.1:6333"
$env:WAFER_VLLM_BASE_URL = "http://127.0.0.1:8333/v1"
$env:WAFER_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
$env:WAFER_OLLAMA_EMBED_KEEP_ALIVE = "30m"
$env:WAFER_IMAGE_SERVER_URL = "http://127.0.0.1:9911/analyze"
$env:WAFER_YEDN_ROOT = Join-Path $ApplicationRoot "data\YEDN"

foreach ($uri in @("http://127.0.0.1:6333", "http://127.0.0.1:8333/v1/models", "http://127.0.0.1:9911/health")) {
    try { Invoke-RestMethod -Uri $uri -TimeoutSec 20 | Out-Null }
    catch { throw "Service is not ready: $uri`n$($_.Exception.Message)" }
}

$arguments = @("scripts\build_unified_dn_collection.py", "--execute")
if ($Recreate) { $arguments += "--recreate" }
Push-Location $ApplicationRoot
try {
    & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "Qdrant index initialization failed." }
    $knowledgeArguments = @("scripts\build_analysis_knowledge.py")
    if ($Recreate) { $knowledgeArguments += "--recreate" }
    & $PythonPath @knowledgeArguments
    if ($LASTEXITCODE -ne 0) { throw "Analysis knowledge initialization failed." }
}
finally { Pop-Location }

Write-Host "[OK] Qdrant collection wafer_dn_records_v2 is initialized."
Write-Host "[OK] Qdrant collection wafer_analysis_knowledge_v1 is initialized."
