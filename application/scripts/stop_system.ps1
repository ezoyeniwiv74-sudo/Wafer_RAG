$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ProjectRoot
$ComposeFile = Join-Path $RepositoryRoot "compose.local.yaml"
$PidFile = Join-Path $ProjectRoot ".logs\web_8002.pid"
if (Test-Path -LiteralPath $PidFile) {
    $processId = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if ($processId -match '^\d+$') { Stop-Process -Id ([int]$processId) -ErrorAction SilentlyContinue }
    Remove-Item -LiteralPath $PidFile -Force
}
try {
    docker compose -f $ComposeFile stop *> $null
} catch {}
Write-Host "System stopped."
