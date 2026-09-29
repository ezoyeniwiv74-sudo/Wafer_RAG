param(
    [string]$EnvironmentPath = ""
)

$ErrorActionPreference = "Stop"
$ApplicationRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $ApplicationRoot
if (-not $EnvironmentPath) {
    $EnvironmentPath = Join-Path $RepositoryRoot ".runtime\wafer-rag"
}

$condaCandidates = @(
    "D:\anaconda3\Scripts\conda.exe",
    "C:\Miniforge3\Scripts\conda.exe",
    "C:\Users\$env:USERNAME\miniforge3\Scripts\conda.exe"
)
$Conda = $condaCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Conda) {
    throw "Conda was not found. Install installers\Miniforge3-Windows-x86_64.exe first."
}

$Python = Join-Path $EnvironmentPath "python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    & $Conda create --prefix $EnvironmentPath python=3.12 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python 3.12 environment." }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $RepositoryRoot "environments\windows_server_runtime_requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install the Windows application dependencies." }

& $Python -c "import fastapi, uvicorn, openai, qdrant_client, numpy, pandas, PIL; print('Windows runtime ready')"
Write-Host "[OK] Python: $Python"
