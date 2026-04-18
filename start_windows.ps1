# Rail Drishti — Windows PowerShell Start Script
# Run this from inside the rail_drishti/ folder

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║      RAIL DRISHTI — STARTING UP              ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "  ✗ Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Python found: $(python --version)" -ForegroundColor Green

# Check data files
$dataFiles = @(
    "data\stations.json",
    "data\schedules.json",
    "data\Train_details_22122017.csv",
    "data\train_delay_data.csv"
)

Write-Host ""
Write-Host "  Checking data files..." -ForegroundColor Yellow
$missing = $false
foreach ($f in $dataFiles) {
    if (Test-Path $f) {
        $size = (Get-Item $f).Length / 1MB
        Write-Host ("  ✓ {0}  ({1:F1} MB)" -f $f, $size) -ForegroundColor Green
    } else {
        Write-Host "  ✗ MISSING: $f" -ForegroundColor Red
        $missing = $true
    }
}

if ($missing) {
    Write-Host ""
    Write-Host "  ERROR: Place missing data files in the data\ folder." -ForegroundColor Red
    exit 1
}

# Create venv if not exists
if (-not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
Write-Host "  Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "  Installing dependencies..." -ForegroundColor Yellow
pip install -q -r backend\requirements.txt
Write-Host "  ✓ Dependencies ready" -ForegroundColor Green

# Set environment variable and start
Write-Host ""
Write-Host "  ┌──────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │  Frontend + API:  http://localhost:8000/app  │" -ForegroundColor Cyan
Write-Host "  │  API Docs:        http://localhost:8000/docs │" -ForegroundColor Cyan
Write-Host "  │  Health:          http://localhost:8000/health│" -ForegroundColor Cyan
Write-Host "  └──────────────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""
Write-Host "  NOTE: First startup takes 60-120 seconds (loading 79MB data + training ML)" -ForegroundColor Yellow
Write-Host ""

$env:RAILWAYS_DATA_DIR = "./data"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
