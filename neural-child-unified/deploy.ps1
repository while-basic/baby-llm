#----------------------------------------------------------------------------
#File:       deploy.ps1
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: PowerShell deployment script for neural child system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Neural Child Development System - Deployment Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker Desktop is not running." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and wait for it to fully start, then try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Steps:" -ForegroundColor Cyan
    Write-Host "  1. Open Docker Desktop application" -ForegroundColor White
    Write-Host "  2. Wait for the whale icon to appear in the system tray" -ForegroundColor White
    Write-Host "  3. Wait until Docker Desktop shows 'Docker Desktop is running'" -ForegroundColor White
    Write-Host "  4. Run this script again" -ForegroundColor White
    exit 1
}

# Check if Docker Compose is available
$dockerComposeCmd = "docker-compose"
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    $dockerComposeCmd = "docker compose"
    if (-not (docker compose version 2>$null)) {
        Write-Host "Error: Docker Compose is not available. Please install Docker Compose." -ForegroundColor Red
        exit 1
    }
}

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
$directories = @(
    "data\checkpoints",
    "data\logs",
    "data\models",
    "data\cache",
    "checkpoints",
    "memories",
    "development_results"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "✓ Directories created" -ForegroundColor Green
Write-Host ""

# Set Flask secret key if not set
if (-not $env:FLASK_SECRET_KEY) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $env:FLASK_SECRET_KEY = [System.BitConverter]::ToString($bytes).Replace("-", "").ToLower()
    Write-Host "Generated FLASK_SECRET_KEY (set this in production):" -ForegroundColor Yellow
    Write-Host "`$env:FLASK_SECRET_KEY='$env:FLASK_SECRET_KEY'" -ForegroundColor Yellow
    Write-Host ""
}

# Build Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
& $dockerComposeCmd.Split(' ') build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker image built" -ForegroundColor Green
Write-Host ""

# Start Ollama service
Write-Host "Starting Ollama service..." -ForegroundColor Yellow
& $dockerComposeCmd.Split(' ') up -d ollama
Start-Sleep -Seconds 5

# Pull Ollama model
Write-Host "Pulling Ollama model (gemma3:1b)..." -ForegroundColor Yellow
docker exec neural-child-ollama ollama pull gemma3:1b
Write-Host "✓ Ollama model pulled" -ForegroundColor Green
Write-Host ""

# Start services
Write-Host "Starting services..." -ForegroundColor Yellow
& $dockerComposeCmd.Split(' ') up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start services" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Services started" -ForegroundColor Green
Write-Host ""

# Wait for services to be ready
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check health
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Neural Child service is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Service may still be starting. Check logs with: docker-compose logs" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:"
Write-Host "  - Neural Child Web Interface: http://localhost:5000"
Write-Host "  - Ollama API: http://localhost:11434"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  - View logs: docker-compose logs -f"
Write-Host "  - Stop services: docker-compose down"
Write-Host "  - Restart services: docker-compose restart"
Write-Host "  - View status: docker-compose ps"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
