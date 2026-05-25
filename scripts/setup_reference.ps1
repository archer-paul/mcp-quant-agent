# Clone third-party reference repos into .\reference\ (gitignored, read-only).
# Run once from the repo root (PowerShell):  .\scripts\setup_reference.ps1
#
# If you get an execution-policy error, run this once in the same session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "reference" | Out-Null
Set-Location "reference"

function Clone-Repo {
    param([string]$Url, [string]$Dir)
    if (Test-Path $Dir) {
        Write-Host "[ok] $Dir already present" -ForegroundColor Green
    } else {
        Write-Host "[..] cloning $Dir" -ForegroundColor Cyan
        git clone --depth 1 $Url $Dir
    }
}

# Architecture references (study only - do not import directly)
Clone-Repo "https://github.com/TauricResearch/TradingAgents.git" "TradingAgents"
Clone-Repo "https://github.com/chmbrs/hedge_fund_agents.git"     "hedge_fund_agents"

# External MCP data servers (may be run as data sources)
Clone-Repo "https://github.com/financial-datasets/mcp-server.git" "mcp-financial-datasets"
Clone-Repo "https://github.com/cfdude/mcp-finnhub.git"            "mcp-finnhub"

Set-Location ".."
Write-Host ""
Write-Host "Done. Reference repos are in .\reference\ (gitignored)." -ForegroundColor Green
Write-Host "Read them, copy/adapt useful bits into src\, cite sources in the thesis."
