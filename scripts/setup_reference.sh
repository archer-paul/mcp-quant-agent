#!/usr/bin/env bash
# Clone third-party reference repos into ./reference/ (gitignored, read-only).
# Run once from the repo root:  bash scripts/setup_reference.sh
set -euo pipefail
mkdir -p reference && cd reference

clone() { # $1=url $2=dir
  if [ -d "$2" ]; then echo "✓ $2 already present"; else
    echo "→ cloning $2"; git clone --depth 1 "$1" "$2";
  fi
}

# Architecture references (study only — do not import directly)
clone https://github.com/TauricResearch/TradingAgents.git        TradingAgents
clone https://github.com/chmbrs/hedge_fund_agents.git            hedge_fund_agents

# External MCP data servers (may be run as data sources)
clone https://github.com/financial-datasets/mcp-server.git       mcp-financial-datasets
clone https://github.com/cfdude/mcp-finnhub.git                  mcp-finnhub

echo
echo "Done. Reference repos are in ./reference/ (gitignored)."
echo "Read them, copy/adapt useful bits into src/, cite sources in the thesis."
