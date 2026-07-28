#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Polymas - Full Pipeline Runner"
echo "    Root: $ROOT_DIR"

# Generate protobuf code
echo ""
echo "==> [1/6] Generating protobuf stubs..."
cd "$ROOT_DIR" && make proto

# Build Scala
echo ""
echo "==> [2/6] Building Scala ingestion service..."
cd "$ROOT_DIR/services/ingestion-scala" && sbt -batch compile 2>&1 | tail -5

# Build Go
echo ""
echo "==> [3/6] Building Go normalization service..."
cd "$ROOT_DIR/services/normalization-go" && go build -o bin/server ./cmd/server 2>&1

# Build Rust
echo ""
echo "==> [4/6] Building Rust control plane..."
cd "$ROOT_DIR/services/control-plane-rust" && cargo build --release 2>&1 | tail -5

# Setup Python
echo ""
echo "==> [5/6] Setting up Python ML engine..."
cd "$ROOT_DIR/services/ml-engine-python"
if [ ! -d ".venv" ]; then
  python3.12 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q 2>/dev/null || true
.venv/bin/pip install -e . -q 2>/dev/null || true

# Build Next.js
echo ""
echo "==> [6/6] Building Next.js dashboard..."
cd "$ROOT_DIR/apps/dashboard-nextjs"
npm install --silent 2>&1 | tail -3
npm run build 2>&1 | tail -5

echo ""
echo "==> All builds complete!"
echo "    Run 'make docker-up' to start all services in containers."
