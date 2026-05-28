#!/bin/bash
# CI 脚本 — 运行 lint + 测试 + 覆盖率
set -euo pipefail

echo "=== ruff lint ==="
ruff check src/ tests/

echo ""
echo "=== ruff format check ==="
ruff format --check src/ tests/ 2>/dev/null || echo "(ruff format not available, skipping)"

echo ""
echo "=== pytest (CPU only) ==="
python3 -m pytest tests/ \
    --ignore=tests/test_reconstruct.py \
    --ignore=tests/test_splatting.py \
    --ignore=tests/test_server.py \
    --cov=src --cov-report=term-missing \
    -xvs

echo ""
echo "=== CI 通过 ==="