#!/bin/bash
# Run job files sequentially through the pool.
cd "$(dirname "$0")/.."
for f in "$@"; do python3 scripts/pool.py "$f" --workers 7; done
