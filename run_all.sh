#!/bin/bash
echo "=== PPG Regressor ==="
python -m src.train_xgb "$@"
echo ""
echo "=== Breakout Classifier ==="
python -m src.train_breakout "$@"
