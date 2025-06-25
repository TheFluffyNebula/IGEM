#!/usr/bin/env bash
nohup python3 -u ./src/run_experiments.py > benchmark.log 2>&1 &
disown
