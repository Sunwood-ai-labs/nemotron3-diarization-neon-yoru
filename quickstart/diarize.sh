#!/bin/bash
# 使い方: ./diarize.sh meeting   （inputs/meeting.wav → results/meeting.rttm）
# 設定は実験（experiment/experiment.py）と同じ。
set -euo pipefail
name="$1"
docker compose run --rm experiment nemo-speech diarize "inputs/$name.wav" \
  --model models/Nemotron-3-Diarization.q8_0.gguf \
  --device cpu --preset v3-offline \
  --format rttm --recording-id "$name" --output "results/$name.rttm" \
  --onset 0.5 --offset 0.5 --pad-onset 0 --pad-offset 0 \
  --min-duration-on 0.1 --min-duration-off 0.1
