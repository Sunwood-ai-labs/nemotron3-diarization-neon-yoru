#!/bin/bash
# 公式C++ランタイム（NeMo-Speech.cpp v0.2.0 CPU版）とNemotron 3 DiarizationのQ8モデルを取得する。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime models inputs enrollment clips results

RELEASE=https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz
REVISION=f667ed73aee57d40cc39428eb768b4fd87a0a29e
GGUF=https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/$REVISION/Nemotron-3-Diarization.q8_0.gguf

curl -L -o runtime/linux-release.tar.gz "$RELEASE"
curl -L "$RELEASE.sha256" | sed 's/nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz/linux-release.tar.gz/' > runtime/linux-release.tar.gz.sha256
(cd runtime && sha256sum -c linux-release.tar.gz.sha256)

curl -L -o models/Nemotron-3-Diarization.q8_0.gguf "$GGUF"
# 実験で使ったGGUFのSHA256
echo "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1  models/Nemotron-3-Diarization.q8_0.gguf" | sha256sum -c
