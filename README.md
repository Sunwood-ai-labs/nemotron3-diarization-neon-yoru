# Nemotron 3 Diarization × NEON YORU：8人の話者分離を実測

NVIDIAの話者分離モデル [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)（最大8人対応）が、8人組ユニットNEON YORUの声を聞き分けられるかを実測した記録です。実験コード、実測結果、自分の音声で試すための一式を置いています。

解説記事は [ARTICLE.md](ARTICLE.md) です。

## 結果

8人の声は4〜5人分にまとめられ、正しく区別できたのは16発話中7発話まででした（2026-10-10、CPU推論）。

| 条件 | 実行方式 | 正しく区別できた発話（満点16） | 検出した人数（正解8人） |
| --- | --- | ---: | ---: |
| 順番発話・BGMなし | C++版 Q8 | 7 | 4 |
| 2人の重なりあり | C++版 Q8 | 7 | 5 |
| 順番発話・BGMあり | C++版 Q8 | 5 | 4 |
| 順番発話・BGMなし | Transformers版 FP32 | 7 | 5 |
| 発話を最後まで残す | Transformers版 FP32 | 5 | 3 |
| 自己紹介の先頭4秒 | Transformers版 FP32 | 4 | 2 |
| 自己紹介の先頭12秒 | Transformers版 FP32 | 6 | 3 |

- 声の照合（TitaNet-Large）は、正しい発話区間を与えると16発話中13発話で正解。
- Nemotron 3の推定区間にTitaNetで名前を付ける一連の処理は、16発話すべて「不明」。
- 同時発話の検出は適合率79.0%、再現率85.1%。72.84秒の音声をCPUで6.40秒で処理。

対象は日本語の合成音声（Gemini TTS）8キャラクターの短い音声です。実在の人間8名の会議や長時間の音声にそのまま当てはまる結果ではありません。指標は正式なDERではなく、発話単位の正答数です。詳しい条件は [docs/REPORT.md](docs/REPORT.md) と [diagnosis/REPORT.md](diagnosis/REPORT.md) にあります。

## 自分の音声で試す

Docker Composeが動く環境があれば試せます。GPUは不要です。

```bash
git clone https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru.git
cd nemotron3-diarization-neon-yoru/quickstart
./fetch.sh                 # 公式C++ランタイムとQ8モデルを取得（チェックサム確認つき）
docker compose build

ffmpeg -i input.mp3 -ac 1 -ar 16000 inputs/meeting.wav
./diarize.sh meeting       # → results/meeting.rttm

# 正解データ（truth.example.json の形）を用意して採点
docker compose run --rm experiment python score.py results/meeting.rttm inputs/meeting.truth.json
```

| ファイル | 内容 | 確認状況 |
| --- | --- | --- |
| `quickstart/Dockerfile` | 実験と同じコンテナ定義 | 実験で使用 |
| `quickstart/diarize.sh` | 実験と同じ設定のNemotron 3実行コマンド | コマンドは実験で使用。このスクリプト経由では未実行 |
| `quickstart/score.py` | 検出人数と発話正答数の採点 | `results/` の実出力で記事と同じ数字になることを確認 |
| `quickstart/fetch.sh` | ランタイムとモデルの取得 | URLの応答のみ確認。通しでは未実行 |
| `quickstart/fp32.py` | 量子化なしFP32版（Transformers）での話者分離 | 実験スクリプトの抜粋。この形では未実行 |
| `quickstart/titanet.py` | TitaNet-Largeでの声の照合 | 実験スクリプトの抜粋。この形では未実行 |

## フォルダ構成

| フォルダ | 内容 |
| --- | --- |
| `quickstart/` | 自分の音声で試すための一式 |
| `experiment/` | 今回の実験で実際に動かしたスクリプトと設定 |
| `results/` | 3条件の話者分離結果（RTTM）、採点、TitaNetの類似度、文字起こし |
| `diagnosis/` | 原因の切り分け実験（発話長・閾値）の結果 |
| `docs/` | 実測レポートと事前調査メモ |

`experiment/` のスクリプトはNEON YORUの音声素材と実験時のフォルダ構成を前提にしています。音声素材、モデルの重み、実行ログはこのリポジトリに含めていないため、そのままでは動きません。処理内容の参照用です。

## 使用したモデルとランタイム

| 役割 | モデル・ランタイム |
| --- | --- |
| 話者分離 | Nemotron 3 Diarization 公式GGUF Q8_0（revision `f667ed73`）、[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) v0.2.0 CPU |
| 話者分離（対照） | 同モデルのTransformers 5.19.0、FP32、CPU |
| 声の照合 | [TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| 文字起こし | Whisper small（faster-whisper、CPU int8） |

各モデルの利用条件は、それぞれの配布元の記載に従ってください。
