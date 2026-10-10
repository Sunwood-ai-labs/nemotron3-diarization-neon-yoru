# 公式仕様の確認

調査日: 2026-10-10。

- [Nemotron 3 Diarization公式モデル](https://huggingface.co/nvidia/Nemotron-3-Diarization): 2026-09-23公開、最大8話者。出力は話者ごとの活動確率・時刻・ラベル。話者ラベルは音声内の登場順であり、NEON YORUのキャラクター名との対応はモデルの標準機能ではない。
- [NVIDIAの解説](https://huggingface.co/blog/nvidia/nemotron-diarization): ストリーミング/オフラインの「誰がいつ話したか」を扱う。文字起こしはASRとの組合せ。発話の重なりを検出することと、混ざった音声を人ごとの独立波形へ分離することは別の処理。
- [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp): 公式のC++ローカル実行経路。今回使用するv0.2.0リリースにはCPU版がある。公式GGUF Q8_0を明示指定する。
- [TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large): 約23Mパラメータの話者埋め込みモデル。公式NeMoの`EncDecSpeakerLabelModel.get_embedding()`で音声間のcosine類似度を比較できる。英語音声中心の学習データで、日本語のNEON YORUでの性能をそのまま保証する資料は確認できない。

今回の問いは「8人の合成キャラクター音声が実際に区別されるか」「単独の紹介音声を登録して独立生成された会話音声へ名前を付けられるか」。日本語ASRは別のWhisper small実験とする。8人という上限だけで精度を保証できず、実測結果から用途を判断する。

入力は`C:/Prj/YORU/public/voice/01.wav`〜`08.wav`と、`public/stream/S1.wav`〜`S8.wav`。音声は既存のGemini TTS生成物で、実人間8名の会議ではない。元ファイルは編集しない。台本の話者ラベル・タイミングから会話音声の異なる16発話を切り出し、1音声に8人を2巡配置する。

今回の基準は発話単位の正答数と、10msごとのエネルギー由来参照マスクによる診断指標。人手で発話境界を検証した正式DER、人物認証の正式EERとして報告しない。登録音声とテスト音声は別のTTS生成物だが、話者は同じ8キャラクターであり、小標本の試験である。

実行方式はDocker Compose v5.5.0。Docker Desktopの起動障害により、専用Podman環境`neon-yoru-diarization`のDocker互換APIへComposeを接続する。Composeコマンドの使用とDocker Engineでの実行は区別する。
