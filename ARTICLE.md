# NVIDIA Nemotron 3 Diarizationは8人組を聞き分けられるか：NEON YORUで実測

2026-10-10

NVIDIAの話者分離AI「Nemotron 3 Diarization」は、8人組ユニットNEON YORUの声を4〜5人分にしか分けられず、16発話のうち正しく区別できたのは7発話でした。量子化なしのモデルでも、発話を長くしても、BGMを外しても、8人には届きませんでした。

## この実験でわかったこと

- **8人は分けられなかった。** Nemotron 3は8人の声を4〜5人にまとめ、正しく区別できたのは16発話中5〜7発話でした。
- **声の照合は単体なら使える水準だった。** 正しい発話区間を渡すと、TitaNetは16発話中13発話で正しい名前を選びました。
- **2つをつなぐと名前は1件も付かなかった。** 混ざった区間を照合に渡したため、16発話すべてが「不明」になりました。

対象は日本語の合成音声（TTS）8キャラクター、合計1〜3分の短い音声です。実在の人間8名の会議や長時間の番組にそのまま当てはまる結果ではありません。

## Nemotron 3 Diarizationとは

Nemotron 3 Diarizationは、NVIDIAが2026年9月23日に公開した話者分離モデルです。複数人の声が入った録音から「誰が、いつ話したか」を割り出します。前身のSortformerは最大4人まででしたが、8人まで扱えるようになりました。

| 項目 | 公式モデルカードの記載 |
| --- | --- |
| 対応人数 | 最大8人 |
| 使い方 | リアルタイム（ストリーミング）と録音済み（オフライン）の両方 |
| 遅延 | 推奨設定で最短0.32秒。オフライン相当の設定は30.4秒 |
| 出力 | 話者ごとの「話している確率」と時刻。番号は声が初めて出た順 |
| 重なり | 同時に話している区間も複数の話者として出せる |
| 学習データ | 実会話 約1万時間と、合成した多人数音声 82,611時間 |
| 学習言語 | 英語、中国語、ヒンディー語など。日本語は名指しされていない |
| 利用条件 | 重み公開。商用・非商用とも可 |
| 動かし方 | NeMo、Transformers、C++版のNeMo-Speech.cpp（GPUなしのPCでも動く） |

出典は[公式モデルカード](https://huggingface.co/nvidia/Nemotron-3-Diarization)です。

「最大8人」は、出力の枠が8人分あるという意味です。8人を必ず聞き分けられるという保証ではありません。そこで、8人組ユニットNEON YORUの声で上限いっぱいを実測しました。

## 何を試したのか：Nemotron 3を主役にした3段構え

Nemotron 3が答えるのは「誰が、いつ」までです。発言者の名前が入った議事録を作るには、ほかのモデルと組み合わせます。

今回は、主役のNemotron 3に2つのモデルをつなぎました。

| 順番 | モデル | 役割 | たとえると |
| --- | --- | --- | --- |
| 1 | Nemotron 3 Diarization | 声を話者A・B・C…に分け、話した時刻を出す | 目隠しで「今の声はさっきと同じ人か」を当てる係 |
| 2 | TitaNet-Large | 登録済みの声と比べ、話者Aが誰かを決める | 声の名簿と照らし合わせる受付係 |
| 3 | Whisper small | 話した内容を文字にする | 速記係 |

Nemotron 3が出すのは「話者A」「話者B」という仮の番号だけです。実名を付ける機能はないため、2番目の照合モデルが必要になります。

## 実験のやり方

8人それぞれの会話音声から2発話ずつ、合計16発話を切り出して1本の音声に並べました。8人が順に1回ずつ話し、そのあと逆順でもう1回ずつ話します。誰がどこで話したかは台本でわかっているので、モデルの答えを採点できます。

音声はNEON YORUの既存素材（Gemini TTSで生成した日本語の声）です。同じ16発話から、条件を1つずつ変えた3本を用意しました。

| 条件 | 内容 | 長さ |
| --- | --- | --- |
| 順番発話 | 1人ずつ話す。BGMなし | 72.84秒 |
| 重なりあり | 後半の4組を1.5秒ずらして2人同時に話させる | 61.07秒 |
| BGMあり | 順番発話に、声より20dB小さいBGMを重ねる | 72.84秒 |

声の照合では、各キャラクターの自己紹介音声（13〜27秒）を「声の名簿」として登録しました。テストに使う会話音声とは別に生成された音声なので、同じ波形を当てさせる出来レースにはなっていません。

採点は発話単位です。16発話のうち、同じ人の2発話を同じ話者として、別の人を別の話者として扱えた数を数えます。

## 結果1：Nemotron 3は8人を4〜5人にまとめてしまった

いちばん条件の良い「順番に話す・BGMなし」でも、Nemotron 3が検出したのは4人でした。別人の声を同じ話者と判定したためで、正しく区別できた発話は半分に届きません。

![条件別の正答数と検出人数](https://raw.githubusercontent.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/main/assets/chart-diarization.png)

グラフの元になった数値は次の表のとおりです。

| 条件 | 実行方式 | 正しく区別できた発話（満点16） | 検出した人数（正解8人） |
| --- | --- | ---: | ---: |
| 順番発話・BGMなし | C++版 Q8 | 7 | 4 |
| 2人の重なりあり | C++版 Q8 | 7 | 5 |
| 順番発話・BGMあり | C++版 Q8 | 5 | 4 |
| 順番発話・BGMなし | Transformers版 FP32 | 7 | 5 |
| 発話を最後まで残す | Transformers版 FP32 | 5 | 3 |
| 自己紹介の先頭4秒 | Transformers版 FP32 | 4 | 2 |
| 自己紹介の先頭12秒 | Transformers版 FP32 | 6 | 3 |

量子化なしのFP32版に替えても、発話を長くしても、結果はほぼ変わりませんでした。条件を変えた各試験のねらいは「原因の切り分け」で説明します。

一方で、同時発話の検出は良好でした。2人が重なって話した区間の85.1%を検出し、重なりと判定した区間の79.0%が実際の重なりでした。処理も速く、72.84秒の音声をCPUだけで6.40秒で終えています。

## 結果2：声の照合は16発話中13発話で正解

正しい発話区間を人間側で与え、「この声は8人のうち誰か」をTitaNetに選ばせると、16発話中13発話で正解しました。話者分離より明らかに良い成績です。

| キャラクター | 正解 | 取り違えた相手 |
| --- | --- | --- |
| 紅月レイ | 1/2 | ユラ |
| 海月よるみ | 2/2 | |
| 天狗カガセ | 2/2 | |
| ぽんた | 2/2 | |
| 神代シオン | 2/2 | |
| 深海ナギ | 1/2 | 神代シオン |
| 黒百合レン | 2/2 | |
| ユラ | 1/2 | 紅月レイ |

間違いは3件で、紅月レイとユラが互いに取り違えられ、深海ナギが神代シオンと判定されました。

![キャラクター別の類似度と採用基準](https://raw.githubusercontent.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/main/assets/chart-titanet.png)

ただし、正解したときの類似度（cosine）も0.53〜0.75と高くありません。「自信があるときだけ名前を出す」基準（類似度0.65以上、かつ2位との差0.05以上）にすると、名前が出たのは16発話中4発話だけでした。その4発話はすべて正解です。

本人を名簿から外したテストでは、16発話中14発話を「登録されていない人」として正しく保留できました。

## 結果3：2つをつなぐと全員「不明」

Nemotron 3が分けた区間をそのままTitaNetに渡すと、正しい名前が付いた発話は16発話中0件でした。全件が「不明」として保留されています。

![照合だけ・話者分離だけ・2つをつないだ場合の正答数](https://raw.githubusercontent.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/main/assets/chart-pipeline.png)

理由は前段の失敗がそのまま後段に流れたことです。Nemotron 3が別人の声を1人分にまとめたため、TitaNetには複数人の声が混ざった音声が届きました。混ざった声は名簿の誰とも十分に似ず、基準に届きません。

基準を下げれば名前は出ます。ただ、混ざった声に名前を付けるだけなので、正しくなる保証はありません。間違った名前を出すより「不明」で止まったのは、安全側の挙動です。

文字起こし（Whisper small）は20区間分を生成できました。内容の正確さは採点していませんが、「神代シオン」が「紙しろしおん」になるなど、固有名詞の誤りは目で見てわかる程度にあります。

## 原因の切り分け

「なぜ8人を分けられないのか」について、疑わしい要因を1つずつ変えて確かめました。どれを変えても8人には届かず、根本原因はまだ特定できていません。

| 疑った要因 | 確かめ方 | 結果 | 判定 |
| --- | --- | --- | --- |
| モデルを軽くしたせい（Q8量子化） | 量子化なしのFP32版で同じ音声を処理 | 正答数は同じ | 主因ではない |
| 発話が短すぎるせい | 3〜4秒の抜粋を、発話の最後まで伸ばす | 正答数は減った | 主因ではない |
| 発話が短すぎるせい（別の音声） | 自己紹介を4秒から12秒に伸ばす | 少し増えたが3人分にしか分かれない | 主因ではない |
| BGMのせい | BGMあり・なしを比較 | BGMで悪化するが、なしでも失敗 | 悪化要因。主因ではない |
| 判定の厳しさ（閾値）のせい | 閾値を0.25〜0.65の4段階で変更 | どの値でも8人に戻らない | 主因ではない |
| CPUで動かしたせい | 未実施（実験機にGPUなし） | | GPUで精度が上がる根拠はない |

Q8とFP32は実行プログラムも後処理も違うため、「量子化による悪化はゼロ」とまでは言えません。言えるのは、量子化をやめても失敗するという点です。

GPUは通常、計算を速くするためのものです。同じモデルなら答えはほぼ変わらないので、GPUに替えるだけで聞き分けが良くなるとは考えにくいです。

残っている候補は3つです。

- 日本語の合成音声が、モデルの学習データと性質が違う
- 8キャラクターの声質が、モデルから見ると似ている
- 推論の実装や設定に、まだ見つけていない問題がある

## 実験の舞台裏：動かすまでが一番かかった

開始から最初の結果まで約24分かかり、その大半は実行環境のトラブル対応でした。Nemotron 3自体の推論は、1本あたり4〜6秒で終わっています。

| 時刻（2026-10-10） | 出来事 |
| --- | --- |
| 9:29 | 調査と実験を開始。公式モデルカードで最大8人対応を確認 |
| 9:35 | Windowsのアプリ制御がPyTorchのDLLをブロック。Docker Composeで実行する方針に変更 |
| 9:39 | Docker Desktopが起動エラーで停止。復旧もリセットもできず |
| 9:44 | 実験専用のPodman環境を作成。Docker Composeのコマンドはそのまま使用 |
| 9:53 | コンテナが起動し、Nemotron 3の最初の結果が出る |
| 9:59 | TitaNet・FP32版・Whisperまで完了し、初回レポート |
| 10:14〜10:20 | 原因の切り分け実験（発話長・閾値・BGM） |

実行エンジンがDockerからPodmanに替わっても、compose.yamlはそのまま使えました。GPUのないノートPCでも、C++版ランタイムならNemotron 3を試せることも確認できました。

## Nemotron 3の評価と、次に試すこと

今回の条件では、Nemotron 3をNEON YORUの8人の自動識別にそのまま使うのは難しい、というのが結論です。ただし、良かった点もはっきりあります。

| 観点 | 実測 | 評価 |
| --- | --- | --- |
| 速さ | 72.84秒の音声をCPUだけで6.40秒で処理（実時間の約11倍速） | 良い |
| 導入のしやすさ | GPUなしのノートPCとコンテナで動作 | 良い |
| 同時発話の検出 | 重なった区間の85.1%を検出し、検出の79.0%が正しい | 良い |
| 8人の聞き分け | 8人を2〜5人にまとめてしまう | 課題 |
| BGMへの強さ | 小さなBGMでも正答数が減る | 課題 |

**この実験から言えないこと**

- 実在の人間8名の会議でも同じ精度になるかどうか。今回はすべて合成音声です。
- 番組本編のような長時間の音声での精度。最長でも3分21秒しか試していません。
- モデルそのものの限界なのか、設定で直せるのか。

**次に試すこと**

- [ ] 実在の人間の声（できれば日本語・英語の両方）で同じ試験を行い、合成音声が原因かを確かめる
- [ ] 人数を2人・4人・6人・8人と増やし、どこから崩れるかを調べる
- [ ] 公式推奨のオフライン設定とfull-attentionの`--offline`で再測定する
- [ ] 番組本編の長い連続音声で測る
- [ ] GPU環境で同じ音声を処理し、結果が変わらないことを確かめる

## 動かし方：同じ実験を自分の音声で再現する

実験に使ったコードと実測結果は、公開リポジトリ [Sunwood-ai-labs/nemotron3-diarization-neon-yoru](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru) に置きました。ここでは要点だけ抜粋します。全文はリンク先を見てください。

NEON YORUの音声素材は配布していないため、手元の音声に置き換えて試す手順です。GPUは不要で、Docker Composeが動くPCがあれば動きます。

### 用意するもの

| 用意するもの | 内容 |
| --- | --- |
| 実行環境 | Docker Compose（Docker DesktopでもPodmanでも可）。今回はCPU 8コア・メモリ16GBを割り当て |
| 検証用の音声 | 複数人が話している音声。16kHz・モノラルのWAV |
| 正解データ | 「誰が、何秒から何秒まで話したか」の一覧。採点に使う |
| 登録用の音声 | 名前を付けたい人それぞれの単独音声（名前付けまで試す場合）。今回は1人13〜27秒 |

### 手順1：リポジトリを取得し、ランタイムとモデルを入手する

```bash
git clone https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru.git
cd nemotron3-diarization-neon-yoru/quickstart
./fetch.sh
docker compose build
```

[`fetch.sh`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/fetch.sh) は、公式のC++ランタイム（NeMo-Speech.cpp v0.2.0のCPU版）とNemotron 3のQ8モデルをダウンロードし、チェックサムを確かめます。モデルは今回使ったrevisionに固定しています。

```bash
RELEASE=https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz
REVISION=f667ed73aee57d40cc39428eb768b4fd87a0a29e
GGUF=https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/$REVISION/Nemotron-3-Diarization.q8_0.gguf
```

コンテナの中身は [`Dockerfile`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/Dockerfile) のとおりで、実験と同じものです。要点は次の2行です。

```dockerfile
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir 'nemo_toolkit[asr]==3.0.0' numpy scipy soundfile 'huggingface_hub>=1.0' faster-whisper
```

### 手順2：Nemotron 3で話者分離する

音声を16kHz・モノラルに変換して `inputs/` に置き、実行します。

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 inputs/meeting.wav
./diarize.sh meeting
```

[`diarize.sh`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/diarize.sh) の中身は、実験で使ったのと同じコマンドです。

```bash
docker compose run --rm experiment nemo-speech diarize "inputs/$name.wav" \
  --model models/Nemotron-3-Diarization.q8_0.gguf \
  --device cpu --preset v3-offline \
  --format rttm --recording-id "$name" --output "results/$name.rttm" \
  --onset 0.5 --offset 0.5 --pad-onset 0 --pad-offset 0 \
  --min-duration-on 0.1 --min-duration-off 0.1
```

`results/meeting.rttm` に、1行1区間で結果が出ます。4列目が開始秒、5列目が長さ（秒）、8列目が話者番号です。

```text
SPEAKER meeting 1 0.780 3.780 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER meeting 1 5.100 4.080 <NA> <NA> speaker_2 <NA> <NA>
```

まず見るべきは、`speaker_` の番号が何種類出たかです。実際の人数より少なければ、今回と同じく別人がまとめられています。

### 手順3：採点する

正解データを [`truth.example.json`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/truth.example.json) の形で用意し、[`score.py`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/score.py) にかけます。

```bash
docker compose run --rm experiment python score.py results/meeting.rttm inputs/meeting.truth.json
```

採点の考え方は実験と同じです。推定話者と本人を最も合うように1対1で対応付け、発話ごとに「その区間で一番長く話していた推定話者」が本人かどうかを数えます。核になるのは次の部分です。

```python
# 推定チャネルと本人の最適な1対1対応
rows, cols = linear_sum_assignment(-(ref.astype(int).T @ pred.astype(int)))
mapping = {c: names[r] for r, c in zip(rows, cols)}

correct = 0
for t in truth:
    counts = pred[round(t['start'] * 100):round(t['end'] * 100)].sum(axis=0)
    guess = mapping.get(int(counts.argmax())) if counts.max() > 0 else None
    correct += guess == t['speaker']
```

このスクリプトは、リポジトリの [`results/`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/tree/main/results) にある今回の出力にかけて、記事と同じ数字（4人・7/16など）が出ることを確認しています。

### 手順4（任意）：FP32版での確認と、名前付け

| やりたいこと | スクリプト | 抜粋元 |
| --- | --- | --- |
| 量子化なしのFP32版で同じ音声を処理する | `fp32.py` | `experiment/native.py` |
| TitaNetで登録済みの声と照合し、名前を付ける | `titanet.py` | `experiment/experiment.py` |

スクリプトは [`fp32.py`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/fp32.py) と [`titanet.py`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/quickstart/titanet.py)、抜粋元は [`experiment/native.py`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/experiment/native.py) と [`experiment/experiment.py`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/experiment/experiment.py) です。

FP32版の核は、Transformersの標準的な呼び出しです。出力は手順2と同じRTTM形式に書き出すので、`score.py` でそのまま採点できます。

```python
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id).eval()
inputs = processor(audio, sampling_rate=sr, return_tensors='pt')
with torch.inference_mode():
    logits = model(**inputs).logits
segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
```

名前付けは、声の特徴ベクトル同士のcosine類似度で決めます。

```python
scores = emb('clips/unknown.wav') @ enroll.T               # cosine類似度
rank = np.argsort(-scores)
first, second = rank[0], rank[1]
accepted = scores[first] >= 0.65 and scores[first] - scores[second] >= 0.05
```

類似度0.65以上・2位との差0.05以上という基準は、今回の実験で仮に置いた値です。実運用では自分のデータで決め直してください。

この2本は実験スクリプトを短くしたもので、短くした形のままでは実行していません。実験ではTransformers 5.19.0を使いました。

### リポジトリの中身

| フォルダ | 内容 |
| --- | --- |
| `quickstart/` | 自分の音声で試すための一式（上の手順で使うもの） |
| `experiment/` | 今回の実験で実際に動かしたスクリプトと設定。NEON YORUの音声素材が前提 |
| `results/` | 3条件の話者分離結果（RTTM）、採点、TitaNetの類似度、文字起こし |
| `diagnosis/` | 原因の切り分け実験の結果 |
| `docs/` | 実測レポートと事前調査メモ |

各フォルダは[リポジトリのトップ](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru)から開けます。

条件を変えて切り分けるなら、BGMあり・なし、発話を短く切った版・最後まで残した版、同時発話を入れた版を作り、手順2と手順3を繰り返します。今回の入力の組み立て方は [`experiment/prepare.py`](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru/blob/main/experiment/prepare.py) にあります。

## 実行条件と指標の定義

| 項目 | 内容 |
| --- | --- |
| 話者分離（主系統） | Nemotron 3 Diarization 公式GGUF Q8_0、NeMo-Speech.cpp v0.2.0、CPU。v3-offlineのチャンク推論 |
| 話者分離（対照） | 同モデルのTransformers 5.19.0、FP32、CPU。processorの標準後処理 |
| 声の照合 | TitaNet-Large。NeMoの`EncDecSpeakerLabelModel.get_embedding()`でcosine類似度を比較 |
| 文字起こし | Whisper small（faster-whisper、CPU int8） |
| 入力 | 16kHz mono。Gemini TTSによる日本語8キャラクター。BGMは音声全体のRMS比 −20dB |
| 実行基盤 | Docker Compose v5.5.0、エンジンは専用Podman（WSL）。GPUはIntel UHD Graphicsのみ |
| 再現性 | 入力とモデルのSHA256、モデルrevision、実行ログを保存 |

指標の読み方には注意点があります。

- **発話正答数**は、既知の発話区間内で最も長く活動した推定チャネルを取り、チャネルと話者の最適な1対1対応を評価時に求めた値です。正式なDER（話者分離誤り率）ではありません。
- **重なり検出**の参照は、10ms単位・RMS 0.006超のエネルギーマスクです。人手で境界を確定したラベルではありません。
- **Q8とFP32の比較**は、実装・推論方式・後処理が異なります。量子化だけの影響を測る対照試験にはなっていません。
- **閾値の探索**は検証データ上で行った診断です。運用向けの校正値ではありません。
- **自己紹介の試験**は2巡目が同じ波形の逆順です。別の発話への一般化は測っていません。
- **標本数**は各条件16発話です。1発話の差は誤差の範囲と考えてください。

## 参考リンク

- [Nemotron 3 Diarization（公式モデルカード）](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [NVIDIAによる解説記事](https://huggingface.co/blog/nvidia/nemotron-diarization)
- [NeMo-Speech.cpp（C++版ランタイム）](https://github.com/NVIDIA/NeMo-Speech.cpp)
- [TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)
- [実験コードと実測結果（公開リポジトリ）](https://github.com/Sunwood-ai-labs/nemotron3-diarization-neon-yoru)
