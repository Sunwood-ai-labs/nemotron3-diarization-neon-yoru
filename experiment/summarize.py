import json
from pathlib import Path
root=Path(__file__).resolve().parent
def read(p): return json.loads((root/p).read_text(encoding='utf8'))
cells=read('results/diarization-summary.json');identity=read('results/titanet-summary.json');named=read('results/named-summary.json')
manifest=read('manifest.json');queries=read('results/titanet-queries.json')
label={'clean':'順番発話・BGMなし','overlap':'2人の重なりあり','bgm_minus20db':'順番発話・BGM -20dB'}
lines=['# NEON YORU 8人の話者分離・名前照合 実験結果', '',
 '2026-10-10。既存の日本語Gemini TTS音声8キャラクターを使い、実際のNVIDIAモデルでCPU推論した。', '',
 '## 実測結果', '', '| 条件 | 音声長 | 検出話者数 | 発話正答数 | 単独発話フレーム一致 | 推論時間 |',
 '|---|---:|---:|---:|---:|---:|']
for c in cells:
    lines.append(f"| {label[c['name']]} | {c['audio_seconds']:.2f}秒 | {c['detected_speaker_channels']}/8 | {c['turns_correct']}/{c['turns_total']} | {c['exclusive_energy_frame_exact_accuracy']*100:.1f}% | {c['wall_seconds']:.2f}秒 |")
ov=next(c for c in cells if c['name']=='overlap')
def pct(v): return '検出なし' if v is None else f'{v*100:.1f}%'
lines += ['',f"重なりの検出: 適合率 {pct(ov['overlap_precision'])}、再現率 {pct(ov['overlap_recall'])}。エネルギー参照による重なりは{ov['reference_overlap_seconds']:.2f}秒、モデルの重なり判定は{ov['predicted_overlap_seconds']:.2f}秒。",
 '',f"TitaNetの8人からの強制選択: **{identity['closed_set_correct']}/{identity['queries']}発話が正解**。cosine≥0.65かつ1位と2位の差≥0.05で採用する試作判定は、採用{identity['accepted']}、採用中の正解{identity['accepted_correct']}。閾値は今回の出力を見る前に固定したが、実運用向けの校正は行っていない。",
 '',f"本人を登録候補から外した留出試験では、不明として拒否できたのは{identity['unknown_holdout_rejected']}/{identity['queries']}。未登録者の安全な判定を保証する正式な評価ではない。",
 '',f"Nemotronの推定クラスタにTitaNetで名前を付けた一連の処理: **{named['correct']}/{named['total']}発話が正しい名前**。この値は正解ラベルを使ったクラスタ対応付けとは別の、実際の名前照合結果。",
 '', '## キャラクターごとの照合', '', '| キャラクター | 強制選択の正答 | テスト音声1の1位 | cosine | テスト音声2の1位 | cosine |','|---|---:|---|---:|---|---:|']
for sp,cast in manifest['cast'].items():
    q=[q for q in queries if q['speaker']==sp]
    lines.append(f"| {cast['name']} | {sum(x['correct_closed_set'] for x in q)}/2 | {q[0]['top1_name']} | {q[0]['top1_cosine']:.3f} | {q[1]['top1_name']} | {q[1]['top1_cosine']:.3f} |")
lines += ['', '## 実行条件と評価の意味', '',
 '- Docker Compose v5.5.0、専用Podmanエンジン。Windows上のDocker Desktopでの実行ではない。Docker Desktopの起動障害・復旧試行・CLIリセット拒否はevidenceに記録した。',
 '- Nemotron 3 Diarization公式GGUF Q8_0、NeMo-Speech.cpp v0.2.0 CPU。v3-offlineのチャンク推論。全音声へのfull-attentionを指定する`--offline`とは異なる。',
 '- TitaNet-Largeは公式NeMo `EncDecSpeakerLabelModel.get_embedding()`。登録音声は単独の紹介音声13〜27秒、テストは別生成の会話シーンから各人2発話。登録・照合が同一波形になる漏洩はない。',
 '- 16kHz mono。8人が1巡目で初登場し、別発話で逆順に再登場。重なり条件は後半の4組を1.5秒差で配置。BGM条件は音声全体RMS比-20dB。',
 '- 発話正答数は、既知の切出し区間内で最も多く活動した推定チャネルを用い、全16発話に対する最適な1対1対応を評価時のみ求めた値。実名を自動認証した値ではない。',
 '- フレーム一致率と重なりの指標は10ms・RMS>0.006の参照マスクによる診断値。人手で音声境界を検収した正式DERではない。台本・発話時刻は既存プロジェクトの記録を利用した。',
 '- 小標本・日本語TTSキャラクター音声の結果。実人間8名の会議、長時間番組、任意の環境・BGM・同時発話人数へ一般化しない。',
 '', '## 成果物', '',
 '- `inputs/*.wav`: 聴ける3条件の実験音声。`clips.json`: 切出し元・時刻・台本。',
 '- `results/*.rttm`: 実モデルの話者区間。`*.turns.json`と`*.metrics.json`: 評価の全記録。',
 '- `results/titanet-queries.json`: 16発話×8人の類似度。`named-turns.json`: 推定名付きの全発話。',
 '- `manifest.json`: 固定条件、入力/モデルSHA256、モデルrevision。`evidence/*.log`: 実行ログ。']
if (root/'results/native-clean-summary.json').exists():
    n=read('results/native-clean-summary.json')
    lines += ['', '## 公式Transformers FP32での確認', '',
      f"同じBGMなし72.84秒音声を公式Transformers 5.19.0 / CPU FP32で実行し、{n['detected_speakers']}話者を検出、発話正答数は{n['correct_turns']}/{n['total_turns']}だった。ロードを含む実行時間は{n['wall_seconds_including_load']:.2f}秒。processorの標準後処理を使用しており、C++ Q8_0と設定が完全に同じ比較ではない。生出力・全発話・重みハッシュは`results/native-clean-*`に保存。"]
if (root/'results/transcript.json').exists():
    a=read('results/transcript.json')
    lines += ['',f"日本語ASRの追加試験: Whisper small CPU int8で{len(a['segments'])}区間を文字起こしした（モデル取得・ロードを含む{a['wall_seconds']:.2f}秒）。`results/transcript.md`と`transcript.json`に保存。文字起こし精度は採点していない。名前は推定クラスタに対する照合結果で、正しい名前の保証は上の実測範囲に限る。"]
lines += ['', '## 再実行', '', '専用Podman環境を起動した状態で、Windows PowerShellから:', '',
 '```powershell', 'wsl -d podman-neon-yoru-diarization -u root --exec /bin/bash /mnt/c/Prj/pm/2026-10-10-neon-yoru-diarization/build-compose.sh',
 'wsl -d podman-neon-yoru-diarization -u root --exec /bin/bash /mnt/c/Prj/pm/2026-10-10-neon-yoru-diarization/run-compose.sh', '```', '',
 'Docker Engineが動いている別環境では、同フォルダで`docker compose build`と`docker compose run --rm experiment`。元音声のマウント先はcompose.yamlに定義。再実行は入力と実験出力を再生成する。',
 '', '## 公式資料', '',
 '- [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)',
 '- [NVIDIAの解説](https://huggingface.co/blog/nvidia/nemotron-diarization)',
 '- [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)',
 '- [TitaNet-Large](https://huggingface.co/nvidia/speakerverification_en_titanet_large)', '']
(root/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print('REPORT_CREATED')
