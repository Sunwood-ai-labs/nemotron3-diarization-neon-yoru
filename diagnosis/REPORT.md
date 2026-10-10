# Q8・発話長・判定閾値の切り分け

2026-10-10。Docker Compose経由の専用Podman環境で、同じ公式Nemotron 3のTransformers FP32 / CPUを追加実行した。終了コード0、ログ末尾DIAGNOSIS_COMPLETEDを確認。

## 確認した結果

| 入力 | 音声長 | 話者検出数（閾値0.5） | 正答発話 |
|---|---:|---:|---:|
| 既存の会話3〜4秒抜粋 | 72.84秒 | 5 | 7/16 |
| 同じ16発話を最後まで残した会話 | 109.52秒 | 3 | 5/16 |
| 8人の自己紹介の先頭4秒 | 73.40秒 | 2 | 4/16 |
| 同じ自己紹介の先頭12秒 | 201.40秒 | 3 | 6/16 |

自己紹介の2巡目は同じ波形の逆順再提示。別発話への一般化を測ったものではなく、入力長の対照試験。会話の完全区間は既存プロジェクトの時刻を利用しており、人手で新しく確定した区間ではない。全条件16kHz mono、BGM・重なりなし、同じFP32重みと同じ推論・後処理。音量は同じRMS/ピーク上限で正規化したが、長さ変更に伴って各波形の正規化係数も変わる。

閾値0.25/0.35/0.5/0.65も診断した。会話抜粋は6/7/7/5発話正解、完全区間は全て5発話正解、自己紹介4秒は全て4、12秒は全て6。閾値を下げても8人の識別は回復しなかった。低閾値では本来重なりのない音声へ同時発話を付ける例も増えた。これは検証データで探索した診断であり、運用閾値の校正ではない。

## 原因について言える範囲

- Q8を使わないFP32でも話者が統合されるため、今回の低精度をQ8だけでは説明できない。
- 発話を完全な長さへ戻しても改善せず、自己紹介を12秒にしても8人を区別できない。短い抜粋だけが原因という説明にも根拠が不足する。
- 単純な話者活動閾値の変更で回復しなかった。
- 主なボトルネックは、この8声に対するNemotronの話者分離処理。TitaNetは正しい区間を渡した別の試験で13/16正解しており、後段の照合だけで今回の失敗を説明できない。
- 学習対象と日本語TTS音声の違い、声質の類似、推論実装・設定の影響は未分離。モデル全般の性能不良や特定の学習原因は断定しない。
- C++ Q8とTransformers FP32は実装・推論方式・後処理が違う。量子化だけの影響を測った対照試験ではない。
- GPUの実測比較はしていない。このWindows環境のWin32_VideoController列挙ではIntel UHD Graphicsのみが表示された。GPUを使えば精度が改善するという証拠はない。

## 記録

summary.json、contract.json、各入力wav/reference/placements、各閾値のsegments/turns、../evidence/diagnose-model.log、../evidence/diagnose-compose.exit.txt。
元の実験出力は変更していない。summary.jsonのprobability_min/maxと*.probabilities.npyは実際にはモデルのraw logitsであり、確率ではない（診断スクリプトの表示名を後からlogit_min/maxへ修正）。評価は公式processor.extract_speaker_dictがlogitsに対して行う後処理を使用。

## PM対応の訂正と公式NeMo追試

原因の質問に対し失敗箇所と候補の説明を繰り返して終了し、原因調査を継続する対応が不足していた。初回実験の実行完了と、その後に依頼された原因調査の完了を区別する。後者は未完了。

2026-10-10に同じclean.wavと公式.nemo重みを使い、NVIDIA-NeMo/Speechの公式実装で追加確認した。最初のNeMo 3.0.0はRoPE方式をサポートせず復元時に失敗。公式ソースe4a34fd25f895ecd7f64f0e56bc288165e078da9と、そのpyproject指定のlhotse==2.0.0a6を専用コンテナへ適用して復旧した。これらの実行失敗は元のC++/Transformersによる精度計測とは別の事象。ホストや他案件のPython依存は変更していない。

公式NeMo経路の最終結果は**5話者検出、7/16発話正解**。全パラメータtorch.float32、確率出力7284×8。公式高遅延構成（chunk_len340、right_context40、fifo40、spkcache264、update_period300）で実行し、確率0.5以上を用いた同じ発話単位の最適1対1対応評価を行った。Q8との量子化だけの比較には依然としてならない。終了コード0とNEMO_PROBE_COMPLETEDを確認した。

証拠: nemo-clean-summary.json、nemo-clean-probabilities.npy、nemo-clean-turns.json、../evidence/nemo-probe.log、../evidence/nemo-probe-compose.exit.txt。失敗ログはnemo-probe-nemo3-incompatible.logとnemo-probe-lhotse-incompatible.logに保存。

この入力での話者統合はC++ Q8・Transformers FP32・公式NeMo FP32で再現する。Q8だけ、BGMだけ、Transformers経路だけで説明する根拠はない。一方、正解区間の独立検収、正常な対照入力による経路検収、量子化だけを変えた比較、音声特性と学習対象の不一致の特定は未実施・未解決として保持する。
