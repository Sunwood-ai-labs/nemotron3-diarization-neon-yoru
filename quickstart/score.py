# 採点: RTTMと正解JSONから「検出した人数」と「正しく区別できた発話」を数える。
# 使い方: python score.py results/meeting.rttm inputs/meeting.truth.json
import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

rttm, truth_path = sys.argv[1], sys.argv[2]
truth = json.load(open(truth_path, encoding='utf-8'))
names = sorted({t['speaker'] for t in truth})
frames = round(max(t['end'] for t in truth) * 100) + 1      # 10ms単位

pred = np.zeros((frames, 8), dtype=bool)                    # 推定話者は最大8人
for line in open(rttm, encoding='utf-8'):
    p = line.split()
    if not p:
        continue
    start, dur = float(p[3]), float(p[4])
    ch = int(p[7].split('_')[-1]) - 1
    pred[round(start * 100):min(frames, round((start + dur) * 100)), ch] = True

ref = np.zeros((frames, len(names)), dtype=bool)
for t in truth:
    ref[round(t['start'] * 100):round(t['end'] * 100), names.index(t['speaker'])] = True

# 推定チャネルと本人の最適な1対1対応
rows, cols = linear_sum_assignment(-(ref.astype(int).T @ pred.astype(int)))
mapping = {c: names[r] for r, c in zip(rows, cols)}

correct = 0
for t in truth:
    counts = pred[round(t['start'] * 100):round(t['end'] * 100)].sum(axis=0)
    guess = mapping.get(int(counts.argmax())) if counts.max() > 0 else None
    correct += guess == t['speaker']

print('検出した人数:', int((pred.sum(axis=0) > 0).sum()), '/', len(names))
print('正しく区別できた発話:', correct, '/', len(truth))
