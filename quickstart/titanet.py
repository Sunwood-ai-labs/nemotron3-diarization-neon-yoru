# TitaNet-Largeで、切り出した音声が登録済みの誰に近いかを調べる。
# 実験の experiment/experiment.py から要点を抜き出したもの。短くした形では未実行。
from pathlib import Path
import numpy as np
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel

speaker = EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large').eval()

def emb(path):
    with torch.inference_mode():
        e = speaker.get_embedding(str(path)).detach().cpu().numpy().reshape(-1)
    return e / (np.linalg.norm(e) + 1e-12)

ids = sorted(p.stem for p in Path('enrollment').glob('*.wav'))
enroll = np.stack([emb(f'enrollment/{s}.wav') for s in ids])

scores = emb('clips/unknown.wav') @ enroll.T               # cosine類似度
rank = np.argsort(-scores)
first, second = rank[0], rank[1]
accepted = scores[first] >= 0.65 and scores[first] - scores[second] >= 0.05
print(ids[first] if accepted else '不明', float(scores[first]))
