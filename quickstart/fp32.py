# 量子化なしのFP32版（Transformers）で話者分離し、RTTMに書き出す。
# 実験の experiment/native.py から要点を抜き出したもの。短くした形では未実行。
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

model_id = 'nvidia/Nemotron-3-Diarization'
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id).eval()

audio, sr = sf.read('inputs/meeting.wav', dtype='float32')
inputs = processor(audio, sampling_rate=sr, return_tensors='pt')
with torch.inference_mode():
    logits = model(**inputs).logits
segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]

with open('results/meeting-fp32.rttm', 'w') as out:
    for s in segments:
        out.write(f"SPEAKER meeting 1 {s['Start']:.3f} {s['End'] - s['Start']:.3f} "
                  f"<NA> <NA> speaker_{int(s['Speaker']) + 1} <NA> <NA>\n")
