import time,json,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor,AutoModelForAudioFrameClassification
root=Path(__file__).resolve().parent
torch.set_num_threads(8)
path=root/'models/nemotron-native'
start=time.perf_counter()
processor=AutoProcessor.from_pretrained(path)
model=AutoModelForAudioFrameClassification.from_pretrained(path).eval()
audio,sr=sf.read(root/'inputs/clean.wav',dtype='float32')
inputs=processor(audio,sampling_rate=sr,return_tensors='pt')
with torch.inference_mode():
    logits=model(**inputs).logits
segments=processor.extract_speaker_dict(logits,inputs.attention_mask)[0]
probs=logits.detach().cpu().numpy()[0]
np.save(root/'results/native-clean-logits.npy',probs)
(root/'results/native-clean-segments.json').write_text(json.dumps(segments,indent=2),encoding='utf8')
with (root/'results/native-clean.rttm').open('w') as out:
    for s in segments:
        out.write(f"SPEAKER native-clean 1 {s['Start']:.3f} {s['End']-s['Start']:.3f} <NA> <NA> speaker_{int(s['Speaker'])+1} <NA> <NA>\n")
ref=np.load(root/'inputs/clean.reference.npy')
pred=np.zeros_like(ref)
for s in segments:
    pred[round(s['Start']*100):min(len(ref),round(s['End']*100)),int(s['Speaker'])]=True
from scipy.optimize import linear_sum_assignment
rows,cols=linear_sum_assignment(-(ref.astype(int).T@pred.astype(int)))
mapping={str(c):f'{r+1:02}' for r,c in zip(rows,cols)}
placements=json.loads((root/'inputs/clean.placements.json').read_text())
turns=[]
for p in placements:
    a=round(p['start']*100);b=min(len(ref),round(p['end']*100));r=int(p['speaker'])-1
    scores=pred[a:b][ref[a:b,r]].sum(0)
    c=int(scores.argmax()) if scores.max()>0 else None
    guess=mapping[str(c)] if c is not None else None
    turns.append({**p,'channel':c,'mapped':guess,'correct':guess==p['speaker']})
result={'runtime':'Transformers 5.19.0 CPU float32','model_id':'nvidia/Nemotron-3-Diarization','revision':'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
 'weight_sha256':hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest(),
 'detected_speakers':int((pred.sum(0)>0).sum()),'correct_turns':sum(t['correct'] for t in turns),'total_turns':len(turns),
 'wall_seconds_including_load':time.perf_counter()-start,'mapping':mapping,'logit_shape':list(probs.shape),
 'postprocessing':'processor.extract_speaker_dict defaults; distinct from C++ baseline postprocessing'}
(root/'results/native-clean-summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
(root/'results/native-clean-turns.json').write_text(json.dumps(turns,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result),flush=True)
