import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

root=Path(__file__).resolve().parent
def dump(path,x):
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
revision=(root/'evidence/whisper-revision.txt').read_text().strip()
start=time.perf_counter()
model_path=root/'models/whisper-small'
model=WhisperModel(str(model_path),device='cpu',compute_type='int8',cpu_threads=8)
audio,sr=sf.read(root/'inputs/clean.wav',dtype='float32')
assert sr == 16000
segments,meta=model.transcribe(audio,language='ja',beam_size=5,word_timestamps=True,vad_filter=True)
named=json.loads((root/'results/named-summary.json').read_text())['clusters']
frames=int(np.ceil(72.84*100));activity=np.zeros((frames,8),bool)
for line in (root/'results/clean.rttm').read_text().splitlines():
    p=line.split();c=int(p[7].split('_')[-1])-1
    a=round(float(p[3])*100);b=round((float(p[3])+float(p[4]))*100)
    activity[a:b,c]=True
rows=[]
for s in segments:
    words=[]
    for w in s.words or []:
        a=max(0,round(w.start*100));b=min(frames,max(a+1,round(w.end*100)))
        counts=activity[a:b].sum(0);c=int(counts.argmax()) if counts.max()>0 else None
        identity=named.get(str(c),{'name':'不明','cast':None})
        words.append({'start':w.start,'end':w.end,'word':w.word,'probability':w.probability,'speaker_channel':c,'name':identity['name']})
    rows.append({'start':s.start,'end':s.end,'text':s.text,'words':words})
dump(root/'results/transcript.json',{'model':'Systran/faster-whisper-small','revision':revision,'language':meta.language,
 'compute_type':'int8','wall_seconds':time.perf_counter()-start,'segments':rows,'accuracy':'not scored; no independently hand-checked reference'})
lines=['# NEON YORU 実モデルによる名前付き文字起こし試作','','Whisper small / 日本語。名前はNemotronの推定区間とTitaNetの照合結果。正確性は未校正。','']
for s in rows:
    names=list(dict.fromkeys(w['name'] for w in s['words']))
    lines.append(f"{s['start']:.2f}–{s['end']:.2f}秒 | {' / '.join(names) or '不明'}: {s['text']}")
(root/'results/transcript.md').write_text('\n\n'.join(lines),encoding='utf8')
print('ASR_COMPLETED',len(rows),flush=True)
