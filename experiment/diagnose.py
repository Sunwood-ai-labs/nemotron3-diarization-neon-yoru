"""Additional causal probes; preserve all original experiment outputs."""
import hashlib, json, time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'diagnosis'
OUT.mkdir(exist_ok=True)
SR = 16000

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

def read(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    return resample_poly(x.mean(1), SR, sr).astype('float32')

def normalized(x):
    return x * min(.08/(np.sqrt(np.mean(x*x))+1e-9), .65/(np.max(np.abs(x))+1e-9))

def assemble(name, items):
    cursor = .7
    placements = []
    for sp, x in items:
        placements.append({'speaker': sp, 'start': cursor, 'end': cursor+len(x)/SR})
        cursor += len(x)/SR+.5
    x = np.zeros(round((cursor+.7)*SR), dtype='float32')
    ref = np.zeros((int(np.ceil(len(x)/160)),8), dtype=bool)
    for p, (_, y) in zip(placements, items):
        start = round(p['start']*SR)
        x[start:start+len(y)] = y
        n = len(y)//160
        m = np.sqrt(np.mean(y[:n*160].reshape(n,160)**2,axis=1)) > .006
        frame = round(p['start']*100)
        ref[frame:frame+n, int(p['speaker'])-1] = m
    sf.write(OUT/f'{name}.wav', x, SR)
    np.save(OUT/f'{name}.reference.npy', ref)
    dump(f'{name}.placements.json', placements)
    return x, ref, placements

torch.set_num_threads(8)
processor = AutoProcessor.from_pretrained(ROOT/'models/nemotron-native')
model = AutoModelForAudioFrameClassification.from_pretrained(ROOT/'models/nemotron-native').eval()

def score(name, probs, ref, placements, threshold):
    segments = processor.extract_speaker_dict(torch.from_numpy(probs)[None], threshold=threshold)[0]
    pred = np.zeros_like(ref)
    for s in segments:
        pred[round(s['Start']*100):min(len(ref),round(s['End']*100)),int(s['Speaker'])] = True
    rows, cols = linear_sum_assignment(-(ref.astype(int).T@pred.astype(int)))
    mapping = {int(c):f'{r+1:02}' for r,c in zip(rows, cols)}
    turns = []
    for p in placements:
        a, b = round(p['start']*100), min(len(ref),round(p['end']*100))
        counts = pred[a:b][ref[a:b,int(p['speaker'])-1]].sum(0)
        c = int(counts.argmax()) if counts.max()>0 else None
        turns.append({**p,'channel':c,'mapped':mapping.get(c),'correct':mapping.get(c)==p['speaker']})
    dump(f'{name}.threshold-{threshold}.turns.json', turns)
    dump(f'{name}.threshold-{threshold}.segments.json', segments)
    return {'name':name,'threshold':threshold,'detected_speakers':int((pred.sum(0)>0).sum()),
            'correct_turns':sum(t['correct'] for t in turns),'total_turns':len(turns),
            'speech_fraction_with_multiple_predicted_speakers':float((pred.sum(1)[ref.any(1)]>1).mean()),
            'logit_min':float(probs.min()),'logit_max':float(probs.max())}

cells=[]
baseline=read(ROOT/'inputs/clean.wav')
cells.append(('conversation_4s',baseline,np.load(ROOT/'inputs/clean.reference.npy'),
              json.loads((ROOT/'inputs/clean.placements.json').read_text()),
              np.load(ROOT/'results/native-clean-logits.npy')))

# Same original conversational utterances, but keep their complete recorded spans.
clips=json.loads((ROOT/'clips.json').read_text())
lookup={m['clip']:m for m in clips}
order=[p['clip'] for p in cells[0][3]]
timing=json.loads(Path('/source/src/data/stream-timing.json').read_text())
items=[]
spans=[]
for clip in order:
    m=lookup[clip]
    y=read(Path(m['source']))
    end=timing[m['scene']]['lines'][m['line']]['end']-.1
    y=normalized(y[round(m['source_start']*SR):round(end*SR)])
    items.append((m['speaker'],y))
    spans.append({'clip':clip,'speaker':m['speaker'],'source':m['source'],
                  'start':m['source_start'],'end':end,'duration':len(y)/SR})
dump('conversation-full-source-spans.json',spans)
cells.append(('conversation_full',*assemble('conversation_full',items),None))

# Controlled duration contrast: identical introduction source/prefix per speaker.
# The second round repeats the same waveform. This is a consistency probe,
# not independent utterance or enrollment accuracy.
for duration in (4,12):
    ids=[f'{i:02}' for i in range(1,9)]
    voices={sp:normalized(read(Path('/source/public/voice')/f'{sp}.wav')[:duration*SR]) for sp in ids}
    items=[(sp,voices[sp]) for sp in ids+list(reversed(ids))]
    name=f'introduction_{duration}s'
    cells.append((name,*assemble(name,items),None))

dump('contract.json',{'model':'nvidia/Nemotron-3-Diarization','revision':'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
    'weight_sha256':hashlib.sha256((ROOT/'models/nemotron-native/model.safetensors').read_bytes()).hexdigest(),
    'runtime':'Transformers 5.19.0 CPU FP32','sample_rate':SR,'thresholds':[.25,.35,.5,.65],
    'planned_cells':[{'name':n,'duration_seconds':len(x)/SR,'turns':len(p)} for n,x,r,p,raw in cells],
    'limits':['energy reference, not hand-annotated DER','full conversation inherits project timings',
              'introduction second round repeats same waveform','threshold grid is diagnostic, not held-out calibration']})
summary=[]
for name,x,ref,placements,probs in cells:
    start=time.perf_counter()
    if probs is None:
        inputs=processor(x,sampling_rate=SR,return_tensors='pt')
        with torch.inference_mode():
            probs=model(**inputs).logits.cpu().numpy()[0]
        np.save(OUT/f'{name}.probabilities.npy',probs)
    for threshold in (.25,.35,.5,.65):
        result=score(name,probs,ref,placements,threshold)
        result.update({'audio_seconds':len(x)/SR,'inference_reused':name=='conversation_4s',
                       'wall_seconds_for_cell':time.perf_counter()-start})
        summary.append(result)
        print(json.dumps(result),flush=True)
    dump('summary.json',summary)
print('DIAGNOSIS_COMPLETED',flush=True)
