import json, hashlib, platform, os
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
SRC = Path(os.environ.get('YORU_SOURCE','C:/Prj/YORU'))
SR = 16000
def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    return resample_poly(x.mean(axis=1), SR, sr).astype('float32')
def normalize(x):
    return x * min(.08 / (np.sqrt(np.mean(x*x))+1e-9), .65/(np.max(np.abs(x))+1e-9))
def mask(x):
    # Diagnostic reference only: energy-based, not human speech annotation.
    n = len(x)//160
    rms = np.sqrt(np.mean(x[:n*160].reshape(n,160)**2,axis=1))
    return rms > .006

for d in ['inputs','clips','enrollment','results','evidence']:
    (ROOT/d).mkdir(exist_ok=True)
cast = json.loads((SRC/'src/data/cast.json').read_text(encoding='utf-8'))
script = json.loads((SRC/'src/data/stream.json').read_text(encoding='utf-8'))
timing = json.loads((SRC/'src/data/stream-timing.json').read_text(encoding='utf-8'))
candidates = {sp: [] for sp in cast}
sources = []
for scene in script['scenes']:
    p = SRC/'public/stream'/f"{scene['id']}.wav"
    sources.append({'path': str(p), 'sha256': sha(p)})
    audio = load(p)
    assert len(scene['lines']) == len(timing[scene['id']]['lines'])
    for i,(line,t) in enumerate(zip(scene['lines'], timing[scene['id']]['lines'])):
        if t['end']-t['start'] >= 3.5:
            start = t['start'] + .1
            end = min(t['end']-.1, start+4)
            x = normalize(audio[round(start*SR):round(end*SR)])
            candidates[line['sp']].append((x, {'speaker':line['sp'], 'scene':scene['id'],
                'line':i, 'source_start':start, 'source_end':end,'text':line['text'],'source':str(p)}))
clips=[]
for sp in cast:
    assert len(candidates[sp]) >= 2, sp
    enrollment = SRC/'public/voice'/f'{sp}.wav'
    x = normalize(load(enrollment))
    sf.write(ROOT/'enrollment'/f'{sp}.wav',x,SR)
    sources.append({'path':str(enrollment),'sha256':sha(enrollment)})
    for turn,(x,meta) in enumerate(candidates[sp][:2]):
        filename=f'{sp}_{turn}.wav'
        sf.write(ROOT/'clips'/filename,x,SR)
        meta.update({'round':turn,'clip':filename,'duration':len(x)/SR})
        clips.append(meta)
dump(ROOT/'clips.json',clips)
order = [f'{sp}_{turn}.wav' for turn in range(2) for sp in (list(cast) if turn==0 else list(reversed(cast)))]
meta_by_clip={m['clip']:m for m in clips}
audio_by_clip={n:load(ROOT/'clips'/n) for n in order}
def assemble(overlap=False):
    placements=[]; cursor=.7
    for i,n in enumerate(order):
        if overlap and i>=8 and i%2==1:
            start=placements[-1]['start']+1.5
        else:
            start=cursor
        end=start+len(audio_by_clip[n])/SR
        placements.append({**meta_by_clip[n],'start':start,'end':end})
        cursor=max(cursor,end)+.5
    out=np.zeros(round((cursor+.7)*SR),dtype='float32')
    ref=np.zeros((int(np.ceil(len(out)/160)),8),dtype=bool)
    for p in placements:
        x=audio_by_clip[p['clip']]; a=round(p['start']*SR)
        out[a:a+len(x)]+=x
        m=mask(x); f=round(p['start']*100)
        ref[f:f+len(m),int(p['speaker'])-1] |= m
    return out,ref,placements
clean,ref,placements=assemble()
overlap,oref,oplace=assemble(True)
bgm=normalize(load(SRC/'public/audio/stream-bgm.wav'))[:len(clean)]
bgm *= np.sqrt(np.mean(clean*clean))/(np.sqrt(np.mean(bgm*bgm))+1e-9)*.1
cells=[('clean',clean,ref,placements),('overlap',overlap,oref,oplace),('bgm_minus20db',clean+bgm,ref,placements)]
inputs=[]
for name,x,r,p in cells:
    wav=ROOT/'inputs'/f'{name}.wav'
    sf.write(wav,x,SR)
    np.save(ROOT/'inputs'/f'{name}.reference.npy',r)
    dump(ROOT/'inputs'/f'{name}.placements.json',p)
    inputs.append({'name':name,'duration_seconds':len(x)/SR,'sha256':sha(wav),
                   'turns':len(p),'speakers':8,'reference':'10ms energy mask RMS > 0.006 before mixing'})
manifest={'created_before_inference':True,'source_type':'existing NEON YORU Japanese Gemini TTS, not human meeting',
 'os':platform.platform(),'source_root':str(SRC),'sample_rate':SR,
 'model_id':'nvidia/Nemotron-3-Diarization','revision':'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
 'model_file':'Nemotron-3-Diarization.q8_0.gguf','model_sha256':sha(ROOT/'models/Nemotron-3-Diarization.q8_0.gguf'),
 'runtime':'NVIDIA NeMo-Speech.cpp v0.2.0 Linux x86_64 CPU via Docker Compose v5.5.0 on dedicated Podman engine','runtime_archive_sha256':sha(ROOT/'runtime/linux-release.tar.gz'),
 'settings':{'preset':'v3-offline','onset':.5,'offset':.5,'pad_onset':0,'pad_offset':0,'min_duration_on':.1,'min_duration_off':.1},
 'identity_model':'nvidia/speakerverification_en_titanet_large','identity_accept_similarity':.65,'identity_accept_top1_minus_top2':.05,
 'metrics':['optimal one-to-one speaker/channel mapping','majority speaker per known clip','energy-reference frame diagnostics','TitaNet independent enrollment top1 and conservative rejection'],
 'limitations':['line timings inherited from project and not newly hand-annotated','energy reference is not formal DER','Q8 CPU runtime, not FP32 NeMo','8 synthetic voices are not 8 real people'],
 'inputs':inputs,'sources':sources,'cast':cast}
dump(ROOT/'manifest.json',manifest)
print(json.dumps(inputs))
