import json, time, subprocess, importlib.metadata, hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment
from prepare import dump, ROOT, SR

# Importing prepare regenerates and seals inputs before any inference.
def predict_frames(path, frames):
    pred=np.zeros((frames,8),dtype=bool)
    for line in path.read_text().splitlines():
        p=line.split()
        if not p: continue
        assert p[0]=='SPEAKER',line
        start=float(p[3]); end=start+float(p[4])
        speaker=int(p[7].split('_')[-1])-1
        # Official app/diarize.cpp v0.2.0 renders segment.speaker + 1 in RTTM.
        pred[max(0,round(start*100)):min(frames,round(end*100)),speaker]=True
    return pred

manifest=json.loads((ROOT/'manifest.json').read_text())
versions={m:importlib.metadata.version(m) for m in ['torch','torchaudio','nemo-toolkit','numpy','faster-whisper']}
dump(ROOT/'evidence/package-versions.json',versions)
subprocess.run(['nemo-speech','doctor'],stdout=(ROOT/'evidence/doctor.txt').open('w'),stderr=subprocess.STDOUT,check=True)
summary=[]
for cell in manifest['inputs']:
    name=cell['name']; out=ROOT/'results'/f'{name}.rttm'
    cmd=['nemo-speech','diarize',str(ROOT/'inputs'/f'{name}.wav'),'--model',str(ROOT/'models/Nemotron-3-Diarization.q8_0.gguf'),
        '--device','cpu','--preset','v3-offline','--format','rttm','--recording-id',name,'--output',str(out),
        '--onset','0.5','--offset','0.5','--pad-onset','0','--pad-offset','0','--min-duration-on','0.1','--min-duration-off','0.1']
    start=time.perf_counter()
    with (ROOT/'evidence'/f'{name}.log').open('w') as log:
        run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-start
    dump(ROOT/'evidence'/f'{name}.run.json',{'command':cmd,'exit_code':run.returncode,'wall_seconds':elapsed})
    assert run.returncode==0,(name,run.returncode)
    ref=np.load(ROOT/'inputs'/f'{name}.reference.npy')
    pred=predict_frames(out,len(ref))
    score=ref.astype(int).T @ pred.astype(int)
    rows,cols=linear_sum_assignment(-score)
    mapped=np.zeros_like(pred)
    for r,c in zip(rows,cols): mapped[:,r]=pred[:,c]
    mapping={str(c):f'{r+1:02}' for r,c in zip(rows,cols)}
    placements=json.loads((ROOT/'inputs'/f'{name}.placements.json').read_text())
    turns=[]
    for p in placements:
        a=round(p['start']*100);b=min(len(ref),round(p['end']*100));r=int(p['speaker'])-1
        keep=ref[a:b,r]
        counts=pred[a:b][keep].sum(axis=0)
        c=int(counts.argmax()) if counts.max()>0 else None
        guess=mapping[str(c)] if c is not None else None
        turns.append({**p,'predicted_channel':c,'mapped_speaker':guess,'correct':guess==p['speaker'],
                      'true_speaker_active_recall':float(mapped[a:b,r][keep].mean()) if keep.any() else None})
    exclusive=ref.sum(1)==1
    # Correct only when the entire predicted active set equals the true speaker.
    exact=np.all(mapped==ref,axis=1)
    ov=ref.sum(1)>=2; pov=pred.sum(1)>=2
    res={'name':name,'audio_seconds':cell['duration_seconds'],'wall_seconds':elapsed,'real_time_factor':elapsed/cell['duration_seconds'],
         'detected_speaker_channels':int((pred.sum(0)>0).sum()),'mapping_channel_to_cast':mapping,
         'turns_correct':sum(t['correct'] for t in turns),'turns_total':len(turns),
         'exclusive_energy_frame_exact_accuracy':float(exact[exclusive].mean()),
         'energy_speech_recall':float((mapped & ref).sum()/ref.sum()),
         'overlap_precision':float((ov & pov).sum()/pov.sum()) if pov.any() else None,
         'overlap_recall':float((ov & pov).sum()/ov.sum()) if ov.any() else None,
         'reference_overlap_seconds':float(ov.sum()/100),'predicted_overlap_seconds':float(pov.sum()/100)}
    dump(ROOT/'results'/f'{name}.turns.json',turns)
    dump(ROOT/'results'/f'{name}.metrics.json',res)
    summary.append(res); print('DIARIZATION',json.dumps(res),flush=True)
dump(ROOT/'results/diarization-summary.json',summary)

import torch
torch.set_num_threads(8)
from nemo.collections.asr.models import EncDecSpeakerLabelModel
start=time.perf_counter()
model_path=ROOT/'models/titanet/speakerverification_en_titanet_large.nemo'
speaker=EncDecSpeakerLabelModel.restore_from(str(model_path),map_location='cpu').eval()
def emb(path):
    with torch.inference_mode():
        e=speaker.get_embedding(str(path)).detach().cpu().numpy().reshape(-1)
    return e/(np.linalg.norm(e)+1e-12)
cast=manifest['cast']; ids=list(cast)
enroll=np.stack([emb(ROOT/'enrollment'/f'{s}.wav') for s in ids])
clips=json.loads((ROOT/'clips.json').read_text())
queries=np.stack([emb(ROOT/'clips'/m['clip']) for m in clips])
sim=queries@enroll.T
rows=[]
for m,scores in zip(clips,sim):
    ranked=np.argsort(-scores); first,second=ranked[:2]
    accepted=float(scores[first])>=.65 and float(scores[first]-scores[second])>=.05
    # Hold out the true identity to exercise the unknown-person decision.
    loo=scores.copy();loo[ids.index(m['speaker'])]=-2
    lr=np.argsort(-loo);unknown_rejected=not(float(loo[lr[0]])>=.65 and float(loo[lr[0]]-loo[lr[1]])>=.05)
    rows.append({**m,'top1':ids[first],'top1_name':cast[ids[first]]['name'],'top1_cosine':float(scores[first]),
        'top2':ids[second],'top2_cosine':float(scores[second]),'margin':float(scores[first]-scores[second]),
        'correct_closed_set':ids[first]==m['speaker'],'accepted':accepted,'unknown_holdout_rejected':unknown_rejected,
        'all_scores':{s:float(v) for s,v in zip(ids,scores)}})
dump(ROOT/'results/titanet-queries.json',rows)
np.savez(ROOT/'results/titanet-embeddings.npz',enrollment=enroll,query=queries,similarity=sim)
ident={'model_id':manifest['identity_model'],'revision':(ROOT/'evidence/titanet-revision.txt').read_text().strip(),
 'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-start,
 'queries':len(rows),'closed_set_correct':sum(r['correct_closed_set'] for r in rows),
 'accepted':sum(r['accepted'] for r in rows),'accepted_correct':sum(r['accepted'] and r['correct_closed_set'] for r in rows),
 'unknown_holdout_rejected':sum(r['unknown_holdout_rejected'] for r in rows),
 'thresholds':{'cosine':.65,'margin':.05},'thresholds_calibrated':False}
dump(ROOT/'results/titanet-summary.json',ident);print('TITANET',json.dumps(ident),flush=True)

# Name the diarization clusters without using their oracle mapping.
clean_audio,_=sf.read(ROOT/'inputs/clean.wav',dtype='float32')
ref=np.load(ROOT/'inputs/clean.reference.npy')
pred=predict_frames(ROOT/'results/clean.rttm',len(ref))
cluster_names={}
for c in range(8):
    keep=pred[:,c] & (pred.sum(1)==1)
    samples=np.repeat(keep,160)[:len(clean_audio)]
    x=clean_audio[samples]
    if len(x)<SR: continue
    p=ROOT/'results'/f'cluster_{c}.wav';sf.write(p,x,SR)
    scores=emb(p)@enroll.T;rank=np.argsort(-scores);a,b=rank[:2]
    accepted=float(scores[a])>=.65 and float(scores[a]-scores[b])>=.05
    cluster_names[str(c)]={'cast':ids[a] if accepted else None,'top1':ids[a],
        'name':cast[ids[a]]['name'] if accepted else '不明','cosine':float(scores[a]),'margin':float(scores[a]-scores[b])}
turns=json.loads((ROOT/'results/clean.turns.json').read_text())
for t in turns:
    t['identity']=cluster_names.get(str(t['predicted_channel']),{'name':'不明','cast':None})
    t['named_correct']=t['identity']['cast']==t['speaker']
dump(ROOT/'results/named-turns.json',turns)
dump(ROOT/'results/named-summary.json',{'clusters':cluster_names,'correct':sum(t['named_correct'] for t in turns),'total':len(turns)})
print('EXPERIMENT_COMPLETED',flush=True)
