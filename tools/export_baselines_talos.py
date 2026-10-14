#!/usr/bin/env python3
"""Read-only export of the five historical public-answer baseline caches."""
import io,json,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import hashlib,io,json,math,sys,tarfile
from huggingface_hub import hf_hub_download
H=Path('/home/users/yzhang92')
D=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return [json.loads(l) for l in f if l.strip()]
files={};sources={};banks={}
def put(n,b):files[n]=b

def source(p):
 p=Path(p);b=p.read_bytes();sources[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return p
base=H/'overrefusal_extension_20260917'
raw={k:rows(source(base/'data'/f'{k}.jsonl')) for k in ('clean1000','mixed2000')}
assert D(raw['clean1000'])=='2db72fa9d8af60c1560ed89dd15283b72e397b6da6cdc0bb5004fb9182e05894'
assert D(raw['mixed2000'])=='9794d3290488e5bc593f309c56f10df1b087896cc124adcbc8ffb1ca60ed6820'
mixed_map={r['pair_id']:r for r in raw['mixed2000']}
assert all(dict(r,comparison_source=None)==dict(mixed_map[r['pair_id']],comparison_source=None) for r in raw['clean1000'])
for k,v in raw.items():put(f'baseline_{k}.jsonl',(base/'data'/f'{k}.jsonl').read_bytes())
roots={'qwen3_06b':base,'qwen3_17b':H/'qwen3_17b_suite_20260919','qwen3_8b':base,'r1_7b':H/'r1_7b_suite_20260920'}
encs={};refs={};models={}
for model,root in roots.items():
 enc=load(source(root/'reference'/model/'encoded.json'));ref=rows(source(root/'reference'/model/'scores.jsonl'))
 assert len(enc)==len(ref)==2000
 entry=load(source(root/'state'/f'{model}.download.json'))
 models[model]={'path':entry['snapshot'],'model':entry['model'],'revision':entry['revision'],'files':entry['files'],'family':model}
 encs[model]=enc;refs[model]={r['pair_id']:r for r in ref}
assert encs['qwen3_06b']==encs['qwen3_17b']==encs['qwen3_8b']
for family,key in [('qwen','qwen3_17b'),('r1','r1_7b')]:
 put(f'baseline_{family}.encoded.json',(json.dumps(encs[key],ensure_ascii=False,indent=2)+'\n').encode())
root=H/'safety_pair_v3_20260909/curated_final_v4c/training_15b'
r15=rows(source(root/'reference.jsonl'));seq=rows(source(root/'standard_dpo_sequence_reference_20260912.jsonl'))
r15m={r['pair_id']:r for r in r15};seqm={r['pair_id']:r for r in seq}
clean=rows(source(H/'safety_harmful_compare_20260916/reference/clean1000/records.jsonl'))
cleanm={r['pair_id']:r for r in clean};assert len(cleanm)==1000
assert len(r15m)==len(seqm)==2000
for r in r15:
 for side in ('positive','negative'):
  a=r[side];b=encs['r1_7b'][r['pair_id']][side]
  assert (a['ids'],a['start'],a['target_tokens'],a['serialized_sha256'])==(b['ids'],b['start'],b['target_tokens'],b['serialized_sha256'])
  assert seqm[r['pair_id']][side+'_serialized_sha256']==a['serialized_sha256']
for r in clean:
 assert all(r[s]['ids']==r15m[r['pair_id']][s]['ids'] and r[s]['start']==r15m[r['pair_id']][s]['start'] for s in ('positive','negative'))
models['r1_15b']={'model':'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B','revision':'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562','family':'r1_15b'}
p=Path(hf_hub_download(models['r1_15b']['model'],filename='config.json',revision=models['r1_15b']['revision'],local_files_only=True)).parent
models['r1_15b'].update(path=str(p),files={f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in p.iterdir() if f.is_file()})
for model in models:
 for k,data in raw.items():
  key=f'{model}_{k}';family='r1' if model.startswith('r1') else 'qwen';enc=encs['r1_7b' if family=='r1' else 'qwen3_17b'];norm=[]
  for row in data:
   rid=row['pair_id'];assert rid in enc
   if model=='r1_15b':
    orig=cleanm[rid] if k=='clean1000' else seqm[rid]
   else:orig=refs[model][rid]
   assert all(math.isfinite(orig[s+'_sum']) for s in ('positive','negative'))
   if model!='r1_15b':assert orig['row_sha256'] in (D(row),D(mixed_map[rid]))
   if model=='r1_15b' and k=='mixed2000':assert orig['pair_sha256']==D(row)
   norm.append({'pair_id':rid,'row_sha256':D(row),'scores':{'positive':orig['positive_sum'],'negative':orig['negative_sum']},'original_reference_record_sha256':D(orig),'original_run_fingerprint':orig['run_fingerprint']})
  name=f'baseline_{key}.reference.jsonl';put(name,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in norm).encode())
  banks[key]={'rows_file':f'baseline_{k}.jsonl','encoding_file':f'baseline_{family}.encoded.json','reference_file':name,'rows':len(data),'harmful':1000,'benign':0 if k=='clean1000' else 1000,'rows_digest':D(data),'encoding_digest':D(enc),'reference_digest':D(norm),'reference_model':model,'reference_conversion':'lossless sequence sums; row metadata rebound to identical frozen public-answer bank; original record hashes retained'}
manifest={'status':'verified_export','banks':banks,'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()},'original_sources':sources,'source':'Historical clean1000/mixed2000; no sampling or regeneration of retired stage2 banks','reference_model_files':models}
put('baseline_manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
put('baseline_model_identities.json',(json.dumps(models,indent=2)+'\n').encode())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tf:
 for name,b in files.items():
  t=tarfile.TarInfo(name);t.size=len(b);t.mtime=0;tf.addfile(t,io.BytesIO(b))
'''
def main():
 with tempfile.TemporaryDirectory() as temp:
  p=Path(temp)/'baseline_export.tar.gz'
  with p.open('wb') as f:
   subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','yzhang92@talos.cilab.gmu.edu','/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python','-'],input=REMOTE.encode(),stdout=f,check=True)
  with tarfile.open(p,'r:gz') as tf:
   for t in tf.getmembers():
    assert t.isfile() and Path(t.name).name==t.name
    raw=tf.extractfile(t).read();target=ROOT/('provenance' if t.name=='baseline_model_identities.json' else 'data')/t.name
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)
 print('Exported five frozen baseline caches without GPU work.')
if __name__=='__main__':main()
