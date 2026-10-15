#!/usr/bin/env python3
"""Read-only export of verified public-answer Wild harmful/mixed data; CPU tokenization only."""
import json,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import hashlib,io,json,sys,tarfile,collections
from transformers import AutoTokenizer
sys.path.insert(0,'/tmp/yzhang92/twostage-handoff-wild-v3-validation/vendor')
from safety_pair_v3_encoding import freeze_response,encode_response
D=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return [json.loads(l) for l in f if l.strip()]
files={};sources={}
def source(p):
 p=Path(p);b=p.read_bytes();sources[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return p
root=Path('/tmp/yzhang92/qwen17-benign-source-20261003/wild')
cfg=load(source(root/'config.json'));assert D(cfg)==load(root/'config.sha.json')['sha256']
data=rows(source(root/'data/mixed.jsonl'));assert D(data)==cfg['banks']['mixed']['sha256']=='0e4470454a06182772da55ae60e194c94a28dbc7e4b43922493e5f5800f2c059'
enc=load(source(root/'data/mixed.encoded.json'));assert D(enc)==cfg['banks']['mixed']['encoded_sha256']
refs=rows(source(root/'reference/mixed/scores.jsonl'));proof=load(source(root/'reference/mixed/complete.json'));assert D(refs)==proof['scores_sha256']
assembly=load(source(root/'data/assembly_summary.json'));assert assembly['primary_and_pair_quality_verified'] and assembly['whole_published_answers_preserved']
assert collections.Counter(r['task_type'] for r in data)=={'harmful':828,'benign':1000}
assert collections.Counter(r['group'] for r in data if r['task_type']=='benign')=={'vanilla_benign':498,'adversarial_benign':502}
hroot=Path('/tmp/yzhang92/qwen17-existing-rejected-wild-20260930')
harmful=rows(source(hroot/'data/dataset_rejected.jsonl'));lookup={r['pair_id']:r for r in data}
assert len(harmful)==828 and all(r==lookup[r['pair_id']] for r in harmful)
assert set(enc)==set(lookup)=={r['pair_id'] for r in refs}
assert all(r['row_sha256']==D(lookup[r['pair_id']]) for r in refs)
for r in data:
 if r['task_type']=='benign':
  assert r['source_evidence']['chosen']['dataset']=='allenai/wildjailbreak'
  assert r['source_evidence']['rejected']['dataset']=='allenai/wildguardmix'
files['wild_mixed.jsonl']=(root/'data/mixed.jsonl').read_bytes()
files['wild_mixed.qwen17.reference.jsonl']=(root/'reference/mixed/scores.jsonl').read_bytes()
files['wild_assembly_summary.json']=(root/'data/assembly_summary.json').read_bytes()
files['wild_pool_complete.json']=source(root/'pool_complete.json').read_bytes()
models=load('/tmp/yzhang92/twostage-handoff-wild-v3-validation/provenance/baseline_model_identities.json')
assets=load('/home/users/yzhang92/twostage_share_20261014/assets.json')
encoded={};checks={}
for key,entry in models.items():
 tok=AutoTokenizer.from_pretrained(assets['base_models'][key],local_files_only=True)
 modelenc={}
 for row in data:
  answer={}
  for side in ('positive','negative'):
   frozen=row[side];assert frozen['representation']=='final_text';text=frozen['original_text']
   if key.startswith('r1'):
    e=encode_response(tok,row['prompt'],freeze_response(text,'final_text'),max_length=8192)
   else:
    prefix=tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert prefix.endswith('<think>\n\n</think>\n\n')
    context=tok.encode(prefix,add_special_tokens=False);target=tok.encode(text,add_special_tokens=False)
    assert target and tok.eos_token_id not in target;target.append(tok.eos_token_id)
    e={'ids':context+target,'start':len(context),'target_tokens':len(target),'serialized_sha256':D([context,target]),'source_text_sha256':D(text)}
    assert e==enc[row['pair_id']][side]
   assert 0<e['start']<len(e['ids'])<=8192
   assert e['ids'][-1]==tok.eos_token_id
   answer[side]=e
  modelenc[row['pair_id']]=answer
 encoded[key]=modelenc;checks[key]={'model_revision':entry['revision'],'encoding_digest':D(modelenc),'max_tokens':max(len(e['ids']) for r in modelenc.values() for e in r.values()),'EOS_included':True,'no_truncation':True,'CPU_only':True}
assert encoded['qwen3_06b']==encoded['qwen3_17b']==encoded['qwen3_8b']
assert encoded['r1_15b']==encoded['r1_7b']
for family,key in [('qwen','qwen3_17b'),('r1','r1_15b')]:files['wild_'+family+'.encoded.json']=(json.dumps(encoded[key],ensure_ascii=False,indent=2)+'\n').encode()
banks={}
for key in models:
 family='r1' if key.startswith('r1') else 'qwen'
 for kind,values in [('harmful',harmful),('mixed',data)]:
  v={'rows_file':'harmful.jsonl' if kind=='harmful' else 'wild_mixed.jsonl','encoding_file':'wild_'+family+'.encoded.json','rows':len(values),'harmful':828,'benign':0 if kind=='harmful' else 1000,'rows_digest':D(values),'encoding_digest':D(encoded[key]),'reference_model':key,'reference_status':'provided' if key=='qwen3_17b' else 'compute_original_pi0_no_training_no_answer_sampling'}
  if key=='qwen3_17b':
   if kind=='harmful':
    original=rows(hroot/'reference/dataset_rejected/scores.jsonl');v.update(reference_file='harmful.reference.jsonl',reference_digest=D(original))
   else:v.update(reference_file='wild_mixed.qwen17.reference.jsonl',reference_digest=D(refs))
  banks[key+'_wild_'+kind]=v
manifest={'status':'verified_export','banks':banks,'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()},'original_sources':sources,'tokenizer_verification':checks,'source':'WildJailbreak train chosen + exact-prompt WildGuardTrain published rejected on both harmful and benign; screened original1000 benign reused; no new answers','original_qwen_reference_proof':proof,'new_reference_required':['qwen3_06b','qwen3_8b','r1_15b','r1_7b'],'note':'Model-specific reference sums must never be shared across model weights. Other four models compute frozen-pi0 scores once on mixed union, then subset harmful; reuse across seeds.'}
files['wild_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tf:
 for name,b in files.items():
  t=tarfile.TarInfo(name);t.size=len(b);t.mtime=0;tf.addfile(t,io.BytesIO(b))
'''
def main():
 with tempfile.TemporaryDirectory() as temp:
  p=Path(temp)/'export.tar.gz'
  with p.open('wb') as f:subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','yzhang92@talos.cilab.gmu.edu','/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python','-'],input=REMOTE.encode(),stdout=f,check=True)
  with tarfile.open(p,'r:gz') as tf:
   for item in tf.getmembers():
    assert item.isfile() and Path(item.name).name==item.name
    raw=tf.extractfile(item).read();target=ROOT/'data'/item.name
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)
 print('Exported verified all-Wild public pairs and CPU-verified encodings for five models; no GPU work.')
if __name__=='__main__':main()
