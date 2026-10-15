#!/usr/bin/env python3
"""Assemble1000 verified published Wild harmful pairs and frozen1000 benign; CPU only."""
import json,subprocess,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import collections,csv,hashlib,io,json,random,re,sys,tarfile,unicodedata
from transformers import AutoTokenizer
sys.path.insert(0,'/tmp/yzhang92/twostage-handoff-wild1000-v4-validation/vendor')
from safety_pair_v3_encoding import freeze_response,encode_response
D=lambda x:hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return [json.loads(l) for l in f if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
sources={}
def source(p):
 p=Path(p);sources[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)};return p
root=Path('/tmp/yzhang92/qwen17-existing-rejected-wild-20260930')
protocol=load(source(root/'expansion_protocol.json'));assert D(protocol)==load(root/'expansion_protocol.sha.json')['sha256']
pool=rows(source(root/'data/candidate_pool.jsonl'));assert len(pool)==5328 and D(pool)==protocol['pool_sha256']
original=rows(source(root/'data/dataset_rejected.jsonl'));assert len(original)==828
wildroot=Path('/tmp/yzhang92/qwen17-benign-source-20261003/wild')
oldmixed=rows(source(wildroot/'data/mixed.jsonl'));assert D(oldmixed)=='0e4470454a06182772da55ae60e194c94a28dbc7e4b43922493e5f5800f2c059'
benign=[r for r in oldmixed if r['task_type']=='benign'];assert len(benign)==1000
approved={};audits={};roundproofs=[]
pattern=re.compile(r'^\s*Result:\s+\*\*Result:\*\*\s+(Yes|No|Uncertain)\s*$',re.I)
for rp in sorted((root/'rounds').iterdir()):
 if not (rp/'complete.json').exists():continue
 complete=load(source(rp/'complete.json'));inp=load(source(rp/'quality/input.json'))
 links=rows(source(rp/'quality/links.jsonl'));req=rows(source(rp/'quality/requests.jsonl'));raw=rows(source(rp/'quality/judgments.jsonl'))
 assert D(links)==inp['links_sha256'] and D(req)==inp['requests_sha256'] and D(raw)==complete['quality_output_sha256']
 meta=load(source(rp/'quality/judgments.jsonl.run.json'));assert D(meta['config'])==meta['fingerprint']
 assert meta['config']['input_sha256']==D(req) and meta['config']['revision']=='023227c22156b0273b0693c82c65bd7084a50e92'
 assert meta['config']['protocol']=='source_blind_full_pair_quality_v1_20260909'
 assert all(r['run_fingerprint']==meta['fingerprint'] for r in raw)
 judged={r['request_id']:r for r in raw};assert set(judged)=={r['request_id'] for r in req}
 resolved={}
 for rid,r in judged.items():
  v=r
  if not r['parse_ok']:
   lines=r.get('raw_judgment','').rstrip().splitlines();match=pattern.fullmatch(lines[-1]) if lines else None
   if match and not any(re.search(r'Result:',s,re.I) for s in lines[:-1]):
    value={'yes':True,'no':False,'uncertain':None}[match.group(1).lower()]
    v=dict(r,parse_ok=True,value=value);resolved[rid]={'source_record_sha256':D(r),'terminal_label':match.group(1).lower(),'rule':'exact duplicated Result prefix on sole terminal result line'}
  judged[rid]=v
 candidates=rows(source(rp/'dataset_original/labeled_candidates.jsonl'));done=load(source(rp/'dataset_original/complete.json'))
 assert D(candidates)==done['labeled_sha256']
 assert all(r['run_fingerprint']==done['generation_fingerprint'] and r['judge_fingerprint']==done['judge_fingerprint'] for r in candidates)
 cmap={r['candidate_id']:r for r in candidates};rrows=rows(source(rp/'data/wild400.jsonl'));rmap={r['pair_id']:r for r in rrows}
 selected={}
 for link in sorted(links,key=lambda r:(r['pair_id'],r['rollout_index'])):
  q=judged[link['request_id']]
  if link['source']!='dataset_original' or not q['parse_ok'] or q.get('value') is not True:continue
  candidate=cmap[link['candidate_id']];r=rmap[link['pair_id']]
  assert candidate['negative_eligible'] and candidate['finish_reason']=='dataset_record'
  assert candidate['prompt']==r['prompt'] and candidate['raw_response']==r['existing_rejected_candidates'][0]['response']
  assert candidate['model']=='dataset:allenai/wildguardmix' and candidate['revision']=='d29c47f41c8b51348b5c8e8c81c039b3132b66d1'
  if r['pair_id'] in selected:continue
  selected[r['pair_id']]=(r,candidate,link,q)
 assert len(selected)==complete['quality_passed_prompts']['dataset_original']
 roundproofs.append({'round':complete['round'],'approved_off_policy':len(selected),'quality_run_fingerprint':meta['fingerprint']})
 for pid,(r,candidate,link,q) in selected.items():
  assert pid not in approved
  approved[pid]=(r,candidate,link,complete['round'])
  audits[pid]={'pair_id':pid,'round':complete['round'],'chosen_source':r['chosen_source'],'rejected_source':r['rejected_source'],'published_candidate_id':candidate['candidate_id'],'published_source_row_index':candidate['source_row_index'],'published_source_row_sha256':candidate['source_row_sha256'],'primary_judge_fingerprint':candidate['judge_fingerprint'],'quality_request_id':link['request_id'],'quality_judgment':q,'quality_format_resolution':resolved.get(link['request_id']),'student_success_required':False}
assert len(approved)==2057
oldids={r['pair_id'] for r in original};assert oldids<=set(approved)
for r in original:
 p,c,link,index=approved[r['pair_id']]
 assert r['prompt']==p['prompt'] and r['positive']['original_text']==p['chosen'] and r['negative']['original_text']==c['raw_response']
# Audit held-out prompt and underlying-family exclusion before extending the frozen bank.
normalize=lambda x:' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',x).casefold()))
def grams(x):
 w=normalize(x).split();return set(zip(*(w[i:] for i in range(3)))) if len(w)>=3 else {tuple(w)}
class Overlap:
 def __init__(self,texts):
  self.exact=set();self.sets=[];self.index=collections.defaultdict(set)
  for text in texts:
   if not text.strip():continue
   self.exact.add(normalize(text));tokens=grams(text);i=len(self.sets);self.sets.append(tokens)
   for token in tokens:self.index[token].add(i)
 def matches(self,text):
  if normalize(text) in self.exact:return True
  tokens=grams(text);common=collections.Counter(i for token in tokens for i in self.index.get(token,()))
  for i,n in common.items():
   other=self.sets[i]
   if n/len(tokens|other)>=.8 or (min(len(tokens),len(other))>=5 and n/min(len(tokens),len(other))>=.9):return True
  return False
heldout=[r['prompt'] for r in load(source(wildroot/'data/safety_eval.json'))['records']]
csv.field_size_limit(100000000)
wj=Path('/home/users/yzhang92/wildjailbreak_qwen17_pilot_20260930/source')
for r in csv.DictReader(source(wj/'eval/eval.tsv').open(newline=''),delimiter='\t'):
 heldout.extend(r.get(k,'') for k in ('vanilla','adversarial','prompt'))
exclusion=Overlap(heldout)
needed={r['chosen_source']['row_index'] for r,c,l,i in approved.values()}
train=source(wj/'train/train.tsv');assert sha(train)=='376719bfdb46ad1a19e7ba4f587f80cc7cb1368cc213ee647ef739c170550f7a'
indexed={i:r for i,r in enumerate(csv.DictReader(train.open(newline=''),delimiter='\t')) if i in needed}
assert set(indexed)==needed
benign_prompts={normalize(r['prompt']) for r in benign};seen={normalize(r['prompt']) for r in original}
assert len(seen)==828 and not seen&benign_prompts
eligible=[];excluded=collections.Counter();old_overlaps=[]
for pid,(r,c,l,i) in approved.items():
 t=indexed[r['chosen_source']['row_index']];assert D(t)==r['chosen_source']['row_sha256'] and t['completion']==r['chosen']
 assert r['prompt']==(t['vanilla'] if r['group']=='direct' else t['adversarial'])
 overlap=exclusion.matches(r['prompt']) or exclusion.matches(t['vanilla'])
 if pid in oldids:
  if overlap:old_overlaps.append(pid)
  continue
 if overlap:excluded['heldout_prompt_or_underlying_family']+=1;continue
 if normalize(r['prompt']) in seen or normalize(r['prompt']) in benign_prompts:excluded['duplicate_or_benign_prompt']+=1;continue
 eligible.append(pid)
assert not old_overlaps,('Original828 overlap audit requires attention',len(old_overlaps))
eligible.sort();random.Random(2002).shuffle(eligible);assert len(eligible)>=172
new=[];selected_extra=[]
for pid in eligible:
 r,c,l,i=approved[pid]
 if normalize(r['prompt']) in seen:continue
 seen.add(normalize(r['prompt']));selected_extra.append(pid)
 new.append(dict(pair_id=pid,prompt=r['prompt'],task_type='harmful',group=r['group'],positive=dict(representation='final_text',original_text=r['chosen']),negative=dict(representation='final_text',original_text=c['raw_response']),negative_evidence=dict(source_identity=c['model'],source_revision=c['revision'],candidate_id=c['candidate_id'],provenance_kind='published_dataset_original',raw_text_sha256=D(c['raw_response']),response_origin_fingerprint=c['run_fingerprint'],primary_judge_fingerprint=c['judge_fingerprint'],quality_request_id=l['request_id'],round=i),chosen_source=r['chosen_source']))
 if len(new)==172:break
assert len(new)==172
harmful=original+new;mixed=harmful+benign
assert len(harmful)==1000 and len(mixed)==2000 and len({r['pair_id'] for r in mixed})==2000
assert len({normalize(r['prompt']) for r in mixed})==2000
files={}
def put(n,values):files[n]=(json.dumps(values,ensure_ascii=False,indent=2)+'\n').encode()
def putrows(n,values):files[n]=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values).encode()
putrows('wild1000_harmful.jsonl',harmful);putrows('wild2000_mixed.jsonl',mixed)
audit=[dict(audits[r['pair_id']],retained_from_original828=r['pair_id'] in oldids) for r in harmful]
putrows('wild1000_selection.jsonl',audit)
models=load('/tmp/yzhang92/twostage-handoff-wild1000-v4-validation/provenance/baseline_model_identities.json')
assets=load('/home/users/yzhang92/twostage_share_20261014/assets.json');encoded={};checks={}
for key,entry in models.items():
 tok=AutoTokenizer.from_pretrained(assets['base_models'][key],local_files_only=True);values={}
 for row in mixed:
  sides={}
  for side in ('positive','negative'):
   f=row[side];assert f['representation']=='final_text';text=f['original_text'];assert text.strip() and '<think>' not in text and '</think>' not in text
   if key.startswith('r1'):e=encode_response(tok,row['prompt'],freeze_response(text,'final_text'),max_length=8192)
   else:
    prefix=tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert prefix.endswith('<think>\n\n</think>\n\n')
    context=tok.encode(prefix,add_special_tokens=False);answer=tok.encode(text,add_special_tokens=False);assert answer and tok.eos_token_id not in answer;answer.append(tok.eos_token_id)
    e={'ids':context+answer,'start':len(context),'target_tokens':len(answer),'serialized_sha256':D([context,answer]),'source_text_sha256':D(text)}
   assert 0<e['start']<len(e['ids'])<=8192 and e['ids'][-1]==tok.eos_token_id;sides[side]=e
  values[row['pair_id']]=sides
 encoded[key]=values;checks[key]={'revision':entry['revision'],'encoding_digest':D(values),'max_tokens':max(len(e['ids']) for r in values.values() for e in r.values()),'EOS_included':True,'no_truncation':True,'CPU_only':True}
assert encoded['qwen3_06b']==encoded['qwen3_17b']==encoded['qwen3_8b'] and encoded['r1_15b']==encoded['r1_7b']
for family,key in [('qwen','qwen3_17b'),('r1','r1_15b')]:put('wild2000_'+family+'.encoded.json',encoded[key])
assembly={'status':'verified_new_frozen_bank','public_quality_approved_pool':2057,'original_harmful_retained':828,'additional_harmful':172,'harmful_total':1000,'benign_total':1000,'mixed_total':2000,'benign_groups':dict(collections.Counter(r['group'] for r in benign)),'harmful_groups':dict(collections.Counter(r['group'] for r in harmful)),'candidate_exclusions':dict(excluded),'eligible_off_policy_extras':len(eligible),'original828_heldout_overlap':len(old_overlaps),'selection_seed':2002,'selection':'Keep original828 unchanged; append first172 distinct eligible published-only pairs after sorted-ID fixed-seed shuffle; no student-success intersection','heldout_policy':'Full4401+officialWild eval, actual prompt and underlying vanilla; normalized exact/trigram Jaccard>=.8 or containment>=.9 with min5trigrams. Not semantic deduplication.','round_quality_proofs':roundproofs,'new_model_calls':0,'new_training_answers':0,'all_answers_preserved':True,'benign_rows_unchanged':True,'row_digests':{'harmful':D(harmful),'mixed':D(mixed)}}
put('wild1000_assembly.json',assembly)
banks={}
for key in models:
 family='r1' if key.startswith('r1') else 'qwen'
 for kind,values in [('harmful',harmful),('mixed',mixed)]:
  banks[key+'_wild_'+kind]={'rows_file':'wild1000_harmful.jsonl' if kind=='harmful' else 'wild2000_mixed.jsonl','encoding_file':'wild2000_'+family+'.encoded.json','rows':len(values),'harmful':1000,'benign':0 if kind=='harmful' else 1000,'rows_digest':D(values),'encoding_digest':D(encoded[key]),'reference_model':key,'reference_status':'compute_own_original_pi0_on_new2000_bank_no_training_no_new_answers'}
manifest={'status':'verified_export','version':'Wild1000_v4_20261015','banks':banks,'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()},'original_sources':sources,'tokenizer_verification':checks,'new_reference_required':list(models),'source':'All1000 harmful/1000 benign are published off-policy pairs: WildJailbreak prompt/chosen + exact-prompt WildGuardTrain original rejected; no on-policy success requirement for the new172','assembly_digest':D(assembly)}
put('wild_manifest.json',manifest)
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
    if target.exists() and target.name!='wild_manifest.json':assert target.read_bytes()==raw
    else:target.write_bytes(raw)
 print('Assembled1000 harmful +1000 frozen Wild benign from verified public pairs; CPU only, no GPU work.')
if __name__=='__main__':main()
