#!/usr/bin/env python3
"""Read-only export of exact frozen banks, encodings, references and eval prompts.

Run from a computer that can SSH to Talos, or run --local on Talos itself.
No model generation/training, original-file writes, or source dataset rebuilding.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REMOTE = r'''
import collections,hashlib,io,json,os,sys,tarfile,importlib.metadata
from pathlib import Path
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return [json.loads(l) for l in f if l.strip()]
def cfg(p):
 c=load(p/'config.json');assert digest(c)==load(p/'config.sha.json')['sha256'];return c
roots={'stage2':Path('/tmp/yzhang92/wj-s2-20261004'),
 'harmful':Path('/tmp/yzhang92/qwen17-existing-rejected-wild-20260930'),
 'mixed':Path('/tmp/yzhang92/qwen17-wild-mixed-sampling-20261002/mixed')}
keys={'stage2':'sampled','harmful':'dataset_rejected','mixed':'mixed'}
expected={'stage2':(1972,1000,972),'harmful':(828,828,0),'mixed':(1828,828,1000)}
files={};banks={};configs={}
def put(name,content):
 assert name not in files;files[name]=content
for kind,root in roots.items():
 c=cfg(root);configs[kind]=c;key=keys[kind]
 src=root/'data'/f'{key}.jsonl';encpath=root/'data'/f'{key}.encoded.json'
 refpath=root/'reference'/key/'scores.jsonl';proof=load(root/'reference'/key/'complete.json')
 data=rows(src);encoded=load(encpath);refs=rows(refpath)
 assert digest(data)==c['banks'][key]['sha256']
 assert digest(encoded)==c['banks'][key]['encoded_sha256']
 assert digest(refs)==proof['scores_sha256']
 count=collections.Counter(r['task_type'] for r in data)
 assert (len(data),count['harmful'],count['benign'])==expected[kind]
 ids={r['pair_id'] for r in data};assert len(ids)==len(data) and ids==set(encoded)=={r['pair_id'] for r in refs}
 lookup={r['pair_id']:r for r in data};assert all(r['row_sha256']==digest(lookup[r['pair_id']]) for r in refs)
 names=[f'{kind}.jsonl',f'{kind}.encoded.json',f'{kind}.reference.jsonl']
 for name,source in zip(names,[src,encpath,refpath]):put(name,source.read_bytes())
 banks[kind]=dict(rows_file=names[0],encoding_file=names[1],reference_file=names[2],
  rows=len(data),harmful=count['harmful'],benign=count['benign'],rows_digest=digest(data),
  encoding_digest=digest(encoded),reference_digest=digest(refs),original_config_digest=digest(c),
  original_reference_proof=proof,source_root=str(root),source_bank=key)
c=configs['stage2'];root=roots['stage2']
safety=load(root/'data/safety_eval.json');assert digest(safety)==c['safety_manifest_sha256']
assert len(safety['records'])==4401
dev=rows(root/'data/development.jsonl');assert len(dev)==64
assert digest(dev)==load(root/'development_selection.json')['sha256']
put('safety_eval.json',(root/'data/safety_eval.json').read_bytes())
put('development.jsonl',(root/'data/development.jsonl').read_bytes())
manifest=dict(status='verified_export',banks=banks,safety_manifest_digest=digest(safety),
 development_digest=digest(dev),files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()},
 source='Frozen Talos completed Wild experiments; no sampling, filtering or response replacement',
 model_references=dict(stage2='original pi1',harmful='original pi0',mixed='original pi0'))
put('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
guardproof=Path('/home/users/yzhang92/safety_pair_v3_20260909/evaluator/Llama-Guard-3-8B.transfer.json')
put('guard_proof.json',guardproof.read_bytes())
versions={}
for k in ['torch','transformers','vllm','safetensors','accelerate','datasets','huggingface_hub','numpy']:
 try:versions[k]=importlib.metadata.version(k)
 except importlib.metadata.PackageNotFoundError:versions[k]=None
put('environment_versions.json',(json.dumps(versions,indent=2)+'\n').encode())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tf:
 for name,content in files.items():
  info=tarfile.TarInfo(name);info.size=len(content);info.mtime=0;tf.addfile(info,io.BytesIO(content))
'''

def install(archive):
    with tempfile.TemporaryDirectory(prefix='twostage-export-') as temp:
        target=Path(temp)
        with tarfile.open(archive,'r:gz') as tf:
            for item in tf.getmembers():
                assert item.isfile() and Path(item.name).name==item.name and item.size<100*1024**2
                stream=tf.extractfile(item);assert stream is not None
                (target/item.name).write_bytes(stream.read())
        manifest=json.loads((target/'manifest.json').read_text())
        for name,info in manifest['files'].items():
            raw=(target/name).read_bytes()
            assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'],name
        for source in target.iterdir():
            folder='provenance' if source.name in ('guard_proof.json','environment_versions.json') else 'data'
            dst=ROOT/folder/source.name
            if dst.exists():
                assert dst.read_bytes()==source.read_bytes(),f'Existing export differs: {dst}'
            else:
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    subprocess.run([sys.executable,str(ROOT/'reproduce.py'),'verify-data'],check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh',default='yzhang92@talos.cilab.gmu.edu')
    p.add_argument('--python',default='/home/users/yzhang92/safety_pair_v3_20260909/selene-env/bin/python')
    p.add_argument('--local',action='store_true')
    p.add_argument('--from-archive',type=Path)
    a=p.parse_args()
    if a.from_archive:
        install(a.from_archive);return
    with tempfile.TemporaryDirectory(prefix='twostage-transfer-') as temp:
        archive=Path(temp)/'frozen-data.tar.gz'
        cmd=[a.python,'-'] if a.local else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',a.ssh,a.python,'-']
        with archive.open('wb') as f:
            subprocess.run(cmd,input=REMOTE.encode(),stdout=f,check=True)
        install(archive)

if __name__=='__main__':main()
