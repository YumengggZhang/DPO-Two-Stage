#!/usr/bin/env python3
"""Owner-side, account-specific read-only assets for the authorized collaborator.

Run on Talos as the original asset owner. Never alter frozen model contents or
private cache permissions. Public-model assets on the same filesystem use hard
links; the original pi1 is copied across filesystems and hash checked.
"""
import argparse
import errno
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def share(source,target,files=None):
    target.mkdir(parents=True,exist_ok=True,mode=0o755)
    names=list(files) if files is not None else [p.name for p in source.iterdir() if p.is_file()]
    proof={}
    for name in names:
        src=(source/name).resolve();dst=target/name
        before=(src.stat().st_ino,src.stat().st_size,src.stat().st_mtime_ns)
        info=dict(bytes=src.stat().st_size,sha256=sha(src))
        if files is not None:assert info==files[name],f'Original model hash mismatch: {name}'
        if dst.exists():
            assert dst.stat().st_size==info['bytes'] and sha(dst)==info['sha256']
        else:
            linked=False
            if src.stat().st_dev==target.stat().st_dev and stat.S_IMODE(src.stat().st_mode)&0o004:
                try:
                    os.link(src,dst);linked=True
                except OSError as exc:
                    if exc.errno!=errno.EXDEV:raise
            if not linked:
                assert shutil.disk_usage(target).free>info['bytes']+8*1024**3
                tmp=dst.with_name(dst.name+'.partial');shutil.copyfile(src,tmp);tmp.chmod(0o644)
                assert sha(tmp)==info['sha256'];tmp.replace(dst)
        assert (src.stat().st_ino,src.stat().st_size,src.stat().st_mtime_ns)==before
        proof[name]=info
    return proof

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collaborator',required=True)
    p.add_argument('--share-root',type=Path,default=Path.home()/'twostage_share_20261014')
    a=p.parse_args();who=pwd.getpwnam(a.collaborator);owner=pwd.getpwuid(os.getuid())
    assert who.pw_uid!=os.getuid()
    target=a.share_root.expanduser().resolve();assert target.parent==Path.home()
    target.mkdir(mode=0o700,exist_ok=True)
    subprocess.run(['setfacl','-m',f'u:{a.collaborator}:r-x',str(target)],check=True)
    groups=os.getgrouplist(a.collaborator,who.pw_gid)
    home=Path.home()
    if home.stat().st_gid not in groups and not (home.stat().st_mode&0o001):
        subprocess.run(['setfacl','-m',f'u:{a.collaborator}:--x',str(home)],check=True)
    models=json.loads((ROOT/'provenance/checkpoint_identities.json').read_text())
    pi1=share(Path(models['pi1']['path']),target/'pi1',models['pi1']['files'])
    from huggingface_hub import hf_hub_download
    selene_config=hf_hub_download('AtlaAI/Selene-1-Llama-3.3-70B-GPTQ-W8A8',
        revision='023227c22156b0273b0693c82c65bd7084a50e92',filename='config.json',local_files_only=True)
    selene=share(Path(selene_config).parent,target/'selene')
    guard_source=home/'safety_pair_v3_20260909/evaluator/Llama-Guard-3-8B'
    guard_proof=json.loads((guard_source.parent/'Llama-Guard-3-8B.transfer.json').read_text())
    share(guard_source,target/'guard',guard_proof['files'])
    (target/'guard_proof.json').write_text(json.dumps(guard_proof,indent=2)+'\n')
    (target/'guard_proof.json').chmod(0o644)
    assets=dict(initial=models['initial']['path'],pi1=str(target/'pi1'),guard=str(target/'guard'),
        guard_proof=str(target/'guard_proof.json'),selene=str(target/'selene'))
    (target/'assets.json').write_text(json.dumps(assets,indent=2)+'\n');(target/'assets.json').chmod(0o644)
    proof=dict(status='verified_shared_assets',collaborator=a.collaborator,assets=assets,
        selene_files=selene,pi1_files=pi1,original_files_unchanged=True,
        sharing='new directory ACL; same-filesystem read-only model links; copied pi1')
    (target/'shared_proof.json').write_text(json.dumps(proof,indent=2)+'\n');(target/'shared_proof.json').chmod(0o644)
    print(json.dumps(proof))

if __name__=='__main__':main()
