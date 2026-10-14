#!/usr/bin/env python3
"""Verify exact model assets, and optionally audit another Talos user's ACL access."""
import argparse
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from common import load,verify_files

def permissions(path,user):
    user=pwd.getpwnam(user);groups=os.getgrouplist(user.pw_name,user.pw_gid)
    raw=subprocess.check_output(['getfacl','-cpn',str(path)],text=True)
    entries={tuple(line.split(':')[:2]):line.split(':')[2].split()[0]
             for line in raw.splitlines() if line and not line.startswith('#') and ':' in line}
    def bits(s):return sum(v for c,v in [('r',4),('w',2),('x',1)] if c in s)
    st=path.stat();mask=bits(entries.get(('mask',''),'rwx'))
    if user.pw_uid==st.st_uid:return bits(entries[('user','')])
    if ('user',str(user.pw_uid)) in entries:return bits(entries[('user',str(user.pw_uid))])&mask
    relevant=[]
    if st.st_gid in groups:relevant.append(bits(entries[('group','')]))
    relevant += [bits(v) for (kind,key),v in entries.items() if kind=='group' and key and int(key) in groups]
    if relevant:
        value=0
        for v in relevant:value|=v
        return value&mask
    return bits(entries[('other','')])

def audit(path,user,execute=False):
    path=path.resolve();needed=1 if path.is_dir() else 5 if execute else 4
    assert permissions(path,user)&needed==needed,f'{user} cannot access {path}'
    for parent in path.parents:
        assert permissions(parent,user)&1,f'{user} cannot traverse {parent}'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=ROOT/'assets.talos.json')
    p.add_argument('--permissions-for')
    p.add_argument('--runtime',type=Path)
    a=p.parse_args();assets=load(a.assets);models=load(ROOT/'provenance/checkpoint_identities.json')
    guard=load(Path(assets['guard_proof']))
    proofs={'initial':models['initial']['files'],'pi1':models['pi1']['files'],'guard':guard['files']}
    if assets.get('selene'):proofs['selene']=load(ROOT/'provenance/selene_shared_proof.json')['files']
    for key,files in proofs.items():
        verify_files(dict(path=assets[key],files=files),hashes=True)
        if a.permissions_for:
            for name in files:audit(Path(assets[key])/name,a.permissions_for)
    if assets.get('base_models'):
        identities=load(ROOT/'provenance/baseline_model_identities.json')
        for key,entry in identities.items():
            verify_files(dict(path=assets['base_models'][key],files=entry['files']),hashes=True)
            if a.permissions_for:
                for name in entry['files']:audit(Path(assets['base_models'][key])/name,a.permissions_for)
        proofs.update({k:identities[k]['files'] for k in identities})
    if a.runtime:
        if a.permissions_for:audit(a.runtime,a.permissions_for,execute=True)
        else:assert os.access(a.runtime,os.X_OK)
    print(json.dumps(dict(status='passed',models=list(proofs),exact_model_hashes=True,
        permissions_for=a.permissions_for,permission_check='POSIX mode and ACL evaluation, not a login as that user')))

if __name__=='__main__':main()
