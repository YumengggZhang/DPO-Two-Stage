#!/usr/bin/env python3
"""Verify every shard and restore the original data byte-for-byte."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]

def unpack(meta_file):
    meta=json.loads((ROOT/'data'/meta_file).read_text())
    chunks=[]
    for part in meta['parts']:
        data=(ROOT/'data'/part['name']).read_bytes()
        assert len(data)==part['bytes'] and hashlib.sha256(data).hexdigest()==part['sha256']
        chunks.append(data)
    data=b''.join(chunks)
    assert len(data)==meta['archive_bytes'] and hashlib.sha256(data).hexdigest()==meta['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tf:
        members=tf.getmembers()
        assert {v.name for v in members}==set(meta['unpacked_files']) and len(members)==len(meta['unpacked_files'])
        for item in members:
            assert item.isfile() and Path(item.name).name==item.name
            info=meta['unpacked_files'][item.name]
            assert item.size==info['bytes']
            source=tf.extractfile(item);assert source is not None
            raw=source.read();assert hashlib.sha256(raw).hexdigest()==info['sha256']
            target=ROOT/'data'/item.name
            if target.exists():
                assert target.read_bytes()==raw,f'Existing file differs: {target}'
            else:
                temp=target.with_name(target.name+'.tmp');temp.write_bytes(raw);temp.replace(target)
    print(f'Verified and restored {len(members)} frozen data files.')

def main():
    for name in ('archive.json','wild_archive.json'):unpack(name)

if __name__=='__main__':main()
