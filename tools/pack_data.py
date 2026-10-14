#!/usr/bin/env python3
"""Deterministically pack verified frozen data into GitHub-sized binary shards."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reproduce

def main():
    manifest=reproduce.load(ROOT/'data/baseline_manifest.json') if '--baseline' in sys.argv else reproduce.load(ROOT/'data/manifest.json')
    prefix='baseline_data' if '--baseline' in sys.argv else 'frozen_data'
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0,filename='') as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tf:
            for name in sorted(manifest['files']):
                data=(ROOT/'data'/name).read_bytes()
                entry=tarfile.TarInfo(name);entry.size=len(data);entry.mtime=0
                tf.addfile(entry,io.BytesIO(data))
    content=buffer.getvalue();parts=[];size=3*1024**2
    for index,start in enumerate(range(0,len(content),size)):
        name=f'{prefix}.tar.gz.part-{index:03d}';data=content[start:start+size]
        (ROOT/'data'/name).write_bytes(data)
        parts.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    value=dict(format='concatenated gzip tar',archive_bytes=len(content),
        archive_sha256=hashlib.sha256(content).hexdigest(),parts=parts,
        unpacked_files=manifest['files'])
    (ROOT/'data'/('baseline_archive.json' if '--baseline' in sys.argv else 'archive.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(uncompressed_bytes=sum(v['bytes'] for v in manifest['files'].values()),
                         compressed_bytes=len(content),parts=len(parts))))

if __name__=='__main__':main()
