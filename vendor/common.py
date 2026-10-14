"""Portable adapters for the unchanged historical numerical kernels."""
import hashlib
import json
import os
from pathlib import Path
from safety_pair_v3_common import digest, read_rows, save_rows, save_json, bind_run, recover_partial_tail, SYSTEM

HOME = Path.home()
JUDGE_REV = '023227c22156b0273b0693c82c65bd7084a50e92'

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def config(root):
    c = load(root/'config.json')
    assert digest(c) == load(root/'config.sha.json')['sha256']
    return c

def bank(root, key):
    rows = read_rows(root/'data'/f'{key}.jsonl')
    assert digest(rows) == config(root)['banks'][key]['sha256']
    return rows

def encoding(root, key):
    rows = load(root/'data'/f'{key}.encoded.json')
    assert digest(rows) == config(root)['banks'][key]['encoded_sha256']
    return rows

def pause(root):
    assert not (root/'user_pause.json').exists(), 'Run paused by its owner'

def verify_files(entry, hashes=False):
    p = Path(entry['path'])
    assert p.is_dir(), f'Model directory unavailable: {p}'
    for name, v in entry['files'].items():
        f = p/name
        assert f.stat().st_size == v['bytes'], name
        if hashes:
            assert sha(f) == v['sha256'], f'Model hash mismatch: {f}'

def weights_proof(path):
    return {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(Path(path).iterdir()) if p.is_file() and p.name != 'weights_proof.json'}

def identity(root, key):
    models = load(root/'models.json')
    if key in models:
        return models[key]
    done = load(root/'training'/key/'complete.json')
    assert done['status'] == 'complete'
    return dict(path=done['model_output'], files=done['files'], family='qwen3_17b',
                training_fingerprint=done['run_fingerprint'])
