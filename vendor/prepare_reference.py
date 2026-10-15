"""Teacher-forced fixed-pi0 log-probabilities. No training or answer generation."""
import argparse,math
from common import *
from qwen_encoding import logps

def validate_prior(rows, encoded, prior, fingerprint):
    lookup={r['pair_id']:r for r in rows}
    assert len(lookup)==len(rows) and set(lookup)<=set(encoded)
    done={r['pair_id']:r for r in prior};assert len(done)==len(prior)
    for key,value in done.items():
        assert key in lookup and value['run_fingerprint']==fingerprint
        assert value['row_sha256']==digest(lookup[key])
        assert set(value['scores'])=={'positive','negative'}
        assert all(math.isfinite(x) for x in value['scores'].values())
    return done

def score_rows(model, rows, encoded, prior, fingerprint, emit, check_pause=lambda:None):
    import torch
    done=validate_prior(rows,encoded,prior,fingerprint)
    with torch.no_grad():
        for row in rows:
            key=row['pair_id']
            if key in done:continue
            check_pause()
            scores={side:float(logps(model,encoded[key][side]).sum()) for side in ('positive','negative')}
            assert all(math.isfinite(v) for v in scores.values())
            value=dict(pair_id=key,row_sha256=digest(row),scores=scores,run_fingerprint=fingerprint)
            emit(value);done[key]=value
    assert set(done)=={r['pair_id'] for r in rows}
    return done

def run(root):
    import torch
    from transformers import AutoModelForCausalLM
    c=config(root);assert c['purpose']=='frozen_pi0_reference_only' and c['training_gpus']==1
    entry=identity(root,'initial');verify_files(entry,hashes=True)
    rows=bank(root,'sampled');encoded=encoding(root,'sampled')
    path=root/'reference/scores.jsonl'
    fp=bind_run(path,dict(config_sha256=digest(c),reference_model=entry,rows_sha256=digest(rows),encoding_sha256=digest(encoded),score_reduction='sequence_sum_including_EOS',parameters='FP32',forward='BF16',optimization_steps=0))
    recover_partial_tail(path);prior=read_rows(path);done=validate_prior(rows,encoded,prior,fp)
    if len(done)<len(rows):
        assert torch.cuda.device_count()==1
        model=AutoModelForCausalLM.from_pretrained(entry['path'],dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        model.config.use_cache=False;model.requires_grad_(False)
        for layer in model.modules():
            if isinstance(layer,torch.nn.Dropout):layer.p=0.
        with path.open('a') as f:
            counter=[len(done)]
            def emit(value):
                f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();counter[0]+=1
                if counter[0]%100==0:print(json.dumps({'reference_rows':counter[0],'total':len(rows)}),flush=True)
            score_rows(model,rows,encoded,prior,fp,emit,lambda:pause(root))
    final=read_rows(path);validate_prior(rows,encoded,final,fp);assert len(final)==len(rows)
    save_json(root/'reference/complete.json',dict(status='complete',rows=len(final),scores_sha256=digest(final),file_sha256=sha(path),run_fingerprint=fp,reference_model_files_digest=digest(entry['files']),optimization_steps=0,new_answers=0))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);run(a.parse_args().root)
