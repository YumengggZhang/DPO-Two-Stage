"""Resume independent Selene target/counter judgments; never synthesize extra labels."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from safety_pair_v3_common import (SELENE,digest,read_rows,save_json,bind_run,recover_partial_tail)
from safety_pair_v3_binary_rubrics import PROTOCOL,render,parse,preliminary_polarity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-model-len',type=int,default=8192)
    p.add_argument('--max-new-tokens',type=int,default=192)
    p.add_argument('--reuse-input')
    p.add_argument('--reuse-output')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    inputs=read_rows(a.input)
    rows=[r for r in inputs if r['kind']=='response']
    assert rows and len({r['request_id'] for r in rows})==len(rows)
    payloads={(r['request_id'],dim):render(r,dim) for r in rows for dim in ['target','counter']}
    config=dict(protocol=PROTOCOL,model=SELENE,revision=a.revision,input_sha256=digest(rows),
        rendered_sha256=digest([[list(k),v] for k,v in payloads.items()]),
        max_model_len=a.max_model_len,max_new_tokens=a.max_new_tokens)
    reusable=[]
    assert bool(a.reuse_input)==bool(a.reuse_output), 'Reuse requires both original requests and judgments'
    if a.reuse_output:
        old_input=read_rows(a.reuse_input)
        old_results=read_rows(a.reuse_output)
        old_meta=json.loads(Path(a.reuse_output+'.run.json').read_text())
        old_summary=json.loads(Path(a.reuse_output+'.summary.json').read_text())
        assert old_meta['config']['input_sha256']==digest(old_input)
        old_payloads=[[[r['request_id'],dim],render(r,dim)]
                      for r in old_input for dim in ['target','counter']]
        assert old_meta['config']['rendered_sha256']==digest(old_payloads), 'Original rubric rendering changed'
        for key in ['protocol','model','revision','max_model_len','max_new_tokens']:
            assert old_meta['config'][key]==config[key], ('reuse_protocol_mismatch',key)
        assert old_summary['run_fingerprint']==old_meta['fingerprint']
        assert all(r['run_fingerprint']==old_meta['fingerprint'] for r in old_results)
        latest={r['request_id']:r for r in old_results}
        assert len(latest)==old_summary['requests']==len(old_input), 'Reuse only finished prior batches'
        old_requests={r['request_id']:r for r in old_input}
        current={r['request_id']:r for r in rows}
        for rid,r in latest.items():
            if rid not in current or not r.get('parse_ok'):
                continue
            if not all(render(old_requests[rid],dim)==payloads[(rid,dim)] for dim in ['target','counter']):
                raise ValueError('same_request_id_has_different_rendered_content')
            reusable.append(r)
        config['reuse']=dict(source_run_fingerprint=old_meta['fingerprint'],
            source_results_sha256=digest(old_results),source_input_sha256=digest(old_input),
            records=len(reusable))
    if a.dry_run:
        print(json.dumps(dict(requests=len(rows),independent_decisions=2*len(rows),config=config)))
        return
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_name(path.name+'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fingerprint=bind_run(path,config)
    recover_partial_tail(path)
    existing=read_rows(path)
    assert all(r['run_fingerprint']==fingerprint for r in existing)
    done={r['request_id']:r for r in existing}
    if reusable:
        with path.open('a') as f:
            for old in reusable:
                if old['request_id'] in done:
                    continue
                record=dict(old,run_fingerprint=fingerprint,reused_from=dict(
                    source_output=a.reuse_output,source_run_fingerprint=old['run_fingerprint'],
                    identical_source_blind_rendered_content=True))
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
                done[record['request_id']]=record
            f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(reused=len(reusable),total=len(rows))),flush=True)
    assert set(done)<={r['request_id'] for r in rows}
    pending=[r for r in rows if not done.get(r['request_id'],{}).get('parse_ok')]
    if pending:
        from vllm import LLM,SamplingParams
        shared_model=os.environ.get('TWOSTAGE_SELENE_PATH')
        llm=LLM(model=shared_model or SELENE,revision=None if shared_model else a.revision,dtype='auto',max_model_len=a.max_model_len,
            max_num_seqs=a.batch_size,gpu_memory_utilization=.96,
            enable_prefix_caching=True,enforce_eager=True,seed=2002)
        tok=llm.get_tokenizer()
        with path.open('a') as f:
            for budget in [a.max_new_tokens,2*a.max_new_tokens]:
                retry=[]
                for start in range(0,len(pending),a.batch_size):
                    batch=pending[start:start+a.batch_size]
                    requests,keys,judgments=[],[],{}
                    for r in batch:
                        for dim in ['target','counter']:
                            text=tok.apply_chat_template([dict(role='user',content=payloads[(r['request_id'],dim)])],
                                tokenize=False,add_generation_prompt=True)
                            if len(tok.encode(text,add_special_tokens=False))+budget>a.max_model_len:
                                judgments[(r['request_id'],dim)]=dict(parse_ok=False,value=None,
                                    error='context_overflow_no_truncation')
                            else:
                                keys.append((r['request_id'],dim));requests.append(text)
                    outputs=llm.generate(requests,SamplingParams(temperature=0,max_tokens=budget,seed=2002),
                        use_tqdm=False) if requests else []
                    assert len(outputs)==len(keys)
                    for key,result in zip(keys,outputs):
                        raw=result.outputs[0].text
                        judgments[key]=dict(**parse(raw),raw=raw,finish_reason=result.outputs[0].finish_reason)
                    for r in batch:
                        decisions={dim:judgments[(r['request_id'],dim)] for dim in ['target','counter']}
                        parsed=all(x['parse_ok'] for x in decisions.values())
                        target,counter=decisions['target']['value'],decisions['counter']['value']
                        record=dict(request_id=r['request_id'],kind='response',task_type=r['task_type'],
                            run_fingerprint=fingerprint,judge_model=SELENE,judge_revision=a.revision,
                            protocol=PROTOCOL,parse_ok=parsed,decisions=decisions,
                            verdict=dict(target_valid=target,counter_policy=counter),
                            polarity=preliminary_polarity(target,counter),judge_max_tokens=budget,
                            qualification='provisional; secondary checks, completeness and quality audit still required')
                        if not r['response'].strip():
                            record['polarity']='neither'
                            record['eligibility_exclusion']='empty_response_not_a_behavior'
                            # Keep the model's raw verdict even when this
                            # deterministic eligibility rule excludes the text.
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');done[r['request_id']]=record
                        if not parsed and not any(x.get('error') for x in decisions.values()):
                            retry.append(r)
                    f.flush();os.fsync(f.fileno())
                    print(f'budget={budget} attempted={min(start+len(batch),len(pending))}/{len(pending)}',flush=True)
                pending=retry
                if not pending:
                    break
    assert len(done)==len(rows)
    unparsed=sum(not r['parse_ok'] for r in done.values())
    save_json(path.with_name(path.name+'.summary.json'),dict(status='complete' if not unparsed else 'needs_review',
        requests=len(rows),independent_decisions=2*len(rows),parsed=len(rows)-unparsed,unparsed=unparsed,
        uncertain=sum(any(v is None for v in r['verdict'].values()) for r in done.values()),
        run_fingerprint=fingerprint,note='Only target and counter were judged; no invented quality/harm/ambiguity fields'))


if __name__=='__main__':
    main()
