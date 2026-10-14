"""Matched domain-routed KL/DPO; complete continuation paths including EOS."""
import argparse
import math
import random
import shutil
import time
from common import *
from protocol import LEARNING_RATES, objective_paths


def models(root, method):
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.device_count() == config(root).get('training_gpus',1)
    def one(key, frozen):
        entry = identity(root, key); verify_files(entry, hashes=True)
        model = AutoModelForCausalLM.from_pretrained(entry['path'], dtype=torch.float32,
            attn_implementation='sdpa', local_files_only=True).to('cuda')
        model.config.use_cache = False
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout): layer.p = 0.
        if frozen: model.requires_grad_(False); model.eval()
        return model
    student = one('pi1', False)
    keys = {objective_paths(method, d)[1] for d in ('harmful', 'benign')
            if objective_paths(method, d)[0] == 'kl'}
    teachers = {key: one(key, True) for key in sorted(keys)}
    assert all(t.config.vocab_size == student.config.vocab_size for t in teachers.values())
    return student, teachers


def backward_row(model, teachers, row, encoded, refs, method, beta, divisor):
    import torch
    from kl_objective import answer_kl
    from qwen_encoding import logps, backward_dpo
    from standard_dpo_objective_20260912 import standard_dpo_loss
    kind, teacher, sides = objective_paths(method, row['task_type'])
    if kind == 'kl':
        total = 0.
        for side in sides:
            loss = answer_kl(model, teachers[teacher], encoded[side])
            assert torch.isfinite(loss) and float(loss.detach()) >= -1e-5
            scaled = loss / (len(sides) * divisor)
            total += float(scaled.detach()); scaled.backward()
        return total
    lp, ln = (logps(model, encoded[s]) for s in sides)
    r = refs[row['pair_id']]['scores']
    loss = standard_dpo_loss(lp, ln, r['positive'], r['negative'], beta) / divisor
    assert torch.isfinite(loss)
    value = float(loss.detach()); backward_dpo(loss, lp, ln)
    return value


def smoke(root):
    import torch
    from transformers import set_seed
    from optimizer_factory import optimizer
    from kl_objective import self_test, answer_kl
    c = config(root); method = c['method']; set_seed(c['seed'])
    tests = self_test(); student, teachers = models(root, method)
    student.train(); student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    rows, enc = bank(root, 'sampled'), encoding(root, 'sampled')
    refs = {r['pair_id']: r for r in read_rows(root/'reference/sampled/scores.jsonl')}
    opt = optimizer(student,c); opt.zero_grad(); details = []
    probe = student.model.layers[0].self_attn.q_proj.weight; before = probe.detach().clone()
    for domain in ('harmful', 'benign'):
        if not any(r['task_type'] == domain for r in rows):
            continue
        row = max((r for r in rows if r['task_type'] == domain),
                  key=lambda r: sum(len(e['ids']) for e in enc[r['pair_id']].values()))
        kind, key, sides = objective_paths(method, domain)
        if kind == 'kl' and domain == 'harmful':
            with torch.no_grad():
                v = float(answer_kl(student, teachers[key], enc[row['pair_id']]['positive']))
            assert abs(v) < 1e-6, ('Initial student and harmful teacher must match', v)
        loss = backward_row(student, teachers, row, enc[row['pair_id']], refs, method, c['beta'], 2)
        details.append(dict(domain=domain, objective=kind, teacher=key, sides=sides, loss=loss))
    assert all(p.grad is None for t in teachers.values() for p in t.parameters())
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), c['grad_clip'])
    assert torch.isfinite(norm) and float(norm) > 0
    opt.step(); assert not torch.equal(before, probe)
    save_json(root/'state/smoke.json', dict(status='passed', method=method, tests=tests,
        paths=details, teacher_gradients_none=True, initial_harmful_kl_zero=(True if objective_paths(method, 'harmful')[0] == 'kl' else None),
        gpu_optimizer=True, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


def train(root, tag, mode):
    import torch
    from transformers import AutoTokenizer, set_seed
    from optimizer_factory import optimizer
    from qwen_encoding import learning_factor
    c = config(root); method = c['method']; lr = LEARNING_RATES[tag]
    rows, enc = bank(root, 'sampled'), encoding(root, 'sampled')
    refrows = read_rows(root/'reference/sampled/scores.jsonl')
    proof = load(root/'reference/sampled/complete.json')
    assert digest(refrows) == proof['scores_sha256']
    refs = {r['pair_id']: r for r in refrows}
    assert set(refs) == {r['pair_id'] for r in rows}
    assert all(refs[r['pair_id']]['row_sha256'] == digest(r) for r in rows)
    assert load(root/'state/smoke.json')['status'] == 'passed'
    batches = [rows[i:i+c['batch_size']] for i in range(0, len(rows), c['batch_size'])]
    assert digest([[r['pair_id'] for r in b] for b in batches]) == c['banks']['sampled']['schedule_sha256']
    total = len(batches); steps = min(40, total) if mode == 'pilot' else total
    name = mode+'_'+tag; out = root/'training'/name; out.mkdir(parents=True, exist_ok=True)
    fp = bind_run(out/'training', dict(config_sha256=digest(c), reference=proof,
        learning_rate=lr, steps=steps, schedule_steps=total, method=method,
        initialization=identity(root, 'pi1'), teachers={d: objective_paths(method, d) for d in ('harmful', 'benign')}))
    if (out/'complete.json').exists():
        assert load(out/'complete.json')['run_fingerprint'] == fp; return
    memory = Path(c['ephemeral_storage'])/method/name; memory.mkdir(parents=True, exist_ok=True)
    resume = memory/'resume.pt'
    target = (memory if mode == 'pilot' else out)/'model'
    # A model without a completion record is not reusable. Preserve it for inspection.
    if target.exists(): target.rename(target.with_name('model.interrupted.'+str(time.time_ns())))
    set_seed(c['seed']); torch.set_num_threads(8)
    model, teachers = models(root, method); model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    opt = optimizer(model,c,lr=lr,weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt.optimizer, lambda s: learning_factor(s, total))
    start = 0
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
        assert ck['fingerprint'] == fp
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); start = ck['step']
        random.setstate(ck['python_rng']); torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng']); del ck
    old = read_rows(out/'metrics.jsonl'); assert all(v['run_fingerprint'] == fp for v in old)
    if any(v['step'] > start for v in old):
        shutil.copy2(out/'metrics.jsonl', out/f'metrics.interrupted.{time.time_ns()}.jsonl')
    metrics = [v for v in old if v['step'] <= start]
    assert [v['step'] for v in metrics] == list(range(1, start+1))
    save_rows(out/'metrics.jsonl', metrics); began = time.time()
    for i, batch in enumerate(batches[:steps], 1):
        if i <= start: continue
        pause(root); opt.zero_grad(); loss_total = 0.
        for row in batch:
            loss_total += backward_row(model, teachers, row, enc[row['pair_id']], refs,
                                       method, c['beta'], len(batch))
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
        assert math.isfinite(loss_total) and torch.isfinite(grad)
        actual_lr = opt.optimizer.param_groups[0]['lr']; opt.step(); scheduler.step(); opt.zero_grad()
        v = dict(step=i, total_steps=steps, loss=loss_total, grad_norm=float(grad),
                 lr=actual_lr, seconds=time.time()-began, run_fingerprint=fp)
        metrics.append(v); save_rows(out/'metrics.jsonl', metrics)
        if i % 10 == 0: print(json.dumps(dict(method=method, tag=name, **v)), flush=True)
        if i % 50 == 0 or i == steps:
            required_resume=3*sum(p.numel()*4 for p in model.parameters())+3*1024**3
            assert shutil.disk_usage(memory).free > required_resume, 'Need atomic FP32 model/moments resume storage'
            temp = memory/'resume.tmp'
            torch.save(dict(fingerprint=fp, step=i, model=opt.checkpoint_model(model),
                optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), python_rng=random.getstate(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temp)
            temp.replace(resume)
            save_json(out/'resume.json', dict(step=i, path=str(resume), ephemeral=False))
    assert all(p.grad is None for t in teachers.values() for p in t.parameters())
    del teachers; model.gradient_checkpointing_disable(); model.config.use_cache = True
    if mode == 'pilot': model.to(dtype=torch.bfloat16)
    required = sum(p.numel()*p.element_size() for p in model.parameters())+3*1024**3
    assert shutil.disk_usage(target.parent).free > required, 'Insufficient final checkpoint storage'
    target.mkdir(parents=True)
    model.save_pretrained(target, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(identity(root, 'pi1')['path'], local_files_only=True).save_pretrained(target)
    save_json(out/'complete.json', dict(status='complete', run_fingerprint=fp, model_output=str(target),
        files=weights_proof(target), steps=steps, rows_seen=sum(map(len, batches[:steps])), learning_rate=lr,
        metrics_sha256=digest(metrics), initial_policy=c['job']['initial'], reference=c['job']['reference'], method=method, total_schedule_steps=total,
        mode=mode, export_dtype='BF16' if mode=='pilot' else 'FP32', pilot_export_ephemeral=False))
    # Only this run's temporary optimizer state is removed after verified model export.
    resume.unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['smoke','train'], required=True)
    p.add_argument('--tag', choices=list(LEARNING_RATES)); p.add_argument('--mode', choices=['pilot','full'])
    a = p.parse_args(); smoke(a.root) if a.stage=='smoke' else train(a.root, a.tag, a.mode)
