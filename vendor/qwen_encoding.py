"""Frozen protocol and native non-thinking serialization for the Qwen3 comparison."""
import json
import math
import random
from pathlib import Path
from safety_pair_v3_common import digest, read_rows, save_json, save_rows, bind_run, recover_partial_tail, SYSTEM


def config(root):
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text())
    assert digest(cfg) == json.loads((root/'config.sha.json').read_text())['sha256']
    assert cfg['enable_thinking'] is False
    return cfg


def bank(root, name):
    rows = read_rows(Path(root)/'data'/f'{name}.jsonl')
    assert digest(rows) == config(root)['banks'][name]
    return rows


def schedule(root, name):
    rows = bank(root, name)
    rng = random.Random(2002)
    if name == 'clean1000':
        rng.shuffle(rows)
        batches = [rows[i:i+8] for i in range(0, len(rows), 8)]
    else:
        pools = {d: [r for r in rows if r['task_type'] == d] for d in ('harmful', 'benign')}
        assert all(len(v) == 1000 for v in pools.values())
        for v in pools.values(): rng.shuffle(v)
        batches = []
        for i in range(0, 1000, 4):
            batch_rows = pools['harmful'][i:i+4] + pools['benign'][i:i+4]
            rng.shuffle(batch_rows)
            batches.append(batch_rows)
    assert digest([[r['pair_id'] for r in b] for b in batches]) == config(root)['expected_schedule_sha256'][name]
    return batches


def pause(root):
    assert not (Path(root)/'user_pause.json').exists(), 'User paused this experiment'


def model_path(root, size):
    proof = json.loads((Path(root)/'state'/f'{size}.download.json').read_text())
    assert proof['status'] == 'complete'
    assert all(proof[k] == config(root)['models'][size][k] for k in ('model', 'revision'))
    path = Path(proof['snapshot'])
    for name, info in proof['files'].items(): assert (path/name).stat().st_size == info['bytes']
    return path


def encode(tok, prompt, frozen):
    assert frozen['representation'] == 'final_text'
    text = frozen['original_text']
    assert text.strip() and '<think>' not in text and '</think>' not in text
    prefix = tok.apply_chat_template([dict(role='system', content=SYSTEM), dict(role='user', content=prompt)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prefix.endswith('<think>\n\n</think>\n\n'), 'Unexpected native Qwen3 non-thinking prefix'
    context = tok.encode(prefix, add_special_tokens=False)
    answer = tok.encode(text, add_special_tokens=False)
    assert tok.eos_token_id not in answer and answer
    answer.append(tok.eos_token_id)
    assert len(context) + len(answer) <= 8192, 'No truncation allowed'
    return dict(ids=context+answer, start=len(context), target_tokens=len(answer),
                serialized_sha256=digest([context, answer]), source_text_sha256=digest(text))


def logps(model, encoded):
    import torch
    dev = next(model.parameters()).device
    ids = torch.tensor([encoded['ids']], device=dev, dtype=torch.long)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda', cache_enabled=False):
        logits = model(input_ids=ids, use_cache=False).logits[0, encoded['start']-1:-1]
    targets = ids[0, encoded['start']:]
    return torch.cat([x.float().gather(1, y[:, None]).squeeze(1) - x.float().logsumexp(-1)
        for x, y in zip(logits.split(128), targets.split(128))])


def load_model(root, size):
    import torch
    from transformers import AutoModelForCausalLM
    assert torch.cuda.device_count() == 1
    model = AutoModelForCausalLM.from_pretrained(model_path(root, size), dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True).to('cuda')
    assert model.config.model_type == 'qwen3'
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    return model


def backward_dpo(loss, positive, negative):
    """Exact first-order chain rule; release one answer graph before the other."""
    import torch
    gp, gn = torch.autograd.grad(loss, (positive, negative), retain_graph=True)
    positive.backward(gp)
    negative.backward(gn)


def output(root, size, name, smoke=False):
    cfg = config(root)
    return Path(cfg['storage'])/('smoke' if smoke else 'training')/size/name


def learning_factor(step, total):
    warmup = max(1, int(total * .05))
    return step/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
