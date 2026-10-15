"""Frozen text segments and native R1 continuations, shared by all objectives."""
import bisect
import re

from safety_pair_v3_common import SYSTEM, digest

ENCODING_PROTOCOL = 'r1_native_continuation_explicit_empty_think_v1'
SEGMENT_PROTOCOL = 'paragraph_sentence_800chars_v1'


def continuation(response, representation):
    """Preserve supplied reasoning; final-only sources get an empty think block.

    The native generation prefix already ends with <think>\n. Full assistant
    chat templating is deliberately not used: this model's template strips
    everything before </think> from assistant messages.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError('empty_response')
    if representation == 'final_text':
        if '<think>' in response or '</think>' in response:
            raise ValueError('final_text_contains_reasoning_markup_requires_review')
        return '</think>\n\n' + response
    if representation not in ['r1_continuation', 'r1_raw_continuation']:
        raise ValueError('unknown_response_representation')
    text = response
    if text.startswith('<think>\n'):
        text = text[len('<think>\n'):]
    elif text.startswith('<think>'):
        text = text[len('<think>'):]
    if representation == 'r1_raw_continuation':
        # A model can terminate with a bare refusal without closing its think
        # tag. Preserve the actual continuation; completion/behavior is judged
        # separately, and a missing tag alone must not redefine over-refusal.
        if not text.strip() or '</think>' in text or '<think>' in text:
            raise ValueError('unexpected_raw_continuation_markup')
        return text
    if text.count('</think>') != 1 or '<think>' in text:
        raise ValueError('incomplete_or_malformed_reasoning_markup')
    if not text.split('</think>', 1)[1].strip():
        raise ValueError('missing_final_answer')
    return text


def text_segments(text, max_chars=800):
    """Contiguous character spans, fixed before model-specific tokenization.

    Prefer paragraph/list boundaries. Split oversized paragraphs at sentence
    endings, then whitespace; hard character splits are recorded fallbacks.
    No verifier rewards, negative-text alignment, or training outcomes are used.
    """
    if not text or max_chars < 32:
        raise ValueError('invalid_segment_input')
    ends = {m.end() for m in re.finditer(r'\n\s*\n|\n(?=\s*(?:[-*]|\d+[.)])\s)|</think>\s*', text)}
    ends = {end for end in ends if text[:end].strip() != '</think>'}
    ends.add(len(text))
    spans = []
    start = 0
    for end in sorted(ends):
        if end <= start:
            continue
        while end - start > max_chars:
            window = text[start:start + max_chars]
            choices = [m.end() for m in re.finditer(r'[.!?。！？][\"\u201d\u2019\x27)]*\s+', window)]
            choices = [x for x in choices if x >= max_chars // 3]
            if choices:
                cut, rule = choices[-1], 'sentence'
            else:
                cut = max(window.rfind(' '), window.rfind('\n')) + 1
                if cut < max_chars // 3:
                    cut, rule = max_chars, 'hard_character_fallback'
                else:
                    rule = 'whitespace'
            spans.append(dict(start=start, end=start + cut, rule=rule))
            start += cut
        spans.append(dict(start=start, end=end, rule='paragraph_or_section'))
        start = end
    assert spans[0]['start'] == 0 and spans[-1]['end'] == len(text)
    assert all(a['end'] == b['start'] for a, b in zip(spans, spans[1:]))
    return spans


def freeze_response(response, representation):
    text = continuation(response, representation)
    return dict(original_text=response, representation=representation,
                continuation=text, text_sha256=digest(text),
                segments=text_segments(text), encoding_protocol=ENCODING_PROTOCOL,
                segment_protocol=SEGMENT_PROTOCOL)


def encode_response(tokenizer, prompt, frozen, max_length=8192):
    if frozen['encoding_protocol'] != ENCODING_PROTOCOL or frozen['segment_protocol'] != SEGMENT_PROTOCOL:
        raise ValueError('serialization_protocol_mismatch')
    text = continuation(frozen['original_text'], frozen['representation'])
    if text != frozen['continuation'] or digest(text) != frozen['text_sha256']:
        raise ValueError('frozen_text_changed')
    if frozen['segments'] != text_segments(text):
        raise ValueError('frozen_segments_changed')
    prefix = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True)
    if not prefix.endswith('<think>\n'):
        raise ValueError('unexpected_native_R1_generation_prefix')
    context = tokenizer.encode(prefix, add_special_tokens=False)
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    answer = list(tokenized['input_ids'])
    if tokenizer.eos_token_id in answer:
        raise ValueError('embedded_end_of_turn_requires_review')
    ends = [s['end'] for s in frozen['segments']]
    # A token crossing a character boundary belongs to the span where it ends.
    segment_ids = [min(bisect.bisect_right(ends, max(0, end - 1)), len(ends)-1)
                   for _, end in tokenized['offset_mapping']]
    answer.append(tokenizer.eos_token_id)
    segment_ids.append(segment_ids[-1] if segment_ids else 0)
    # A short span may contain no complete tokens. Collapse only empty token
    # groups; the frozen text spans remain unchanged in the bank.
    mapping = {s: i for i, s in enumerate(sorted(set(segment_ids)))}
    segment_ids = [mapping[s] for s in segment_ids]
    if len(context) + len(answer) > max_length:
        raise ValueError('context_overflow_no_truncation')
    return dict(ids=context + answer, start=len(context), segment_ids=segment_ids,
                serialized_sha256=digest([context, answer]),
                token_segment_to_text_segment=sorted(mapping),
                target_tokens=len(answer))


def target_logps(model, item):
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([item['ids']], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = model(input_ids=ids, use_cache=False).logits[0, item['start']-1:-1]
    targets = ids[0, item['start']:]
    # Chunk the FP32 normalization to avoid a second full sequence x vocab copy.
    return torch.cat([x.float().gather(1, y[:, None]).squeeze(1) - x.float().logsumexp(-1)
                      for x, y in zip(logits.split(128), targets.split(128))])
