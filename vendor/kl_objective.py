"""Full-vocabulary forward KL on fixed answer prefixes; no sampled-token surrogate."""
import torch


def teacher_key(domain):
    assert domain in ('harmful', 'benign'), domain
    return 'pi0' if domain == 'benign' else 'pi1'


def answer_slice(logits, encoded):
    start = encoded['start']
    assert 0 < start < len(encoded['ids'])
    assert encoded['target_tokens'] == len(encoded['ids']) - start
    assert logits.shape[:2] == (1, len(encoded['ids']))
    # Position start-1 predicts the first answer token; the last predicts EOS.
    return logits[0, start - 1:-1]


class ForwardKL(torch.autograd.Function):
    """Exact KL and analytic first-order gradient, with bounded FP32 work buffers.

    Store BF16 forward logits; evaluate every vocabulary entry in FP32 chunks.
    No teacher gradient and no top-k approximation. Higher-order derivatives
    are not used in this experiment.
    """
    @staticmethod
    def forward(ctx, student, teacher, chunk):
        assert student.ndim == 2 and student.shape == teacher.shape
        assert student.device == teacher.device
        ctx.save_for_backward(student, teacher.detach())
        ctx.chunk = chunk
        values = []
        for s, t in zip(student.split(chunk), teacher.split(chunk)):
            logq = s.float().log_softmax(-1)
            logp = t.float().log_softmax(-1)
            values.append((logp.exp() * (logp - logq)).sum(-1))
        return torch.cat(values)

    @staticmethod
    def backward(ctx, gradient):
        student, teacher = ctx.saved_tensors
        grad = torch.empty_like(student)
        for i in range(0, len(student), ctx.chunk):
            q = student[i:i+ctx.chunk].float().softmax(-1)
            p = teacher[i:i+ctx.chunk].float().softmax(-1)
            grad[i:i+ctx.chunk] = ((q - p) * gradient[i:i+ctx.chunk, None]).to(student.dtype)
        return grad, None, None


def token_kl(student, teacher):
    return ForwardKL.apply(student, teacher, 64)


def answer_kl(student_model, teacher_model, encoded):
    device = next(student_model.parameters()).device
    assert device.type == 'cuda'
    assert not any(p.requires_grad for p in teacher_model.parameters())
    ids = torch.tensor([encoded['ids']], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        teacher = answer_slice(teacher_model(input_ids=ids, use_cache=False).logits, encoded)
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        student = answer_slice(student_model(input_ids=ids, use_cache=False).logits, encoded)
    values = token_kl(student, teacher)
    assert len(values) == encoded['target_tokens']
    # Each answer has unit weight regardless of its length.
    return values.mean()


def self_test():
    torch.manual_seed(2002)
    s = torch.randn(137, 19, requires_grad=True)
    t = torch.randn(137, 19, requires_grad=True)
    weights = torch.rand(137)
    actual = token_kl(s, t)
    expected = torch.nn.functional.kl_div(s.log_softmax(-1), t.detach().softmax(-1), reduction='none').sum(-1)
    assert torch.allclose(actual, expected, atol=5e-7)
    gs, gt = torch.autograd.grad((actual * weights).sum(), (s, t), allow_unused=True)
    ge, = torch.autograd.grad((expected * weights).sum(), (s,))
    assert gt is None and torch.allclose(gs, ge, atol=2e-7)
    assert torch.allclose(token_kl(s, s.detach()), torch.zeros(137), atol=1e-7)
    assert not torch.allclose(actual, token_kl(t, s.detach()))
    a = token_kl(s[:3], t[:3]).mean()
    b = token_kl(s[:3].repeat(4, 1), t[:3].repeat(4, 1)).mean()
    assert torch.allclose(a, b, atol=1e-6)
    logits = torch.arange(30).reshape(1, 10, 3)
    assert torch.equal(answer_slice(logits, dict(ids=list(range(10)), start=4, target_tokens=6)), logits[0, 3:9])
    assert teacher_key('benign') == 'pi0' and teacher_key('harmful') == 'pi1'
    # Both sides have the same positive KL gradient multiplier.
    p = s[:2].detach().clone().requires_grad_()
    n = s[:5].detach().clone().requires_grad_()
    lp, ln = token_kl(p, t[:2]).mean(), token_kl(n, t[:5]).mean()
    gp, gn = torch.autograd.grad((lp + ln) / 2, (lp, ln))
    assert float(gp) == float(gn) == .5
    return dict(full_vocabulary_forward_kl=True, analytic_gradient_matches_autograd=True,
                teacher_has_no_gradient=True, answer_mask_and_shift=True,
                token_mean_length_invariant=True, both_sides_equal_weight=True, domain_routing=True)
