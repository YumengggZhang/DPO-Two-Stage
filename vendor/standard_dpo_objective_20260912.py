"""Original sigmoid DPO: sequence sums, frozen reference, no outer scaling."""
import math

ARM = 'standard_dpo_beta_0p1'
BETA = 0.1
VERSION = 'standard_sequence_sum_dpo_20260912_v1'


def standard_dpo_loss(positive_logps, negative_logps, reference_positive_sum,
                      reference_negative_sum, beta=BETA):
    import torch
    import torch.nn.functional as F
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('DPO beta must be finite and positive')
    if any(x.ndim != 1 or not x.numel() for x in (positive_logps, negative_logps)):
        raise ValueError('Require nonempty response token vectors')
    rp = torch.as_tensor(reference_positive_sum, device=positive_logps.device,
                         dtype=torch.float32).detach()
    rn = torch.as_tensor(reference_negative_sum, device=positive_logps.device,
                         dtype=torch.float32).detach()
    if rp.numel() != 1 or rn.numel() != 1:
        raise ValueError('Reference scores must be sequence sums')
    margin = (positive_logps.float().sum() - negative_logps.float().sum()) - (rp - rn)
    return -F.logsigmoid(beta * margin)
