"""FP32 AdamW with GPU0 model/gradients and GPU1 moments and update arithmetic.

Only one temporary parameter and gradient are copied to GPU1 at a time. All
moments stay on GPU1 between updates. CPU transfers occur for checkpoints only.
The scalar step counter is the ordinary non-capturable torch AdamW counter.
"""
import torch


class GPUAdamW:
    def __init__(self, model, lr=1e-5, weight_decay=1e-4):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        assert torch.cuda.device_count() == 2
        assert self.parameters and all(p.dtype == torch.float32 and p.device == torch.device('cuda:0') for p in self.parameters)
        self.proxies = [torch.nn.Parameter(torch.empty(0, device='cuda:1')) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.proxies, lr=lr, weight_decay=weight_decay, foreach=False)

    def zero_grad(self):
        for p in self.parameters: p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        assert len(self.optimizer.param_groups) == 1
        group = self.optimizer.param_groups[0]
        try:
            for p, proxy in zip(self.parameters, self.proxies):
                if p.grad is None: continue
                proxy.data = p.detach().to('cuda:1')
                proxy.grad = p.grad.detach().to('cuda:1')
                group['params'] = [proxy]
                self.optimizer.step()
                p.copy_(proxy)
                # Synchronize the cross-device copy before releasing its source.
                torch.cuda.synchronize(0)
                proxy.grad = None
                proxy.data = torch.empty(0, device='cuda:1')
                p.grad = None
        finally:
            group['params'] = self.proxies

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        for p, proxy in zip(self.parameters, self.proxies):
            for key in ('exp_avg', 'exp_avg_sq'):
                value = self.optimizer.state[proxy][key]
                assert value.shape == p.shape and value.dtype == torch.float32
                assert value.device == torch.device('cuda:1')

    @torch.no_grad()
    def checkpoint_model(self, model):
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}
