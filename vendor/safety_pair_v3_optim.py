"""FP32 CPU AdamW states with FP32 model parameters and BF16 forward autocast."""
import torch


class CPUAdamW:
    """Offload optimizer storage without rounding persistent parameters to BF16."""
    def __init__(self, model, lr=1e-5, weight_decay=1e-4):
        self.device_parameters = [p for p in model.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in self.device_parameters):
            raise ValueError('FP32 model parameters required')
        self.cpu_parameters = [torch.nn.Parameter(p.detach().cpu().clone())
                               for p in self.device_parameters]
        self.optimizer = torch.optim.AdamW(self.cpu_parameters, lr=lr,
                                           weight_decay=weight_decay, foreach=False)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.device_parameters:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for source, master in zip(self.device_parameters, self.cpu_parameters):
            master.grad = source.grad.detach().to('cpu', copy=True) if source.grad is not None else None
            source.grad = None
        self.optimizer.step()
        for target, master in zip(self.device_parameters, self.cpu_parameters):
            target.copy_(master)

    @torch.no_grad()
    def refresh_from_model(self):
        for source, master in zip(self.device_parameters, self.cpu_parameters):
            master.copy_(source.detach().cpu())

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        self.refresh_from_model()
