"""FP32 parameters, gradients and AdamW moments resident on the assigned GPU."""
import torch


class GPUAdamW:
    def __init__(self, model, lr=1e-5, weight_decay=1e-4):
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert parameters and all(p.dtype == torch.float32 and p.is_cuda for p in parameters)
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay,
                                           foreach=False)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        for parameter, values in self.optimizer.state.items():
            for key in ('exp_avg', 'exp_avg_sq'):
                assert values[key].device == parameter.device and values[key].dtype == torch.float32

    @torch.no_grad()
    def checkpoint_model(self, model):
        # Only checkpoint serialization copies the model to host memory.
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}
