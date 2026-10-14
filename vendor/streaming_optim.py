"""Same FP32 torch AdamW updates, with only one CPU gradient resident at a time."""
import torch
from safety_pair_v3_optim import CPUAdamW

class StreamingCPUAdamW(CPUAdamW):
    @torch.no_grad()
    def step(self):
        # AdamW updates are parameter-local after global gradient clipping.
        # Keep the original optimizer, states and group hyperparameters so
        # scheduler and checkpoint serialization retain their existing meaning.
        assert len(self.optimizer.param_groups) == 1
        group = self.optimizer.param_groups[0]
        parameters = group['params']
        try:
            for source, master in zip(self.device_parameters, self.cpu_parameters):
                if source.grad is None:
                    continue
                master.grad = source.grad.detach().to('cpu', copy=True)
                source.grad = None
                group['params'] = [master]
                self.optimizer.step()
                master.grad = None
                source.copy_(master)
        finally:
            group['params'] = parameters

    def checkpoint_model(self, model):
        """Reuse synchronized FP32 CPU masters instead of copying GPU weights."""
        masters = {id(p): master.detach() for p, master in
                   zip(self.device_parameters, self.cpu_parameters)}
        named = dict(model.named_parameters())
        return {name: masters[id(named[name])] if name in named and id(named[name]) in masters
                else value.detach().cpu() for name, value in model.state_dict().items()}
