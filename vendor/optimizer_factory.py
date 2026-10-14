"""Preserve each historical model's FP32 AdamW placement."""
def optimizer(model, config, **kwargs):
    kind=config.get('optimizer','gpu')
    if kind=='gpu':
        from gpu_optimizer import GPUAdamW
        return GPUAdamW(model,**kwargs)
    if kind=='two_gpu':
        from two_gpu_optimizer import GPUAdamW
        return GPUAdamW(model,**kwargs)
    if kind=='streaming_cpu':
        from streaming_optim import StreamingCPUAdamW
        return StreamingCPUAdamW(model,**kwargs)
    assert kind=='cpu'
    from safety_pair_v3_optim import CPUAdamW
    class CheckpointCPUAdamW(CPUAdamW):
        def checkpoint_model(self,model):
            return {k:v.detach().cpu() for k,v in model.state_dict().items()}
    return CheckpointCPUAdamW(model,**kwargs)
