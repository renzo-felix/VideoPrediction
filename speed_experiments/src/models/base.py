import torch

class ActivationExtractor:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activations = {}
        self.hooks = []

        block = self.model.blocks[layer_idx]
        hook = block.register_forward_hook(self._hook_fn)
        self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        self.activations[self.layer_idx] = output.detach().cpu()

    def __call__(self, x):
        self.activations = {}
        with torch.no_grad():
            _ = self.model(x)
        return self.activations

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
