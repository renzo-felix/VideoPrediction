import torch

class ActivationExtractor:
    def __init__(self, model, layer_idx, model_name='vjepa'):
        self.model = model
        self.layer_idx = layer_idx
        self.model_name = model_name
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
            if self.model_name == 'videomae':
                B = x.shape[0]
                num_patches = self.model.patch_embed.num_patches
                mask = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
                _ = self.model(x, mask=mask)
            else:
                _ = self.model(x)
        return self.activations

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def preprocess_frames(x, model_name='vjepa'):
    # x is (B, T, C, H, W) -> need (B, C, T, H, W)
    if x.dim() == 5:
        return x.permute(0, 2, 1, 3, 4)
    return x
