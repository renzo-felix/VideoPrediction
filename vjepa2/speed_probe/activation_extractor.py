"""
activation_extractor.py
========================
Extrae activaciones internas de V-JEPA 2 usando forward hooks.
Basado en el trabajo de Luis (VideoMAEv2_Proyecto_Paper/mechanistic_hooks.py).

Extrae 3 tipos por capa:
  block_{i}_residual → salida completa del bloque
  block_{i}_attn     → salida de RoPEAttention
  block_{i}_mlp      → salida de MLP/SwiGLU

Dimensiones (ViT-L, 16 frames, 256px):
  [B, N, 1024]  donde N = 8×16×16 = 2048 tokens
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


DEFAULT_LAYERS = {
    24: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23],   # ViT-L
    32: [0, 4, 8, 12, 16, 20, 24, 28, 31],              # ViT-H
    40: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39],      # ViT-G
}


class VJEPA2ActivationExtractor:
    def __init__(self, model: nn.Module, layers: Optional[List[int]] = None,
                 detach: bool = True, to_cpu: bool = True):
        self.model = model
        self.detach = detach
        self.to_cpu = to_cpu
        self._activations: Dict[str, torch.Tensor] = {}
        self._handles: List = []

        if not hasattr(model, "blocks"):
            raise ValueError("El modelo no tiene 'blocks'. ¿Es un VisionTransformer de vjepa2?")

        num_blocks = len(model.blocks)
        self.embed_dim = model.embed_dim
        self.layers = layers if layers is not None else DEFAULT_LAYERS.get(
            num_blocks, sorted(set([0] + list(range(0, num_blocks, max(1, num_blocks//11))) + [num_blocks-1]))
        )

        for i in self.layers:
            if not (0 <= i < num_blocks):
                raise ValueError(f"Capa {i} fuera de rango [0, {num_blocks-1}]")

        self._register_hooks()

    def _hook(self, name):
        def fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            if self.detach: act = act.detach()
            if self.to_cpu:  act = act.cpu()
            self._activations[name] = act
        return fn

    def _register_hooks(self):
        for i in self.layers:
            block = self.model.blocks[i]
            self._handles.append(block.register_forward_hook(self._hook(f"block_{i}_residual")))
            self._handles.append(block.attn.register_forward_hook(self._hook(f"block_{i}_attn")))
            self._handles.append(block.mlp.register_forward_hook(self._hook(f"block_{i}_mlp")))
        print(f"[Extractor] {len(self._handles)} hooks en capas {self.layers} (embed_dim={self.embed_dim})")

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self._activations.copy()

    def get_pooled_activations(self) -> Dict[str, torch.Tensor]:
        """Mean pool sobre tokens: [B, N, D] → [B, D]"""
        return {name: act.mean(dim=1) for name, act in self._activations.items()}

    def clear_activations(self):
        self._activations.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.clear_activations()
        print("[Extractor] Hooks removidos.")

    def __del__(self):
        try: self.remove_hooks()
        except: pass
