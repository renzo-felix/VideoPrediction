"""
vjepa2_activation_extractor.py
================================
Extrae activaciones internas de V-JEPA 2 (VisionTransformer) usando forward hooks,
sin modificar el código fuente del modelo.

Adaptado de mechanistic_hooks.py (VideoMAEv2) para la arquitectura de V-JEPA 2.

DIFERENCIAS CLAVE VS VideoMAEv2:
---------------------------------
  VideoMAEv2 ViT-Giant:   depth=40, embed_dim=1408, CosAttention
  V-JEPA 2  ViT-Large:    depth=24, embed_dim=1024, RoPEAttention
  V-JEPA 2  ViT-Huge:     depth=32, embed_dim=1280, RoPEAttention
  V-JEPA 2  ViT-Giant:    depth=40, embed_dim=1408, RoPEAttention

  Ambas arquitecturas comparten la misma estructura de bloques:
    model.blocks[i]      → bloque completo (residual stream)
    model.blocks[i].attn → módulo de atención (Attention / RoPEAttention)
    model.blocks[i].mlp  → módulo MLP (MLP / SwiGLUFFN)
  Por eso los hooks funcionan igual en ambas.

  Normalización de entrada:
    VideoMAEv2: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    V-JEPA 2:   mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  (ImageNet)

DIMENSIONES DE TENSORES (V-JEPA 2 ViT-Large, 16 frames, 256px):
-----------------------------------------------------------------
  Entrada:     [B, 3, 16, 256, 256]  → video: batch, canales, frames, H, W
  PatchEmbed3D:[B, N, 1024]          → N = (16//2) × (256//16)² = 8 × 256 = 2048 tokens
  Cada bloque: [B, 2048, 1024]       → residual stream
  Attn output: [B, 2048, 1024]       → salida de RoPEAttention
  MLP output:  [B, 2048, 1024]       → salida de MLP/SwiGLU
  Pooled:      [B, 1024]             → mean pooling sobre tokens

CHECKPOINT KEY:
  Checkpoints de preentrenamiento (guardados por trainer): "target_encoder"
  Checkpoints del release público (vitl.pt, vitg.pt):      "encoder"
  El extractor soporta ambos automáticamente.

USO:
  from vjepa2_activation_extractor import VJEPA2ActivationExtractor
  extractor = VJEPA2ActivationExtractor(model, layers=[0, 6, 12, 18, 23])
  output = model(video_tensor)          # hooks capturan automáticamente
  activations = extractor.get_activations()       # dict de [B, N, D]
  pooled = extractor.get_pooled_activations()     # dict de [B, D]
  extractor.clear_activations()                   # limpiar entre batches
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


# Capas estratégicas por defecto para cada variante de ViT
# Distribución uniforme: temprano / medio / tardío
DEFAULT_LAYERS_VITL = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]   # 24 bloques
DEFAULT_LAYERS_VITH = [0, 4, 8, 12, 16, 20, 24, 28, 31]              # 32 bloques
DEFAULT_LAYERS_VITG = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]      # 40 bloques


def get_default_layers(num_blocks: int) -> List[int]:
    """Retorna capas por defecto según la profundidad del modelo."""
    if num_blocks == 24:
        return DEFAULT_LAYERS_VITL
    elif num_blocks == 32:
        return DEFAULT_LAYERS_VITH
    elif num_blocks == 40:
        return DEFAULT_LAYERS_VITG
    else:
        # Distribución uniforme genérica: ~12 capas
        step = max(1, num_blocks // 12)
        layers = list(range(0, num_blocks - 1, step))
        if (num_blocks - 1) not in layers:
            layers.append(num_blocks - 1)
        return layers


class VJEPA2ActivationExtractor:
    """
    Envuelve un modelo V-JEPA 2 (VisionTransformer) para extraer activaciones
    internas usando forward hooks, sin modificar el código del modelo.

    Extrae 3 tipos de activación por capa:
      - block_{i}_residual: salida completa del bloque (residual stream)
      - block_{i}_attn:     salida del módulo de atención (RoPEAttention)
      - block_{i}_mlp:      salida del módulo MLP (MLP o SwiGLUFFN)

    Compatible con:
      - V-JEPA 2 ViT-Large  (depth=24, embed_dim=1024)
      - V-JEPA 2 ViT-Huge   (depth=32, embed_dim=1280)
      - V-JEPA 2 ViT-Giant  (depth=40, embed_dim=1408)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        detach: bool = True,
        to_cpu: bool = True,
    ):
        """
        Args:
            model: Modelo VisionTransformer de V-JEPA 2 ya cargado en eval().
                   Debe tener atributo `model.blocks` (nn.ModuleList de Block).
            layers: Índices de bloques a monitorear.
                    Default: distribución uniforme según profundidad del modelo.
            detach: Si True, desacopla tensores del grafo para no acumular gradientes.
            to_cpu: Si True, mueve tensores a CPU para liberar VRAM entre batches.
        """
        self.model = model
        self.detach = detach
        self.to_cpu = to_cpu

        # Almacenamiento de activaciones capturadas por hooks
        # Formato: {"block_0_residual": tensor, "block_0_attn": tensor, ...}
        self._activations: Dict[str, torch.Tensor] = {}

        # Handles para poder remover hooks después
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        # Validar estructura del modelo
        if not hasattr(model, "blocks"):
            raise ValueError(
                "El modelo no tiene atributo 'blocks'. "
                "¿Es un VisionTransformer de src/models/vision_transformer.py?"
            )

        num_blocks = len(model.blocks)
        self.embed_dim = model.embed_dim
        self.num_blocks = num_blocks

        # Seleccionar capas
        self.layers = layers if layers is not None else get_default_layers(num_blocks)

        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= num_blocks:
                raise ValueError(
                    f"Índice de capa {layer_idx} fuera de rango [0, {num_blocks - 1}]."
                )

        self._register_hooks()

    def _make_hook(self, name: str):
        """
        Crea un hook que almacena la salida (output) del módulo registrado.

        Para Block (residual stream): output es el tensor x actualizado [B, N, D]
        Para attn:  output es el tensor de atención [B, N, D]
        Para mlp:   output es el tensor del MLP [B, N, D]

        Nota: Block.forward devuelve un tensor directamente (no una tupla),
        igual que los sub-módulos attn y mlp.
        """
        def hook_fn(module, input, output):
            # Algunos módulos pueden devolver tupla (attn_output, attn_weights)
            # En V-JEPA 2 los módulos attn y mlp devuelven tensores simples
            activation = output[0] if isinstance(output, tuple) else output
            if self.detach:
                activation = activation.detach()
            if self.to_cpu:
                activation = activation.cpu()
            self._activations[name] = activation
        return hook_fn

    def _register_hooks(self):
        """
        Registra hooks en los bloques seleccionados.

        Para cada capa i:
          - model.blocks[i]      → residual stream completo (post-attn + post-mlp)
          - model.blocks[i].attn → solo la salida de RoPEAttention (pre-residual)
          - model.blocks[i].mlp  → solo la salida de MLP/SwiGLU (pre-residual)
        """
        for layer_idx in self.layers:
            block = self.model.blocks[layer_idx]

            # Hook en bloque completo → Residual Stream
            h = block.register_forward_hook(
                self._make_hook(f"block_{layer_idx}_residual")
            )
            self._hook_handles.append(h)

            # Hook en atención → RoPEAttention output
            h = block.attn.register_forward_hook(
                self._make_hook(f"block_{layer_idx}_attn")
            )
            self._hook_handles.append(h)

            # Hook en MLP → MLP/SwiGLUFFN output
            h = block.mlp.register_forward_hook(
                self._make_hook(f"block_{layer_idx}_mlp")
            )
            self._hook_handles.append(h)

        print(
            f"[VJEPA2ActivationExtractor] {len(self._hook_handles)} hooks registrados "
            f"en {len(self.layers)} capas: {self.layers} "
            f"(modelo: {self.num_blocks} bloques, embed_dim={self.embed_dim})"
        )

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Retorna las activaciones capturadas.

        Returns:
            Dict con claves "block_{i}_residual", "block_{i}_attn", "block_{i}_mlp".
            Cada tensor tiene forma [B, N, D] donde:
              N = num_tokens (ej. 2048 para 16f, 256px, ViT-L)
              D = embed_dim (1024 para ViT-L)
        """
        return self._activations.copy()

    def get_pooled_activations(self) -> Dict[str, torch.Tensor]:
        """
        Retorna activaciones con mean pooling sobre los N tokens.

        Reduce [B, N, D] → [B, D], creando una representación global del video
        lista para entrenar un clasificador lineal.

        Este es el mismo pooling que V-JEPA 2 usa internamente en su forward pass:
        x = x.mean(dim=1)  antes de la norma final.

        Returns:
            Dict con las mismas claves, pero tensores de forma [B, D].
        """
        pooled = {}
        for name, activation in self._activations.items():
            # [B, N, D] → mean sobre dim=1 → [B, D]
            pooled[name] = activation.mean(dim=1)
        return pooled

    def clear_activations(self):
        """Limpia activaciones almacenadas. Llamar entre batches."""
        self._activations.clear()

    def remove_hooks(self):
        """Remueve todos los hooks y limpia activaciones."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self.clear_activations()
        print("[VJEPA2ActivationExtractor] Todos los hooks removidos.")

    def __del__(self):
        """Limpia automáticamente al destruir el objeto."""
        try:
            self.remove_hooks()
        except Exception:
            pass

    def summary(self) -> str:
        """Resumen de activaciones capturadas."""
        lines = [f"Activaciones capturadas: {len(self._activations)} tensores"]
        for name, tensor in sorted(self._activations.items()):
            lines.append(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
        return "\n".join(lines)
