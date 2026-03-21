"""
mechanistic_hooks.py
====================
Clase para extraer activaciones internas de VideoMAEv2 (ViT-Giant) sin
modificar el código fuente del modelo.

JUSTIFICACIÓN ARQUITECTÓNICA:
-----------------------------
En interpretabilidad mecanicista, el "Residual Stream" (la salida de cada bloque
completo del Transformer) es el punto principal de extracción porque acumula
toda la información procesada hasta ese punto. Es el estado completo del modelo.

Adicionalmente, extraemos las salidas de Attention y MLP por separado para
determinar cuál de los dos componentes es más responsable de codificar la
propiedad física de velocidad:
  - Attention Output: captura relaciones espaciotemporales entre tokens
    (parches del video). Potencialmente codifica movimiento relativo.
  - MLP Output: aplica transformaciones no lineales que, según la literatura
    de interpretabilidad en LLMs (ref: notebooks ARENA), tienden a codificar
    propiedades semánticas y conceptuales.

CÓMO FUNCIONAN LOS HOOKS:
--------------------------
PyTorch `register_forward_hook(fn)` registra una función callback que se
ejecuta DESPUÉS del forward pass del módulo, sin modificar el grafo
computacional original. El hook recibe (module, input, output) y puede
leer `output` sin alterar la ejecución del modelo.

Esto nos permite:
  1. No tocar `modeling_finetune.py` ni `run_class_finetuning.py`
  2. Capturar activaciones intermedias en modo eval()
  3. Seleccionar qué capas extraer para no saturar la memoria

DIMENSIONES DE TENSORES (ViT-Giant):
-------------------------------------
  Entrada:     [B, 3, 16, 224, 224]  → video de B batches, 3 canales, 16 frames
  PatchEmbed:  [B, N, 1408]          → N = (16//2) × (224//14)² = 8 × 256 = 2048 tokens
  Cada bloque: [B, 2048, 1408]       → residual stream (no cambia de forma)
  Attn output: [B, 2048, 1408]       → salida de CosAttention
  MLP output:  [B, 2048, 1408]       → salida de Mlp (fc1→GELU→fc2)
  Pooled:      [B, 1408]             → mean pooling sobre los N tokens

ARQUITECTURA DEL ViT-Giant:
  - depth = 40 bloques (índices 0 a 39)
  - embed_dim = 1408
  - num_heads = 16
  - CosAttention (escalado learnable, no dot-product estándar)
  - LayerScale con gamma_1/gamma_2

USO:
  from mechanistic_hooks import VideoMAEActivationExtractor
  extractor = VideoMAEActivationExtractor(model, layers=[0, 8, 16, 24, 32, 39])
  output = model(video_tensor)  # hooks capturan automáticamente
  activations = extractor.get_activations()       # dict de [B, N, 1408]
  pooled = extractor.get_pooled_activations()      # dict de [B, 1408]
  extractor.clear_activations()                    # limpiar para el siguiente batch
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


# Capas estratégicas por defecto: distribución uniforme sobre los 40 bloques
# Temprano (0-8): features de bajo nivel (bordes, textura, movimiento local)
# Medio (12-24): representaciones espaciotemporales abstractas
# Tardío (28-39): codificación semántica y de acción
DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]


class VideoMAEActivationExtractor:
    """
    Envuelve un modelo VideoMAEv2 (VisionTransformer) para extraer activaciones
    internas usando forward hooks, sin modificar el código del modelo.

    Extrae 3 tipos de activación por capa:
      - block_{i}_residual: salida completa del bloque (residual stream)
      - block_{i}_attn:     salida del módulo de atención (CosAttention)
      - block_{i}_mlp:      salida del módulo MLP (fc1→GELU→fc2)
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
            model: Modelo VisionTransformer de VideoMAEv2 ya cargado.
                   Debe tener atributo `model.blocks` (nn.ModuleList de Block).
            layers: Lista de índices de bloques a monitorear.
                    Default: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]
            detach: Si True, desacopla los tensores del grafo computacional
                    para evitar acumular gradientes en memoria.
            to_cpu: Si True, mueve los tensores a CPU para liberar VRAM.
        """
        self.model = model
        self.layers = layers or DEFAULT_LAYERS
        self.detach = detach
        self.to_cpu = to_cpu

        # Almacenamiento de activaciones capturadas por los hooks
        # Formato: {"block_0_residual": tensor, "block_0_attn": tensor, ...}
        self._activations: Dict[str, torch.Tensor] = {}

        # Lista de handles para poder remover los hooks después
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        # Validar que el modelo tiene la estructura esperada
        if not hasattr(model, "blocks"):
            raise ValueError(
                "El modelo no tiene atributo 'blocks'. "
                "¿Es un VisionTransformer de modeling_finetune.py?"
            )

        num_blocks = len(model.blocks)
        # num_blocks: 40 para ViT-Giant
        for layer_idx in self.layers:
            if layer_idx < 0 or layer_idx >= num_blocks:
                raise ValueError(
                    f"Índice de capa {layer_idx} fuera de rango. "
                    f"El modelo tiene {num_blocks} bloques (0 a {num_blocks - 1})."
                )

        # Registrar hooks
        self._register_hooks()

    def _make_hook(self, name: str):
        """
        Crea una función hook que almacena la salida de un módulo.

        register_forward_hook llama a fn(module, input, output) después
        de que el módulo procesa su entrada. Capturamos `output` sin
        mutar el grafo computacional original.

        Args:
            name: Nombre clave para almacenar la activación (ej. "block_0_residual")
        """
        def hook_fn(module, input, output):
            # output: [B, N, 1408] para bloques, attn y mlp del ViT-Giant
            activation = output
            if self.detach:
                activation = activation.detach()
            if self.to_cpu:
                activation = activation.cpu()
            self._activations[name] = activation
        return hook_fn

    def _register_hooks(self):
        """
        Registra forward hooks en los bloques seleccionados.

        Para cada capa i, registra hooks en:
          - model.blocks[i]      → captura el residual stream completo
          - model.blocks[i].attn → captura solo la salida de CosAttention
          - model.blocks[i].mlp  → captura solo la salida del MLP
        """
        for layer_idx in self.layers:
            block = self.model.blocks[layer_idx]

            # Hook en el bloque completo → Residual Stream
            # Después de: x = x + drop_path(gamma_1 * attn(norm1(x)))
            #             x = x + drop_path(gamma_2 * mlp(norm2(x)))
            h = block.register_forward_hook(
                self._make_hook(f"block_{layer_idx}_residual")
            )
            self._hook_handles.append(h)

            # Hook en atención → solo CosAttention output
            # Captura la salida ANTES de la conexión residual y LayerScale
            h = block.attn.register_forward_hook(
                self._make_hook(f"block_{layer_idx}_attn")
            )
            self._hook_handles.append(h)

            # Hook en MLP → solo MLP output
            # Captura la salida ANTES de la conexión residual y LayerScale
            h = block.mlp.register_forward_hook(
                self._make_hook(f"block_{layer_idx}_mlp")
            )
            self._hook_handles.append(h)

        print(
            f"[VideoMAEActivationExtractor] Registrados {len(self._hook_handles)} hooks "
            f"en {len(self.layers)} capas: {self.layers}"
        )

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Retorna las activaciones capturadas sin modificar.

        Returns:
            Dict con claves como "block_0_residual", "block_0_attn", etc.
            Cada valor es un tensor de forma [B, N, 1408]
            donde N = 2048 para entrada estándar (16 frames, 224×224)
        """
        return self._activations.copy()

    def get_pooled_activations(self) -> Dict[str, torch.Tensor]:
        """
        Retorna las activaciones con mean pooling sobre los N tokens.
        Esto reduce [B, N, 1408] → [B, 1408], creando una representación
        global del video por capa, lista para entrenar un clasificador lineal.

        El mean pooling sobre tokens es la estrategia estándar de reducción
        en ViTs: fc_norm(x.mean(1)) es exactamente lo que VideoMAEv2 hace
        en su forward_features() con use_mean_pooling=True.

        Returns:
            Dict con las mismas claves, pero valores de forma [B, 1408]
        """
        pooled = {}
        for name, activation in self._activations.items():
            # activation: [B, N, 1408] → mean sobre dim=1 → [B, 1408]
            pooled[name] = activation.mean(dim=1)
        return pooled

    def clear_activations(self):
        """
        Limpia las activaciones almacenadas para liberar memoria.
        Llamar ENTRE batches para evitar memory leaks.
        """
        self._activations.clear()

    def remove_hooks(self):
        """
        Remueve todos los hooks registrados y limpia las activaciones.
        Llamar cuando ya no se necesite extraer más activaciones.
        """
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self.clear_activations()
        print("[VideoMAEActivationExtractor] Todos los hooks removidos.")

    def __del__(self):
        """Limpia automáticamente los hooks al destruir el objeto."""
        self.remove_hooks()

    def summary(self) -> str:
        """Retorna un resumen de las activaciones capturadas."""
        lines = [f"Activaciones capturadas: {len(self._activations)} tensores"]
        for name, tensor in sorted(self._activations.items()):
            lines.append(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
        return "\n".join(lines)
