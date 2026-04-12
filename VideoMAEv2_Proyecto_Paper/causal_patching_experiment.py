"""
causal_patching_experiment.py
==============================
Implementa Activation Patching causal sobre VideoMAEv2 (ViT-Giant) para
verificar la relación CAUSAL entre activaciones de capas específicas y la
predicción de velocidad.

JUSTIFICACIÓN ARQUITECTÓNICA:
-----------------------------
El Linear Probing solo muestra CORRELACIÓN: "la velocidad es decodificable
desde la capa X". No demuestra causalidad. El Activation Patching sí:

  1. Toma un video RÁPIDO (A) y un video LENTO (B)
  2. Extrae la activación de la capa L del video A: h_A[L]
  3. Re-ejecuta el video B, pero en la capa L reemplaza la activación con h_A[L]
  4. Si la predicción de B cambia hacia "rápido", entonces la capa L CAUSA
     (no solo correlaciona) la representación de velocidad.

Esto se implementa con PyTorch `register_forward_pre_hook` en el bloque
*siguiente* a la capa que queremos patchear. El pre-hook intercepta el input
del bloque L+1 y lo reemplaza con la activación almacenada de A.

NOTA IMPORTANTE SOBRE EL PATCHING EN RESIDUAL STREAM:
En un ViT, la salida de bloque[i] ES la entrada de bloque[i+1].
Entonces:
  - Para patchear la capa L, registramos un pre_hook en block[L+1]
    que reemplaza su input con la activación de A en block[L].
  - Para patchear la última capa (39), modificamos la entrada a fc_norm.

MÉTRICAS:
  - "Patching Flip Rate": proporción de videos B cuya predicción top-1
    cambia a la clase de A después del patching.
  - "Logit Shift": cambio promedio en el logit de la clase de A.

USO:
  python causal_patching_experiment.py \
      --checkpoint /home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth \
      --csv physical_diagnostics.csv \
      --num_pairs 50 \
      --wandb_project videomae_probing
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import vit_giant_patch14_224
from mechanistic_hooks import VideoMAEActivationExtractor

# Reusar funciones de run_layer_probing
from run_layer_probing import load_model, load_video_frames, load_diagnostic_csv

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ActivationPatcher:
    """
    Implementa Activation Patching causal sobre el residual stream.

    Flujo:
      1. Forward pass del video A → capturar activación en capa L
      2. Registrar pre_hook en capa L+1 que reemplaza el input
      3. Forward pass del video B (con hook activo) → obtener predicción patcheada
      4. Comparar predicción patcheada con predicción original de B
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._patch_hook_handle = None
        self._stored_activation = None

    def _get_source_activation(
        self, video_tensor: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        Extrae la activación del residual stream en la capa `layer_idx`
        para el video fuente (A).

        Args:
            video_tensor: [1, C, T, H, W] tensor del video A
            layer_idx: Índice del bloque (0-39)

        Returns:
            Activación [1, N, 1408] del bloque layer_idx
        """
        activation = {}

        def hook_fn(module, input, output):
            # output: [B, N, 1408] = [1, 2048, 1408]
            activation["value"] = output.detach().clone()

        handle = self.model.blocks[layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(video_tensor)

        handle.remove()
        return activation["value"]

    def _patch_and_forward(
        self,
        video_tensor: torch.Tensor,
        source_activation: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Ejecuta el forward pass del video B con la activación de A inyectada
        en la capa `layer_idx`.

        Implementación:
          - Registra un pre_hook en block[layer_idx+1] que reemplaza su input
            con source_activation.
          - Para la última capa (39), registra un hook en fc_norm.

        Args:
            video_tensor: [1, C, T, H, W] tensor del video B
            source_activation: [1, N, 1408] activación del video A en layer_idx
            layer_idx: Índice del bloque donde se inyecta (0-39)

        Returns:
            logits: [1, num_classes] predicción patcheada
        """
        num_blocks = len(self.model.blocks)

        if layer_idx < num_blocks - 1:
            # Patchear la entrada del bloque SIGUIENTE
            target_block = self.model.blocks[layer_idx + 1]

            def pre_hook_fn(module, input):
                # input es una tupla, reemplazamos el primer elemento
                # input[0]: [B, N, 1408] → reemplazar con source_activation
                return (source_activation,) + input[1:]

            handle = target_block.register_forward_pre_hook(pre_hook_fn)
        else:
            # Última capa: patchear la entrada de self.norm o fc_norm
            # En VisionTransformer.forward_features():
            #   for blk in self.blocks: x = blk(x)
            #   if self.fc_norm is not None: return self.fc_norm(x.mean(1))
            # No podemos hookear fácilmente aquí, así que hookeamos el
            # último bloque directamente para que su salida SEA source_activation
            def hook_fn(module, input, output):
                return source_activation

            handle = self.model.blocks[layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = self.model(video_tensor)

        handle.remove()
        return logits

    def run_patching(
        self,
        video_a: torch.Tensor,
        video_b: torch.Tensor,
        layer_idx: int,
    ) -> dict:
        """
        Ejecuta un experimento completo de patching para un par de videos.

        Args:
            video_a: [1, C, T, H, W] video fuente (rápido)
            video_b: [1, C, T, H, W] video objetivo (lento)
            layer_idx: Capa donde se inyecta la activación

        Returns:
            Dict con:
              - pred_a: predicción top-1 del video A (original)
              - pred_b_original: predicción top-1 de B (sin patching)
              - pred_b_patched: predicción top-1 de B (con patching de A en capa L)
              - logit_shift: cambio en logit de la clase de A
              - flipped: bool, si pred_b_patched == pred_a
        """
        # 1. Predicción original de A
        with torch.no_grad():
            logits_a = self.model(video_a)
        pred_a = logits_a.argmax(dim=1).item()

        # 2. Predicción original de B (sin patching)
        with torch.no_grad():
            logits_b_original = self.model(video_b)
        pred_b_original = logits_b_original.argmax(dim=1).item()

        # 3. Extraer activación de A en capa layer_idx
        # source_activation: [1, N, 1408]
        source_activation = self._get_source_activation(video_a, layer_idx)

        # 4. Forward pass de B con activación de A inyectada
        logits_b_patched = self._patch_and_forward(video_b, source_activation, layer_idx)
        pred_b_patched = logits_b_patched.argmax(dim=1).item()

        # 5. Calcular logit shift: cambio en el logit de la clase de A
        logit_a_class_original = logits_b_original[0, pred_a].item()
        logit_a_class_patched = logits_b_patched[0, pred_a].item()
        logit_shift = logit_a_class_patched - logit_a_class_original

        return {
            "pred_a": pred_a,
            "pred_b_original": pred_b_original,
            "pred_b_patched": pred_b_patched,
            "logit_shift": logit_shift,
            "flipped": pred_b_patched == pred_a,
        }


def main():
    parser = argparse.ArgumentParser(description="Activation Patching causal")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/projects/video-prediction/checkpoints/videomaev2/vit_g_hybrid_pt_1200e_ssv2_ft.pth")
    parser.add_argument("--csv", type=str, default="physical_diagnostics.csv")
    parser.add_argument("--data_root", type=str, default="dataset/ssv2_luis")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_pairs", type=int, default=50,
                        help="Número de pares A/B a testear por capa")
    parser.add_argument("--output_dir", type=str, default="output_dir")
    parser.add_argument("--wandb_project", type=str, default="videomae_probing")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Cargar modelo
    model = load_model(args.checkpoint, device)

    # 2. Crear patcher
    patcher = ActivationPatcher(model, device)

    # 3. Cargar CSV y separar rápidos/lentos
    entries = load_diagnostic_csv(args.csv)
    fast_entries = [e for e in entries if e["speed_label"] == 1]
    slow_entries = [e for e in entries if e["speed_label"] == 0]
    print(f"\nVideos rápidos: {len(fast_entries)}, Videos lentos: {len(slow_entries)}")

    # 4. Seleccionar pares aleatorios
    num_pairs = min(args.num_pairs, len(fast_entries), len(slow_entries))
    fast_sample = random.sample(fast_entries, num_pairs)
    slow_sample = random.sample(slow_entries, num_pairs)
    pairs = list(zip(fast_sample, slow_sample))
    print(f"Pares seleccionados: {num_pairs}")

    # 5. Ejecutar patching por capa
    all_results = {}

    for layer_idx in args.layers:
        print(f"\n{'='*60}")
        print(f"PATCHING EN CAPA {layer_idx}")
        print(f"{'='*60}")

        layer_results = []
        flip_count = 0
        total_logit_shift = 0.0
        errors = 0

        for pair_idx, (fast_entry, slow_entry) in enumerate(pairs):
            if (pair_idx + 1) % 10 == 0:
                print(f"  Par {pair_idx+1}/{num_pairs}...")

            try:
                # Cargar videos
                video_a_path = os.path.join(args.data_root, fast_entry["video_path"])
                video_b_path = os.path.join(args.data_root, slow_entry["video_path"])

                video_a = load_video_frames(video_a_path, args.num_frames)
                video_a = video_a.unsqueeze(0).to(device)
                # video_a: [1, 3, 16, 224, 224]

                video_b = load_video_frames(video_b_path, args.num_frames)
                video_b = video_b.unsqueeze(0).to(device)
                # video_b: [1, 3, 16, 224, 224]

                # Patchear
                result = patcher.run_patching(video_a, video_b, layer_idx)
                layer_results.append(result)

                if result["flipped"]:
                    flip_count += 1
                total_logit_shift += result["logit_shift"]

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  [ERROR] Par {pair_idx+1}: {e}")
                continue

        successful = len(layer_results)
        flip_rate = flip_count / max(successful, 1)
        avg_logit_shift = total_logit_shift / max(successful, 1)

        all_results[f"layer_{layer_idx}"] = {
            "flip_rate": float(flip_rate),
            "avg_logit_shift": float(avg_logit_shift),
            "flip_count": flip_count,
            "total_pairs": successful,
            "errors": errors,
        }

        print(f"  Flip Rate: {flip_rate:.4f} ({flip_count}/{successful})")
        print(f"  Avg Logit Shift: {avg_logit_shift:.4f}")
        print(f"  Errores: {errors}")

    # 6. Guardar resultados
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "patching_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Resultados guardados en: {output_path}")

    # 7. WandB
    if not args.no_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name="causal_patching")

        for layer_name, data in sorted(all_results.items()):
            layer_idx = int(layer_name.split("_")[1])
            wandb.log({
                "patching_flip_rate": data["flip_rate"],
                "patching_logit_shift": data["avg_logit_shift"],
                "layer": layer_idx,
            })

        # Tabla para gráfica
        table_data = []
        for layer_name, data in sorted(all_results.items()):
            layer_idx = int(layer_name.split("_")[1])
            table_data.append([layer_idx, data["flip_rate"], data["avg_logit_shift"]])

        table = wandb.Table(
            data=table_data,
            columns=["layer", "flip_rate", "avg_logit_shift"]
        )
        wandb.log({
            "patching_flip_rate_by_layer": wandb.plot.line(
                table, "layer", "flip_rate",
                title="Causal Patching Flip Rate by Layer"
            ),
            "patching_logit_shift_by_layer": wandb.plot.line(
                table, "layer", "avg_logit_shift",
                title="Causal Patching Logit Shift by Layer"
            ),
        })
        wandb.finish()
        print("✅ Resultados logueados a WandB")

    # 8. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE ACTIVATION PATCHING")
    print("=" * 60)
    for layer_name, data in sorted(all_results.items()):
        layer = layer_name.split("_")[1]
        print(f"  Capa {layer:>2s}: Flip Rate={data['flip_rate']:.4f}, "
              f"Logit Shift={data['avg_logit_shift']:.4f} "
              f"({data['flip_count']}/{data['total_pairs']} pares)")


if __name__ == "__main__":
    main()
