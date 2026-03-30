import json
import numpy as np
import argparse
import sys

try:
    import wandb
except ImportError:
    print("❌ Error: wandb no está instalado. Ejecuta 'pip install wandb' primero.")
    sys.exit(1)


def log_to_wandb(results: dict, project: str, run_name: str = None):
    """
    Loguea los resultados pre-calculados del Linear Probing a Weights & Biases.
    Recibe el diccionario extraído de "probing_results.json".
    """
    print(f"🚀 Iniciando sesión en Weights & Biases (Proyecto: {project})...")
    wandb.init(project=project, name=run_name or "layer_probing_full_run")

    # 1. Loguear accuracy por capa/componente de forma directa
    print("Subiendo métricas de Accuracy individuales...")
    for name, data in sorted(results.items()):
        wandb.log({
            f"probe_accuracy/{name}": data["accuracy"],
        })

    # 2. Crear tabla detallada para la gráfica "Layer-wise Probe Accuracy"
    # Parsear nombres: "block_0_residual" → layer=0, component="residual"
    print("Construyendo tabla de Accuracy vs Capa...")
    table_data = []
    for name, data in sorted(results.items()):
        parts = name.split("_")
        if len(parts) == 3:
            # name = "block_0_residual"
            layer_idx = int(parts[1])
            component = parts[2]  # "residual", "attn", o "mlp"
            table_data.append([layer_idx, component, data["accuracy"]])
        else:
            print(f"Advertencia: Formato de nombre de capa inesperado: {name}")

    table = wandb.Table(
        data=table_data,
        columns=["layer", "component", "accuracy"]
    )

    # 3. Gráfica line plot (accuracy vs layer), con color por componente
    print("Generando Gráfica Especializada (Line Plot)...")
    wandb.log({
        "layer_wise_probe_accuracy": wandb.plot.line(
            table, "layer", "accuracy",
            stroke="component",
            title="Layer-wise Probe Accuracy (Speed Detection)"
        )
    })

    # 4. Loguear matrices de confusión personalizadas
    print("Subiendo Matrices de Confusión por componente...")
    for name, data in results.items():
        if "confusion_matrix" in data:
            cm = np.array(data["confusion_matrix"])
            # wandb.plot.confusion_matrix funciona nativo con listas/probabilidades, 
            # pero al tener el CM directo (pre-computado), simulamos su ingreso
            # Alternativamente usar una métrica custom si wandb no soporta arrays crudos
            # Sin embargo, el script original asume class_names y pre-preds, así que esto 
            # subirá un Heatmap para mayor compatibilidad con CM precalculadas:
            fig, ax = plt_cm(cm, name)
            if fig is not None:
                wandb.log({f"confusion_matrix/{name}": wandb.Image(fig)})

    wandb.finish()
    print("✅ ¡Resultados sincronizados exitosamente a WandB!")


def plt_cm(cm, name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["slow", "fast"], 
                yticklabels=["slow", "fast"], ax=ax)
    ax.set_title(f"Confusion Matrix: {name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description="Subir resultados locales a W&B")
    parser.add_argument("--json_file", type=str, default="output_dir/probing_results.json",
                        help="Ruta al archivo probing_results.json")
    parser.add_argument("--wandb_project", type=str, default="videomae_probing",
                        help="Nombre del proyecto de wandb")
    args = parser.parse_args()

    # Cargar JSON local
    try:
        with open(args.json_file, "r") as f:
            results = json.load(f)
        print(f"✅ Archivo cargado: {args.json_file} ({len(results)} registros encontrados)")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{args.json_file}'. Asegúrate de correr run_layer_probing.py primero.")
        sys.exit(1)

    log_to_wandb(results, args.wandb_project)

if __name__ == "__main__":
    main()
