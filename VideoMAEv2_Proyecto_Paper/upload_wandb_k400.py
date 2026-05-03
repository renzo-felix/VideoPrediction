import json
import wandb
import os

def main():
    json_path = "output_dir/probing_results_k400_40layers.json"
    
    if not os.path.exists(json_path):
        print(f"Error: No se encontró {json_path}")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    # Iniciar WandB
    # El nodo principal (donde se ejecuta esto) SÍ tiene internet
    wandb.init(project="videomae_probing_k400", name="k400_linear_probing")

    # Extraer las accuracies
    # El formato del JSON es:
    # {"block_0": {"attn": 0.58, "mlp": 0.59, "residual": 0.60}, ...}
    
    num_layers = 40
    print(f"Subiendo métricas de {num_layers} capas a WandB...")

    for i in range(num_layers):
        metrics = {"layer": i}
        has_data = False
        
        for feature in ["attn", "mlp", "residual"]:
            key = f"block_{i}_{feature}"
            if key in results:
                metrics[f"accuracy_{feature}"] = results[key]
                has_data = True
                
        if has_data:
            wandb.log(metrics)
            
    wandb.finish()
    print("¡Sincronización con WandB completada!")

if __name__ == "__main__":
    main()
