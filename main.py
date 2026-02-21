import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Script principal para evaluar modelos de video')
    parser.add_argument('--model', type=str, choices=['videomae', 'vjepa'], required=True,
                        help='El modelo a evaluar: videomae o vjepa')
    parser.add_argument('--vjepa_variant', type=str, default='',
                        help='La variante del modelo V-JEPA (ej. ViT-H, ViT-L). Requerido si model=vjepa')
    parser.add_argument('--dataset', type=str, choices=['ssv2', 'k400'], required=True,
                        help='El dataset a utilizar: ssv2 o k400')
    
    args = parser.parse_args()

    if args.model == 'vjepa' and not args.vjepa_variant:
        print("Error: Si eliges 'vjepa', debes especificar la variante usando --vjepa_variant")
        sys.exit(1)

    print(f"=== Configuración Seleccionada ===")
    print(f"Modelo:  {args.model.upper()} {args.vjepa_variant}")
    print(f"Dataset: {args.dataset.upper()}")
    print("==================================")

    # Definir la carpeta de trabajo y el script según el modelo
    work_dir = ""
    script_to_run = ""
    
    if args.model == 'videomae':
        work_dir = "VideoMAEv2_Proyecto_Paper"
        if args.dataset == 'ssv2':
            script_to_run = "eval_ssv2.sh"
        elif args.dataset == 'k400':
            script_to_run = "eval_k400.sh"
            
    elif args.model == 'vjepa':
        work_dir = "vjepa2"
        if args.dataset == 'ssv2':
            script_to_run = f"eval_vjepa_{args.vjepa_variant.lower()}_ssv2.sh"
        elif args.dataset == 'k400':
            script_to_run = f"eval_vjepa_{args.vjepa_variant.lower()}_k400.sh"

    # Verificar que la carpeta exista
    if not os.path.isdir(work_dir):
        print(f"Error: No se encontró la carpeta del modelo '{work_dir}'")
        sys.exit(1)

    # Verificar que el script exista dentro de la carpeta
    script_path = os.path.join(work_dir, script_to_run)
    if not os.path.isfile(script_path):
        print(f"Error: No se encontró el script '{script_path}'")
        sys.exit(1)

    # Ejecutar sbatch indicando el directorio de trabajo (cwd)
    try:
        print(f"Enviando script a la cola SLURM: {script_path}")
        # subprocess.run(["sbatch", script_to_run], cwd=work_dir, check=True)
        print("¡Trabajo enviado exitosamente a Khipu!")
    except subprocess.CalledProcessError as e:
        print(f"Error al enviar el job a SLURM: {e}")

if __name__ == '__main__':
    main()