import re
import numpy as np
import os
import urllib.request

# --- CONFIGURACIÓN ---
result_path = 'work_dir/eval_test/0.txt'
NUM_CLASES = 710
LABEL_URL = "https://raw.githubusercontent.com/OpenGVLab/VideoMAEv2/master/misc/label_map_k710.txt"
LABEL_FILE = "label_map_k710.txt"

def get_labels():
    """Descarga o lee el archivo de etiquetas oficial de Kinetics-710"""
    labels = {}
    
    # 1. Si no lo tenemos, intentamos descargarlo
    if not os.path.exists(LABEL_FILE):
        print(f"⬇️  Descargando lista de etiquetas oficial de: {LABEL_URL}...")
        try:
            urllib.request.urlretrieve(LABEL_URL, LABEL_FILE)
            print("✅ Descarga completada.")
        except Exception as e:
            print(f"⚠️  No se pudo descargar la lista de etiquetas: {e}")
            print("   (Se mostrará solo el ID numérico)")
            return None

    # 2. Leemos el archivo (formato: una clase por línea, o índice:clase)
    try:
        with open(LABEL_FILE, 'r') as f:
            lines = f.readlines()
            # El archivo oficial suele ser una lista plana, línea 0 = clase 0
            for idx, line in enumerate(lines):
                labels[idx] = line.strip()
        return labels
    except Exception as e:
        print(f"⚠️  Error leyendo el archivo de etiquetas: {e}")
        return None

# --- PROGRAMA PRINCIPAL ---
print(f"Leyendo archivo de resultados: {result_path}...")

if not os.path.exists(result_path):
    print("❌ Error: El archivo de resultados no existe.")
    exit()

try:
    # 1. Obtener etiquetas
    label_map = get_labels()

    # 2. Leer archivo de predicciones
    with open(result_path, 'r') as f:
        content = f.read()

    # 3. Extraer números con Regex
    start = content.find('[')
    end = content.rfind(']') 

    if start != -1 and end != -1:
        data_block = content[start:end+1]
        pattern = r'-?\d+\.\d+(?:[eE][-+]?\d+)?'
        numbers = re.findall(pattern, data_block)
        scores = np.array([float(x) for x in numbers])

        if len(scores) == 0:
            print("❌ Error: No se detectaron números.")
            exit()

        # 4. Cálculos
        raw_max_index = np.argmax(scores)
        max_score = scores[raw_max_index]
        real_class_id = raw_max_index % NUM_CLASES
        view_index = raw_max_index // NUM_CLASES

        # 5. Obtener el nombre
        action_name = "Desconocido"
        if label_map and real_class_id in label_map:
            action_name = label_map[real_class_id]

        # 6. Mostrar Resultados
        print("\n" + "=" * 50)
        print(f"   🎬  RESULTADO DEL VIDEO")
        print("=" * 50)
        print(f"Probabilidad (Logit): {max_score:.4f}")
        print("-" * 50)
        print(f"✅ ID CLASE:   {real_class_id}")
        print(f"🏷️  ACCIÓN:     {action_name.upper()}")
        print("-" * 50)
        print(f"👁️  Detalle: Detectado en la vista #{view_index} (de 90)")
        print("=" * 50 + "\n")
        
    else:
        print("❌ Error: Formato de archivo no reconocido (faltan corchetes).")

except Exception as e:
    print(f"\n❌ Error inesperado: {e}")