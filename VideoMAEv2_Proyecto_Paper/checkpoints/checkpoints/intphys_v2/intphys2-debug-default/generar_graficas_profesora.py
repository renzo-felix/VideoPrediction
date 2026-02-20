import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 🛠️ RUTAS A EDITAR (LUIS, VERIFICA ESTO)
# ==========================================
# 1. Tu archivo de resultados .pth
PATH_RESULTADOS = "/home/renzo.felix/Luis/VideoMAEv2_Proyecto_Paper/checkpoints/checkpoints/intphys_v2/intphys2-debug-default/losses_10fs_4_6_8_10_12_14ctxt.pth"

# 2. Tu archivo CSV de metadatos
PATH_CSV_LABELS = "/home/renzo.felix/Luis/IntPhys2_Proyecto_Paper/dataset/debug/Debug/metadata.csv"
# ==========================================

def analizar_resultados():
    print("--- 📊 GENERANDO GRÁFICAS PARA EL FRAMEWORK DE INVESTIGACIÓN ---")
    
    # ---------------------------------------------------------
    # PASO 1: Cargar Resultados del Modelo (.pth)
    # ---------------------------------------------------------
    try:
        print(f"📂 Cargando resultados: {PATH_RESULTADOS}")
        
        # --- CORRECCIÓN: Quitamos weights_only=False para compatibilidad ---
        data = torch.load(PATH_RESULTADOS, map_location='cpu')
        
        nombres_pth = data['names']        # Lista de nombres
        losses = data['losses']            # Tensor [60, 6, 31]
        contextos = data['context_lengths'] # [4, 6, 8, 10, 12, 14]
        
        # Promediamos el loss (dim 2) para tener un valor por video y contexto
        losses_avg = losses.mean(dim=2).numpy() # Forma [60, 6]
        
        print(f"✅ Resultados cargados. Total videos evaluados: {len(nombres_pth)}")
        
    except Exception as e:
        print(f"❌ Error cargando .pth: {e}")
        return

    # ---------------------------------------------------------
    # PASO 2: Cargar Etiquetas Reales (.csv)
    # ---------------------------------------------------------
    try:
        print(f"📂 Cargando metadatos: {PATH_CSV_LABELS}")
        df_labels = pd.read_csv(PATH_CSV_LABELS)
        
        # Limpiar espacios en nombres de columnas
        df_labels.columns = df_labels.columns.str.strip()
        
        # --- CORRECCIÓN CLAVE: LIMPIEZA DE NOMBRES ---
        # El CSV tiene "Videos/hash.mp4", pero el PTH suele tener solo "hash.mp4"
        df_labels['clean_name'] = df_labels['file_name'].apply(lambda x: os.path.basename(x))
        
        print(f"✅ Metadatos cargados. Total filas en CSV: {len(df_labels)}")
        
    except Exception as e:
        print(f"❌ Error cargando CSV: {e}")
        return

    # ---------------------------------------------------------
    # PASO 3: Cruzar la Información (Merge)
    # ---------------------------------------------------------
    print("🔄 Uniendo resultados con etiquetas...")
    
    lista_consolidada = []
    encontrados = 0
    
    for i, video_name_pth in enumerate(nombres_pth):
        nombre_simple = os.path.basename(video_name_pth)
        
        # Buscamos este nombre en el CSV (usando la columna limpia)
        fila = df_labels[df_labels['clean_name'] == nombre_simple]
        
        if not fila.empty:
            encontrados += 1
            # Extraemos los datos físicos
            propiedad_fisica = fila['condition'].values[0] 
            tipo_evento = fila['type'].values[0] # Ej: 1_Possible
            
            es_posible = "Possible" if "Possible" in tipo_evento else "Impossible"
            
            # Guardamos los datos para cada contexto
            for j, ctx in enumerate(contextos):
                loss_valor = losses_avg[i, j]
                lista_consolidada.append({
                    'Video': nombre_simple,
                    'Propiedad_Fisica': propiedad_fisica, 
                    'Plausibilidad': es_posible,          
                    'Context_Frames': ctx,
                    'Reconstruction_Loss': loss_valor
                })
        else:
            if i == 0:
                print(f"⚠️ Aviso: El video '{nombre_simple}' no se encontró en el CSV.")
    
    print(f"✅ Cruce exitoso: {encontrados} de {len(nombres_pth)} videos tienen etiqueta.")
    
    if encontrados == 0:
        print("❌ DETENIDO: No se encontraron coincidencias. Revisa los nombres de archivo.")
        return

    df_final = pd.DataFrame(lista_consolidada)

    # ---------------------------------------------------------
    # PASO 4: Generar las Gráficas
    # ---------------------------------------------------------
    sns.set_style("whitegrid")
    
    # --- GRÁFICA 1: Loss por Propiedad Física ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_final, 
        x='Context_Frames', 
        y='Reconstruction_Loss', 
        hue='Propiedad_Fisica', 
        style='Propiedad_Fisica',
        markers=True, dashes=False, linewidth=2.5
    )
    plt.title('VideoMAE v2: Reconstruction Loss by Physical Concept', fontsize=14)
    plt.ylabel('Reconstruction Loss (Surprise)', fontsize=12)
    plt.xlabel('Context Frames', fontsize=12)
    plt.legend(title='Physical Label')
    plt.tight_layout()
    plt.savefig('grafica_1_por_propiedad.png', dpi=300)
    print("🎨 Gráfica 1 guardada: grafica_1_por_propiedad.png")

    # --- GRÁFICA 2: Posible vs Imposible ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_final,
        x='Propiedad_Fisica',
        y='Reconstruction_Loss',
        hue='Plausibilidad',
        palette={'Possible': 'green', 'Impossible': 'red'}
    )
    plt.title('Sanity Check: Surprise on Possible vs Impossible Events', fontsize=14)
    plt.ylabel('Reconstruction Loss', fontsize=12)
    plt.xlabel('Physical Concept', fontsize=12)
    plt.tight_layout()
    plt.savefig('grafica_2_posible_vs_imposible.png', dpi=300)
    print("🎨 Gráfica 2 guardada: grafica_2_posible_vs_imposible.png")

    print("\n✅ ¡LISTO LUIS!")

if __name__ == "__main__":
    analizar_resultados()