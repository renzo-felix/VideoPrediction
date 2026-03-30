import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import gc

try:
    import umap
except ImportError:
    print("Por favor, instala umap-learn primero: pip install umap-learn")
    import sys
    sys.exit(1)


class ClusteringEvolutionAnimator:
    """
    Carga activaciones desde un archivo NPZ (generado por el pipeline de probing)
    y genera una animación MP4 que muestra la evolución de los clústeres capa por capa.
    Soporta visualización simultánea de 3 paneles: Residual, Attention y MLP.
    """

    def __init__(self, npz_path: str, output_dir: str):
        self.npz_path = npz_path
        self.output_dir = output_dir
        self.layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]
        self.components = ["residual", "attn", "mlp"]
        
        # Almacenarán las proyecciones 2D (diccionarios component -> int(layer) -> np.ndarray [N, 2])
        self.projections = {comp: {} for comp in self.components}
        self.labels = None

    def load_data(self):
        """Carga el .npz y aisla las etiquetas."""
        print(f"[INFO] Cargando datos desde {self.npz_path}...")
        self.data = np.load(self.npz_path)
        self.labels = self.data["labels"]
        print(f"       ✅ Se cargaron {len(self.labels)} muestras/videos.")

    def apply_dimensionality_reduction(self):
        """
        Aplica PCA (hasta 50 dims) y luego UMAP (hasta 2 dims).
        CRÍTICO PARA ANIMACIÓN: Para que las nubes de puntos no salten caóticamente 
        entre cada fotograma/capa, construimos un ÚNICO espacio global (X_all) sumando
        los tensores de las 11 capas. UMAP aprenderá la estructura sobre todas las capaz a la vez,
        lo que garantiza una coherencia espacio-temporal perfecta para la animación.
        """
        num_videos = len(self.labels)

        for comp in self.components:
            print(f"\n[UMAP] Procesando componente: {comp.upper()}")
            
            # 1. Acumular todas las capas para construir X_all
            X_list = []
            for layer in self.layers:
                key = f"block_{layer}_{comp}_features"
                if key in self.data:
                    X_list.append(self.data[key])
                else:
                    raise KeyError(f"No se halló la característica {key} en el npz.")
            
            # X_all: [N_videos * N_capas, 1408]
            X_all = np.vstack(X_list)
            print(f"       Matriz global combinada: {X_all.shape}")

            # 2. PCA para reducir ruido y acelerar UMAP (1408 -> 50)
            print("       -> Aplicando PCA rápido (50 componentes)...")
            pca = PCA(n_components=50, random_state=42)
            X_pca = pca.fit_transform(X_all)

            # 3. UMAP para la topología global (50 -> 2)
            # init="pca" y setear un random_state previene rotaciones arbitrarias
            print("       -> Aplicando UMAP (2 componentes). Esto tomará algunos minutos...")
            reducer = umap.UMAP(n_components=2, init="pca", random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap_all = reducer.fit_transform(X_pca)

            # 4. Dividir de nuevo en chunks por capa
            for idx, layer in enumerate(self.layers):
                start = idx * num_videos
                end = start + num_videos
                self.projections[comp][layer] = X_umap_all[start:end, :]

            # Liberar RAM intensiva
            del X_all, X_list, X_pca, X_umap_all, reducer, pca
            gc.collect()

    def generate_animation(self):
        """Genera el MP4 renderizando cada capa como un fotograma de la animación."""
        os.makedirs(self.output_dir, exist_ok=True)
        video_path = os.path.join(self.output_dir, "clustering_evolution.mp4")
        
        print("\n[RENDER] Preparando figura para renderizar la animación MP4...")
        
        # Paleta de colores para las etiquetas (Fast=1, Slow=0 en SSv2)
        # Soportamos Intphys si labels es continuo usando scatter con 'c=self.labels'
        is_continuous = self.labels.dtype.kind in 'fc' and len(np.unique(self.labels)) > 10
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        
        def update(frame_idx):
            layer = self.layers[frame_idx]
            
            for i, comp in enumerate(self.components):
                ax = axes[i]
                ax.clear()
                
                # Coordenadas 2D de la capa actual
                X_2d = self.projections[comp][layer]
                x_coords = X_2d[:, 0]
                y_coords = X_2d[:, 1]
                
                if is_continuous:
                    sc = ax.scatter(x_coords, y_coords, c=self.labels, cmap='viridis', 
                                    s=10, alpha=0.6, edgecolors='none')
                else:
                    # Clasificación Binaria Rápido vs Lento
                    scatter_kws = {'s': 10, 'alpha': 0.6, 'edgecolors': 'none'}
                    sns.scatterplot(x=x_coords, y=y_coords, hue=self.labels, 
                                    palette=["#3498db", "#e74c3c"], ax=ax, **scatter_kws, legend=False)
                
                ax.set_title(f"{comp.upper()}", fontsize=14, fontweight='bold')
                ax.axis('off') # Remover ejes para estética más limpia
                
            fig.suptitle(f"VideoMAEv2 Clustering Evolution - Capa {layer}", fontsize=20, fontweight='bold')
            return axes
        
        # Animación con FuncAnimation
        anim = FuncAnimation(fig, update, frames=len(self.layers), interval=800, blit=False)
        
        # Configurar writer para archivo .mp4 (30 fps / 0.8 seg por frame no es standard en fps, 
        # pero matplotlib respeta el interval al reproducirlo)
        print(f"       -> Guardando render final en {video_path}...")
        anim.save(video_path, writer='ffmpeg', fps=2, dpi=120)
        plt.close(fig)
        
        print("✅ ¡Animación generada con éxito!")


def main():
    parser = argparse.ArgumentParser(description="Animación de Clustering de capas")
    parser.add_argument("--npz", type=str, default="output_dir/activations.npz", 
                        help="Ruta a las matrices de activaciones extraídas.")
    parser.add_argument("--output_dir", type=str, default="videos_simulation_clustering",
                        help="Dónde guardar el MP4 resultante.")
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        print(f"❌ ERROR: El archivo {args.npz} no existe. Por favor ejecuta run_layer_probing.py primero.")
        sys.exit(1)

    animator = ClusteringEvolutionAnimator(args.npz, args.output_dir)
    animator.load_data()
    animator.apply_dimensionality_reduction()
    animator.generate_animation()

if __name__ == "__main__":
    main()
