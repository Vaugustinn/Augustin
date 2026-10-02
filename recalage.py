import math
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

# ==========================================
# FONCTIONS DE FILTRAGE (Question 2)
# ==========================================

def creer_noyau_gaussien(size=21, sigma=5):
    """Génère un noyau Gaussien 2D avec meshgrid."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    G = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return G / G.sum()

def apply_filters(image_array):
    """Applique le flou Gaussien et les filtres de Sobel sur une image numpy."""
    # Filtre Gaussien
    G = creer_noyau_gaussien(size=21, sigma=5)
    img_gauss = convolve2d(image_array, G, mode='same')
    
    # Filtres de Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]])
    
    img_sob_x = convolve2d(image_array, sobel_x, mode='same')
    img_sob_y = convolve2d(image_array, sobel_y, mode='same')
    
    return img_gauss, img_sob_x, img_sob_y

# ==========================================
# TRAITEMENT PAR LOT (Registration + Filtrage)
# ==========================================

def process_full_batch(dossier_source, max_images=10, target_size=(200, 100)):
    """
    Parcourt le dossier, gère les clics interactifs pour le recalage,
    applique les filtres, et affiche une grille récapitulative.
    """
    extensions = ('*.png', '*.jpg', '*.jpeg')
    fichiers = []
    for ext in extensions:
        fichiers.extend(glob.glob(os.path.join(dossier_source, ext)))
    fichiers = fichiers[:max_images]

    if not fichiers:
        print(f"Aucune image trouvée dans le dossier '{dossier_source}'.")
        return

    resultats = []

    # 1. Boucle sur toutes les images
    for index, fichier in enumerate(fichiers):
        try:
            img = Image.open(fichier).convert('L')
            
            # Saisie interactive des coordonnées
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Image {index + 1}/{len(fichiers)} : Cliquez sur l'aile GAUCHE puis DROITE")
            ax.axis('off')
            
            coords = plt.ginput(n=2, timeout=0)
            plt.close(fig)

            if len(coords) == 2:
                # Trigonométrie pour l'angle
                dx = coords[1][0] - coords[0][0]
                dy = coords[1][1] - coords[0][1]
                angle = math.degrees(math.atan2(dy, dx))
                
                # Recalage et redimensionnement
                img_aligned = img.rotate(angle, resample=Image.BICUBIC, expand=True)
                img_resized = img_aligned.resize(target_size, resample=Image.BICUBIC)
                
                # Filtrage
                img_np = np.array(img_resized, dtype=float)
                gauss, sob_x, sob_y = apply_filters(img_np)
                
                # Sauvegarde en mémoire pour l'affichage final
                resultats.append({
                    'nom': os.path.basename(fichier),
                    'recalee': img_np,
                    'gauss': gauss,
                    'sob_x': sob_x,
                    'sob_y': sob_y
                })
            else:
                print(f"Ignoré : clics insuffisants pour {fichier}")
                
        except Exception as e:
            print(f"Erreur sur {fichier} : {e}")

    # 2. Affichage final sous forme de grille (4 colonnes, X lignes)
    nb_images = len(resultats)
    if nb_images > 0:
        fig_finale, axes = plt.subplots(nb_images, 4, figsize=(16, 3 * nb_images))
        fig_finale.suptitle("Résultats : Registration & Filtering", fontsize=16)
        
        # Sécurité au cas où il n'y aurait qu'une seule image (axes devient 1D)
        if nb_images == 1:
            axes = [axes]
            
        for i, res in enumerate(resultats):
            # Colonne 1 : Originale Recalée
            axes[i][0].imshow(res['recalee'], cmap='gray')
            axes[i][0].set_title(f"{res['nom']} (200x100)")
            axes[i][0].axis('off')

            # Colonne 2 : Gaussien
            axes[i][1].imshow(res['gauss'], cmap='gray')
            axes[i][1].set_title('Filtre Gaussien')
            axes[i][1].axis('off')

            # Colonne 3 : Sobel X
            axes[i][2].imshow(res['sob_x'], cmap='gray')
            axes[i][2].set_title('Sobel X (Bords Verticaux)')
            axes[i][2].axis('off')

            # Colonne 4 : Sobel Y
            axes[i][3].imshow(res['sob_y'], cmap='gray')
            axes[i][3].set_title('Sobel Y (Bords Horizontaux)')
            axes[i][3].axis('off')

        plt.tight_layout()
        plt.show()

# ==========================================
# EXÉCUTION
# ==========================================
if __name__ == "__main__":
    dossier_test = "face_dataset"
    
    if not os.path.exists(dossier_test):
        os.makedirs(dossier_test)
        print(f"Dossier '{dossier_test}' créé. Ajoutez des images et relancez.")
    else:
        process_full_batch(dossier_test, max_images=10)
