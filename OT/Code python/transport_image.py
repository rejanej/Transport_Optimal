"""
Réjane Joyard - Projet Transport Optimal
Transport image : affichage images et histogrammes associés (possibilité de changer les images à condition
qu'elles soient de même taille), et graphe de l'énergie.
"""

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.restoration import denoise_bilateral
import cv2
from tqdm import tqdm

def vector_random(d):
    """
    Fonction permettant d'obtenir un vecteur de dimension souhaitée
    :param d: dimension du vecteur que l'on souhaite
    :return: vecteur de norme 1
    """
    random_vector = np.random.normal(0, 1, d)
    normalization = random_vector / np.linalg.norm(random_vector)
    return normalization

def sort(image, theta):
    """
    Fonction permettant de trier la projection des échantillons
    d'une image sur une direction theta aléatoire
    :param image: image sur laquelle on souhaite réaliser les projections
    :param theta: direction aléatoire
    :return: un tableau trié par couleur en fonction de leur projection dans la direction theta
    """
    h, l, _ = image.shape
    color = image.reshape(-1, 3)
    projection = np.dot(color, theta)
    indices = np.arange(h * l).reshape(h, l)
    sorting = np.column_stack((projection, indices.ravel()))
    sorting = sorting[sorting[:, 0].argsort()]

    return sorting

def transport(image1, image2, theta, alpha):
    """
    Fonction réalisant le transport optimal sur l'image input
    :param image1: image input
    :param image2: image target
    :param theta: direction aléatoire
    :return: résultat du transport optimal des échantillons de l'image
    input vers l'image target dans la direction theta
    """
    sorted_source = sort(image1, theta)
    sorted_target = sort(image2, theta)
    advected_samples = np.zeros_like(image1, dtype=np.uint8)

    source_indices = sorted_source[:, 1].astype(int)
    source_pixels = np.unravel_index(source_indices, image1.shape[:2])

    valid_pixels = np.logical_and.reduce([(0 <= source_pixels[0]),
                                        (source_pixels[0] < image1.shape[0]),
                                        (0 <= source_pixels[1]),
                                        (source_pixels[1] < image1.shape[1]),])

    differences = sorted_source[:, 0] - sorted_target[:, 0]

    advected_samples[source_pixels[0][valid_pixels], source_pixels[1][valid_pixels]] = np.clip(
                image1[source_pixels[0][valid_pixels],
                source_pixels[1][valid_pixels]] -(alpha * differences[valid_pixels][:, np.newaxis] * theta),
                0,
                255).astype(np.uint8)

    return advected_samples

def regularization(image_source, transported_image, sigma):
    """
    Fonction permettant la régularisation (gestion du bruit), afin d'obtenir un résultat plus lisse
    :param image_source: image input
    :param transported_image: image output
    :param sigma: paramètre de régularisation
    :return: image output lissée
    """
    difference = transported_image - image_source
    filtered_difference = cv2.bilateralFilter(difference, d=15, sigmaColor=sigma, sigmaSpace=sigma)

    final_image = np.clip(image_source + filtered_difference, 0, 255).astype(np.uint8)
    return final_image

def calculate_energy(image1, image2, theta):
    """
    Fonction permettant le calcul de l'énergie lors de la réalisation du transport optimal
    :param image1: image input
    :param image2: image target
    :param theta: direction aléatoire dans laquelle on veut projeter
    :return: valeur de l'énergie associée au transport optimal des échantillons
    de l'image input vers l'image target dans la direction theta
    """
    sorted_source = sort(image1, theta)
    sorted_target = sort(image2, theta)

    source_indices = sorted_source[:, 1].astype(int)
    source_pixels = np.unravel_index(source_indices, image1.shape[:2])

    valid_pixels = np.logical_and.reduce([(0 <= source_pixels[0]),
                                          (source_pixels[0] < image1.shape[0]),
                                          (0 <= source_pixels[1]),
                                          (source_pixels[1] < image1.shape[1]), ])

    differences = sorted_source[:, 0] - sorted_target[:, 0]
    transport_energy = np.sum(np.abs(differences[valid_pixels])**2)

    return transport_energy

if __name__ == '__main__':
    # Initialisation des données
    image_input = cv2.imread('Matisse.png')
    image_target = cv2.imread('vas.png')
    image_source_copy = image_input.copy()
    alpha = 0.5
    energies = [calculate_energy(image_input, image_target, vector_random(3))]

    # Boucle principale
    for iteration in tqdm(range(200)):
        image_result = transport(image_source_copy, image_target, vector_random(3), alpha)
        transport_energy = calculate_energy(image_source_copy, image_result, vector_random(3))
        energies.append(transport_energy)
        image_source_copy = image_result

    final_image = regularization(image_input, image_result, 15)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # Histogrammes et résultats des images
    plt.figure(figsize=(9, 12))

    plt.subplot(3, 2, 1)
    plt.title('Input image')
    plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title('Pixels distribution of the image input')
    for j, color in enumerate(['b', 'g', 'r']):
        hist1 = cv2.calcHist([image_input], [j], None, [256], [0, 256])
        plt.plot(hist1, color=color, linewidth=0.5, alpha=1)
        plt.xlim([0, 256])
        plt.ylim(0, 30000)
        plt.ylabel('Number of Pixels')
        plt.xlabel('Pixel Value')

    plt.subplot(3, 2, 3)
    plt.title('Target image')
    plt.imshow(cv2.cvtColor(image_target, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title('Pixels distribution of the image target')
    for j, color in enumerate(['b', 'g', 'r']):
        hist2 = cv2.calcHist([image_target], [j], None, [256], [0, 256])
        plt.plot(hist2, color=color, linewidth=0.5, alpha=1)
        plt.xlim([0, 256])
        plt.ylim(0, 30000)
        plt.ylabel('Number of Pixels')
        plt.xlabel('Pixel Value')

    plt.subplot(3, 2, 5)
    plt.title('Output image')
    plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title('Pixels distribution of the image output')
    for channel, color in enumerate(['b', 'g', 'r']):
        hist_final = cv2.calcHist([final_image[:, :, channel].astype(np.uint8)], [0], None, [256], [0, 256])
        plt.plot(hist_final, color=color, label=f'Channel {color}', linewidth=0.5, alpha=1)
        plt.xlim([0, 256])
        plt.ylim(0, 30000)
        plt.ylabel('Number of Pixels')
        plt.xlabel('Pixel Value')

    # Ouvre une fenêtre pour afficher le graphe de l'énergie
    plt.figure()
    plt.plot(energies, linewidth=0.5, alpha=1)
    plt.xlabel('Itération')
    plt.ylabel('Energy')
    plt.xlim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()