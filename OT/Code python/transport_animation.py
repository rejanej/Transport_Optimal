"""
Réjane Joyard - Projet Transport Optimal
Transport animation : animation image/histogramme.
"""
import matplotlib
# Importation des bibliothèques nécessaires et du fichier .py qui contient les fonctions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.restoration import denoise_bilateral
import cv2
from tqdm import tqdm
import transport_image

def update_plot(iteration):
    """
    Fonction permettant d'afficher l'évolution de l'image et son histogramme en fonction
    du transport réalisé
    :param iteration: nombre d'itérations sur lequel on déroule l'animation
    """
    plt.clf()

    if iteration == 0:
        original_image = image_input
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Image Target')
    else:
        current_iteration = iteration - 1
        current_image = images[current_iteration]
        plt.subplot(1, 2, 1)
        plt.imshow(current_image)
        plt.title('Image Target')
    plt.axis('off')

    # Afficher l'histogramme
    plt.subplot(1, 2, 2)
    for color in ['r', 'g', 'b']:
        plt.plot(histograms[iteration][color], color=color, linewidth=0.5, alpha=1)
    plt.title('Pixels distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    plt.ylim(0, 30000)

if __name__ == '__main__':
    # Initialisation des données
    image_input = cv2.imread('pexelA-0.png')
    image_target = cv2.imread('pexelB-0.png')
    image_source_copy = image_input.copy()
    histograms = []
    images = [image_input]
    alpha = 0.5
    energies = [transport_image.calculate_energy(image_input, image_target, transport_image.vector_random(3))]

    # Boucle principale
    for iteration in tqdm(range(20)):
        image_result = transport_image.transport(image_source_copy, image_target, transport_image.vector_random(3))
        images.append(image_result)
        iteration_histogram = {'b': [], 'g': [], 'r': []}
        transport_energy = transport_image.calculate_energy(image_source_copy,
                                                            image_result,
                                                            transport_image.vector_random(3))
        energies.append(transport_energy)

        for channel, color in enumerate(['b', 'g', 'r']):
            hist_channel = cv2.calcHist([image_result[:, :, channel].astype(np.uint8)],
                                        [0],
                                        None,
                                        [256],
                                        [0, 256])
            iteration_histogram[color] = hist_channel.tolist()

        histograms.append(iteration_histogram)
        image_source_copy = image_result

    final_image = transport_image.regularization(image_input, image_source_copy, 15)
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(16, 6))
    animation = FuncAnimation(fig, update_plot, frames=len(images), repeat=True)
    plt.show()