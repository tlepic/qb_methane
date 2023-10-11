import matplotlib.pyplot as plt
import numpy as np

def visualize_with_data(images, titles, data, ncols=4):
    nrows = len(images) // ncols + int(len(images) % ncols != 0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18))
    fig.suptitle('Model Predictions and Accumulated Accuracy', fontsize=20, y=1.05)

    for ax, img, title, datum in zip(axes.ravel(), images, titles, data):
        ax.imshow(img, cmap='gray', interpolation='bicubic')  # Using 'bicubic' interpolation for smoother images
        title_text = title.split(", ")
        # Split title and data for better clarity
        ax.set_title(f"{title_text[0]}\n{title_text[1]}", fontsize=12)
        # Displaying data at the bottom of the image
        ax.set_xlabel(datum, fontsize=12, color='blue', weight='bold')
        ax.axis('off')

    for ax in axes.ravel()[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust spacing between images for better clarity
    plt.show()
