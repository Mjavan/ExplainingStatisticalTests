import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from PIL import Image


class ZeroGradientError(Exception):
    """Custom exception to handle cases where the gradient is zero."""
    pass


def save_cam_with_alpha(image, gcam, alpha=0.5):
    # Standardization for ImageNet: mean and std for RGB channels
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean for RGB channels
    std = np.array([0.229, 0.224, 0.225])   # ImageNet std for RGB channels

    # Denormalize the image (image is assumed to be in (C, H, W) format)
    image = image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    image = image * std + mean  # Reverse normalization

    # Clip to ensure values are within [0, 1] range before scaling to [0, 255]
    image = np.clip(image, 0, 1) * 255.0
    image = image.astype(np.uint8)

    # Normalize the Grad-CAM values to [0, 1]
    # Normalize the Grad-CAM values to [0, 1], handling zero gradients
    gcam_min = np.min(gcam)
    gcam_max = np.max(gcam)

    try:
        if gcam_max == gcam_min:  # If all values are zero, raise an error
            raise ZeroGradientError(
                "Gradient map contains only zero values, cannot overlay.")
        # Normalize gradient map
        gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
    except ZeroGradientError as e:
        print(f"Error: {e}")
        # Handle the error (for example, return the original image or skip processing)
        return image, image  # Return the original image if error occurs

    # Resize Grad-CAM to match the image dimensions (224x224)
    # Get height and width (height, width) from image shape
    h, w = image.shape[:2]
    gcam_resized = np.array(Image.fromarray(
        gcam).resize((w, h), Image.BILINEAR))

    # Apply a colormap (similar to cv2.applyColorMap)
    # Apply colormap and select RGB channels
    gcam_colored = plt.cm.jet(gcam_resized)[:, :, :3] * 255
    gcam_colored = gcam_colored.astype(np.uint8)

    # Add Grad-CAM on top of the original image using alpha blending
    heatmap = gcam_colored.astype(np.float64)

    overlaid_image = (alpha * heatmap + (1 - alpha) *
                      image.astype(np.float64)).astype(np.uint8)

    return image, overlaid_image


def overlay_heatmap_single(image, heatmap, img_idx, overlay_dir=None, alpha=0.5):
    # Extract the original image and create the overlay
    rescale_image, overlaid_img = save_cam_with_alpha(
        image, heatmap, alpha=alpha)

    # Create a 1x3 grid of subplots with consistent aspect ratios
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={
                            'width_ratios': [1, 1, 1.2]})

    # Original image
    axs[0].imshow(rescale_image, aspect='auto')
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    # Heatmap
    axs[1].imshow(heatmap, cmap='jet', aspect='auto')
    axs[1].axis('off')
    axs[1].set_title("Grad-CAM Heatmap")

    # Grad-CAM heatmap overlaid on the image
    im = axs[2].imshow(overlaid_img, aspect='auto')
    axs[2].axis('off')
    axs[2].set_title("Overlayed Image")

    # Add colorbar for the Grad-CAM heatmap with proper alignment
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                        ax=axs[2], fraction=0.046, pad=0.04)
    cbar.set_label('Grad-CAM Intensity')
    plt.tight_layout()

    file_path = os.path.join(overlay_dir, f'overlay_{img_idx}.png')
    fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Plutcurve


def plotCurves(stats, results_dir=None):
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(stats['train'], label='train_loss')
    plt.plot(stats['val'], label='valid_loss')
    textsize = 12
    marker = 5
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('NLL')
    lgd = plt.legend(['train', 'validation'], markerscale=marker,
                     prop={'size': textsize, 'weight': 'normal'})
    plt.savefig(results_dir, bbox_extra_artists=(lgd,), bbox_inches='tight')
