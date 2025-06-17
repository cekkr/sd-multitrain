# /advanced-sd2-lora-trainer/scripts/advanced/dynamic_cropping.py

import random
from PIL import Image

def perform_random_crop(image: Image.Image, target_resolution: int) -> Image.Image:
    """
    Performs a random crop on a PIL image.

    Args:
        image (Image.Image): The input image.
        target_resolution (int): The desired square resolution for the crop.

    Returns:
        Image.Image: The randomly cropped and resized image.
    """
    if image.width == target_resolution and image.height == target_resolution:
        return image

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Determine crop dimensions while maintaining aspect ratio
    if original_width < original_height:
        new_width = target_resolution
        new_height = int(target_resolution / aspect_ratio)
    else:
        new_height = target_resolution
        new_width = int(target_resolution * aspect_ratio)

    # Resize to the smallest dimension that fits the target resolution
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate random crop coordinates
    x_offset = random.randint(0, new_width - target_resolution)
    y_offset = random.randint(0, new_height - target_resolution)

    # Perform the crop
    cropped_image = image.crop((
        x_offset, 
        y_offset, 
        x_offset + target_resolution, 
        y_offset + target_resolution
    ))

    return cropped_image