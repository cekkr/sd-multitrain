# /advanced-sd2-lora-trainer/scripts/core/project_utils.py

import os
import re

# Get the root directory of the Stable Diffusion WebUI
# This is a simplified approach; a more robust method might involve parsing command-line args
SD_WEBUI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

def get_model_path(model_name: str) -> str:
    """
    Constructs the full path to a model file in the 'models/Stable-diffusion' directory.
    
    Args:
        model_name (str): The filename of the model (e.g., 'v2-1_768-v-ema.ckpt').

    Returns:
        str: The absolute path to the model file.
    """
    return os.path.join(SD_WEBUI_ROOT, 'models', 'Stable-diffusion', model_name)

def get_lora_output_path() -> str:
    """
    Returns the absolute path to the LoRA output directory.
    """
    return os.path.join(SD_WEBUI_ROOT, 'models', 'Lora')

def sanitize_filename(name: str) -> str:
    """
    Removes characters that are invalid for filenames.
    
    Args:
        name (str): The desired filename.

    Returns:
        str: A sanitized version of the filename.
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    return sanitized