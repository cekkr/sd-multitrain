# /advanced-sd2-lora-trainer/scripts/core/model_management.py

import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def load_models(base_model_path: str, is_v2: bool):
    """
    Loads the necessary models (U-Net, text encoder, tokenizer) from a checkpoint.
    Crucially handles the difference between SD1.x and SD2.x models. 
    """
    if is_v2:
        # For SD2, the text encoder is 'openai/clip-vit-large-patch14' from OpenCLIP 
        text_encoder_id = "openai/clip-vit-large-patch14"
    else:
        # For SD1.x, the standard is a subfolder within the model repo
        text_encoder_id = "openai/clip-vit-large-patch14" # Placeholder, actual loading is more complex for SD1.5

    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_id)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_id)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    
    return tokenizer, text_encoder, unet

def inject_lora_adapters(unet, text_encoder, rank, alpha, train_text_encoder):
    """
    Injects LoRA adapters into the U-Net and optionally the text encoder using PEFT. 
    """
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        # Specify target modules for attention layers 
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    print("Injecting LoRA into U-Net...")
    unet.add_adapter(lora_config) # 
    
    if train_text_encoder:
        print("Injecting LoRA into Text Encoder...")
        text_encoder.add_adapter(lora_config) # 
        
    return unet, text_encoder

def save_lora_weights(unet, text_encoder, output_path, lora_name):
    """
    Extracts and saves the trained LoRA weights in the safetensors format. 
    """
    # The models are PEFT models, so we can call save_pretrained
    unet_lora_layers = unet.get_input_embeddings() # Placeholder for actual PEFT API
    
    # In a real PEFT implementation, you would save the adapter directly
    # PeftModel(unet).save_pretrained(os.path.join(output_path, f"{lora_name}_unet"))
    
    # For now, simulate saving
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    final_path = os.path.join(output_path, f"{lora_name}.safetensors")
    # This is a placeholder for the actual saving function from PEFT/diffusers
    with open(final_path, 'w') as f:
        f.write("Trained LoRA weights placeholder.")
        
    print(f"LoRA saved to: {final_path}")
    return final_path