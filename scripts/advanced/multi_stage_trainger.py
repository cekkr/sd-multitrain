# /advanced-sd2-lora-trainer/scripts/advanced/multi_stage_trainer.py

import os
import torch
from core.training_logic import run_training_loop # We can reuse the core training loop
from core.data_handler import LoRATrainingDataset

def divide_dataset(full_dataset: LoRATrainingDataset, keywords_a: str, keywords_b: str):
    """
    Divides a dataset into two subsets based on keywords in the captions.
    """
    keywords_a_list = [k.strip().lower() for k in keywords_a.split(',') if k.strip()]
    keywords_b_list = [k.strip().lower() for k in keywords_b.split(',') if k.strip()]
    
    indices_a = []
    indices_b = []

    for i, caption in enumerate(full_dataset.captions):
        caption_lower = caption.lower()
        if any(key in caption_lower for key in keywords_a_list):
            indices_a.append(i)
        elif any(key in caption_lower for key in keywords_b_list):
            indices_b.append(i)
            
    dataset_a = torch.utils.data.Subset(full_dataset, indices_a)
    dataset_b = torch.utils.data.Subset(full_dataset, indices_b)
    
    return dataset_a, dataset_b

def run_multi_stage_training(training_params: dict):
    """
    Generator function that orchestrates the entire multi-stage training process
    and yields status updates to the UI.
    
    Args:
        training_params (dict): A dictionary containing all parameters from the UI.
    """
    base_lora_name = f"{training_params['output_lora_name']}_base"
    
    # --- STAGE 1: Train Base LoRA ---
    yield "--- Initiating Multi-Stage Training ---"
    yield f"Stage 1: Training base LoRA '{base_lora_name}'..."
    
    # Train the base model for a fraction of the total epochs (e.g., 40%)
    stage1_epochs = max(1, int(training_params['num_epochs'] * 0.4))
    stage1_params = training_params.copy()
    stage1_params['num_epochs'] = stage1_epochs
    stage1_params['output_lora_name'] = base_lora_name

    # The run_training_loop is a generator, so we must iterate through its yields
    for status in run_training_loop(stage1_params, is_sub_process=True):
        yield f"[Stage 1] {status}"
        
    yield "Stage 1 finished. Base LoRA has been saved."

    # --- DATASET DIVISION ---
    yield "Analyzing and dividing dataset for Stage 2..."
    full_dataset = LoRATrainingDataset(
        training_params['image_dataset_directory'],
        training_params['tokenizer'], # The tokenizer needs to be passed in params
        training_params['enable_dynamic_cropping'],
        training_params['crop_resolution']
    )
    
    dataset_a, dataset_b = divide_dataset(
        full_dataset,
        training_params['concept_a_keywords'],
        training_params['concept_b_keywords']
    )
    
    if not dataset_a or not dataset_b:
        yield "Error: Could not divide the dataset. Ensure captions match the keywords provided. Aborting Stage 2."
        return

    yield f"Dataset divided: Concept A has {len(dataset_a)} images, Concept B has {len(dataset_b)} images."

    # --- STAGE 2: Train Specialized LoRAs ---
    stage2_epochs = training_params['num_epochs'] - stage1_epochs
    if stage2_epochs <= 0:
        yield "Warning: No epochs remaining for Stage 2. Increase total epochs to train specialized models."
        return

    # Train LoRA for Concept A
    lora_a_name = f"{training_params['output_lora_name']}_concept_A"
    yield f"Stage 2: Fine-tuning '{lora_a_name}'..."
    stage2a_params = training_params.copy()
    stage2a_params.update({
        'num_epochs': stage2_epochs,
        'output_lora_name': lora_a_name,
        'custom_dataset': dataset_a,
        'base_lora_to_load': base_lora_name # Signal to load the base LoRA before training
    })
    for status in run_training_loop(stage2a_params, is_sub_process=True):
        yield f"[Stage 2A] {status}"
    yield f"Concept A LoRA saved as {lora_a_name}.safetensors."

    # Train LoRA for Concept B
    lora_b_name = f"{training_params['output_lora_name']}_concept_B"
    yield f"Stage 2: Fine-tuning '{lora_b_name}'..."
    stage2b_params = training_params.copy()
    stage2b_params.update({
        'num_epochs': stage2_epochs,
        'output_lora_name': lora_b_name,
        'custom_dataset': dataset_b,
        'base_lora_to_load': base_lora_name # Load the original base LoRA again
    })
    for status in run_training_loop(stage2b_params, is_sub_process=True):
        yield f"[Stage 2B] {status}"
    yield f"Concept B LoRA saved as {lora_b_name}.safetensors."

    yield "--- Multi-Stage Training Finished ---"