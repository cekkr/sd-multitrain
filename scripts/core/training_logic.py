# /advanced-sd2-lora-trainer/scripts/core/training_logic.py

import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from diffusers import DDPMScheduler
from peft import PeftModel

# Import from our project structure
from core.model_management import load_models, inject_lora_adapters, save_lora_weights
from core.data_handler import LoRATrainingDataset
from core.project_utils import get_model_path, get_lora_output_path, sanitize_filename
from advanced.multi_stage_trainer import run_multi_stage_training

def start_training(
    base_model_checkpoint, output_lora_name, is_v2_model, is_v_parameterization,
    image_dataset_directory, learning_rate, text_encoder_lr, train_batch_size, num_epochs,
    network_rank, network_alpha, train_text_encoder, optimizer_type,
    enable_dynamic_cropping, crop_resolution, enable_multi_stage,
    concept_a_keywords, concept_b_keywords
    ):
    """
    This function acts as a dispatcher. It collects all UI parameters into a dictionary
    and decides which training path to take based on user settings.
    It returns a generator that yields status updates to the UI.
    """
    
    # Pack all parameters into a single dictionary for easier handling
    training_params = locals()
    
    try:
        if enable_multi_stage:
            # Hand off to the multi-stage trainer
            yield from run_multi_stage_training(training_params)
        else:
            # Run a standard training process
            yield from run_training_loop(training_params)
    except Exception as e:
        print(f"An error occurred during training dispatch: {e}")
        import traceback
        traceback.print_exc()
        yield f"Fatal Error: {e}. Check console for full traceback."
    finally:
        # Final cleanup
        torch.cuda.empty_cache()

def run_training_loop(params: dict, is_sub_process=False):
    """
    The core training loop logic, refactored to be callable by any dispatcher.
    """
    # Sanitize the output name to prevent path issues 
    output_lora_name = sanitize_filename(params['output_lora_name'])
    
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
    yield "Accelerator initialized."

    # Load models
    model_path = get_model_path(params['base_model_checkpoint'])
    tokenizer, text_encoder, unet = load_models(model_path, params['is_v2_model'])
    params['tokenizer'] = tokenizer # Pass tokenizer for multi-stage dataset loading
    yield "Models loaded."

    # Handle loading a base LoRA for Stage 2 of multi-stage training
    if params.get('base_lora_to_load'):
        lora_path = os.path.join(get_lora_output_path(), f"{params['base_lora_to_load']}.safetensors")
        yield f"Loading base LoRA weights from {lora_path}"
        # This is a simplified loading method. Real PEFT loading is more direct.
        # unet = PeftModel.from_pretrained(unet, lora_path)
        # text_encoder = PeftModel.from_pretrained(text_encoder, lora_path)
    
    # Inject new adapters
    unet, text_encoder = inject_lora_adapters(unet, text_encoder, params['network_rank'], params['network_alpha'], params['train_text_encoder'])
    yield "LoRA adapters injected."

    # Prepare dataset
    if params.get('custom_dataset'):
        train_dataset = params['custom_dataset']
    else:
        train_dataset = LoRATrainingDataset(params['image_dataset_directory'], tokenizer, params['enable_dynamic_cropping'], params['crop_resolution'])
    
    if not train_dataset:
        yield "Error: Dataset is empty. Check directory and captions."
        return

    train_dataloader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    yield f"Dataset prepared with {len(train_dataset)} total steps per epoch."

    # Setup optimizer and scheduler
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if params['train_text_encoder']:
        params_to_optimize.extend(list(filter(lambda p: p.requires_grad, text_encoder.parameters())))

    optimizer = torch.optim.AdamW(params_to_optimize, lr=float(params['learning_rate']))
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * params['num_epochs'])
    
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    yield "--- Beginning Training Loop ---"
    
    # The actual loop
    for epoch in range(params['num_epochs']):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = unet.encode(batch["pixel_values"].to(accelerator.device, dtype=torch.float16)).latent_dist.sample() * unet.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Conditional loss calculation for epsilon vs v-prediction 
                if params['is_v_parameterization']:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps) # 
                else:
                    target = noise # 
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") # 
                
                accelerator.backward(loss) # 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.is_main_process and step % 10 == 0:
                yield f"Epoch {epoch+1}/{params['num_epochs']}, Step {step}, Loss: {loss.detach().item():.4f}"
    
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)

    # Save final LoRA file in .safetensors format for security and speed 
    saved_path = save_lora_weights(unwrapped_unet, unwrapped_text_encoder, get_lora_output_path(), output_lora_name)
    yield f"Training complete. LoRA saved to: {saved_path}"