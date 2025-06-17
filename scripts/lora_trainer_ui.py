# /advanced-sd2-lora-trainer/scripts/lora_trainer_ui.py

import gradio as gr
from modules import scripts, call_callbacks
from modules.ui_components import FormRow

# Import the placeholder for our backend logic
# This will be properly implemented in the next steps.
from core.training_logic import start_training

class AdvancedLoraTrainerScript(scripts.Script):
    """
    A class to house the script's metadata and good practices, even when creating a dedicated tab.
    """
    def title(self):
        # Returns the display name of the script 
        return "Advanced SD2 LoRA Trainer"

    def show(self, is_img2img):
        # This script is not meant to be shown in the default scripts dropdown 
        return scripts.AlwaysVisible

def on_ui_tabs():
    """
    This function is registered with the on_ui_tabs callback to add a new top-level tab to the WebUI.
    """
    # Use gr.Blocks for full control over the UI layout 
    with gr.Blocks(analytics_enabled=False) as ui_component:
        gr.Markdown("# Advanced LoRA Trainer for Stable Diffusion 2.0")
        gr.Markdown("A comprehensive trainer with SD2-specific settings and advanced features.")

        with gr.Row():
            # The main action button to trigger the backend training process 
            start_training_button = gr.Button("Start Training", variant="primary")
            # A non-interactive textbox for real-time status updates and logs 
            status_output = gr.Textbox(label="Status/Log", interactive=False, lines=8, show_copy_button=True)

        with gr.Tabs():
            with gr.TabItem("Core Settings"):
                with gr.Accordion("1. Model & Data", open=True):
                    with FormRow():
                        # Dropdown to select the base model 
                        base_model_checkpoint = gr.Dropdown(label="Base Model Checkpoint", info="Select the base .ckpt or .safetensors file.", choices=["sd_v2-1_768-ema.safetensors", "sd_v2-1_512-ema-pruned.safetensors"]) # This would be populated dynamically in a full implementation
                        output_lora_name = gr.Textbox(label="Output LoRA Name", value="my-sd2-lora", info="Filename for the final trained LoRA. Will be saved to models/Lora.") [cite: 91]
                    with FormRow():
                        # Crucial checkboxes for SD2 architecture 
                        is_v2_model = gr.Checkbox(label="v2 Model", value=True, info="Flags the model as an SD2.x architecture. Loads the OpenCLIP text encoder.") [cite: 89]
                        is_v_parameterization = gr.Checkbox(label="v_parameterization", value=True, info="Enables v-prediction loss objective. Must be checked for models like 768-v-ema.ckpt.") [cite: 90]
                    with FormRow():
                        # Path to the training images 
                        image_dataset_directory = gr.Textbox(label="Image/Dataset Directory", placeholder="e.g., C:/Users/Me/MyTrainingImages", info="Path to the folder containing training images structured like 'repeats_classname'.")

                with gr.Accordion("2. Training Parameters", open=True):
                    with FormRow():
                        learning_rate = gr.Textbox(label="Learning Rate", value="1e-4", info="Learning rate for the U-Net. LoRA can tolerate higher rates than full fine-tuning.") [cite: 91, 150]
                        text_encoder_lr = gr.Textbox(label="Text Encoder Learning Rate", value="2e-5", info="A lower learning rate for the text encoder is often recommended.") [cite: 151]
                        train_batch_size = gr.Number(label="Batch Size", value=1, precision=0, info="Number of images to process in a single step. Directly impacts VRAM usage.") [cite: 91]
                        num_epochs = gr.Number(label="Number of Epochs", value=10, precision=0, info="Total number of times the training process iterates over the entire dataset.") [cite: 91]

                with gr.Accordion("3. LoRA Parameters", open=True):
                    with FormRow():
                        # Controls for LoRA rank and alpha 
                        network_rank = gr.Slider(label="Network Rank (dim)", minimum=1, maximum=256, step=1, value=32, info="The rank 'r' of the LoRA matrices. Higher ranks allow more detail but increase file size.") [cite: 90, 152]
                        network_alpha = gr.Slider(label="Network Alpha", minimum=1, maximum=256, step=1, value=16, info="Scaling factor. A common heuristic is to set this to half the rank.") [cite: 91, 155]
                    with FormRow():
                        # Option to train the text encoder is highly recommended for SD2 
                        train_text_encoder = gr.Checkbox(label="Train Text Encoder", value=True, info="Highly Recommended for SD2. Helps the model learn new concepts associated with trigger words.") [cite: 90]
                        optimizer_type = gr.Dropdown(label="Optimizer", choices=["AdamW8bit", "AdamW", "Lion", "AdaFactor"], value="AdamW8bit", info="AdamW8bit is recommended to reduce VRAM usage.") [cite: 91, 157]

            with gr.TabItem("Advanced Features"):
                with gr.Accordion("4. Advanced Augmentations", open=False):
                    enable_dynamic_cropping = gr.Checkbox(label="Enable Dynamic Cropping", value=False, info="Randomly crops images during training to improve composition understanding.")
                    crop_resolution = gr.Slider(label="Crop Resolution", minimum=256, maximum=1024, step=64, value=768, info="The target resolution for dynamic crops.")

                with gr.Accordion("5. Multi-Stage Training", open=False):
                    enable_multi_stage = gr.Checkbox(label="Enable Multi-Stage Training", value=False, info="Train a base LoRA, then split the dataset to train specialized versions.")
                    gr.Markdown("Define keyword sets to split your dataset. Images with captions matching a set will be used to train a specialized LoRA in Stage 2.")
                    concept_a_keywords = gr.Textbox(label="Concept A Keywords", placeholder="e.g., close-up, portrait, face")
                    concept_b_keywords = gr.Textbox(label="Concept B Keywords", placeholder="e.g., full body, wide shot, outdoors")
        
        # Collect all UI components into a list to pass to the backend
        all_inputs = [
            base_model_checkpoint, output_lora_name, is_v2_model, is_v_parameterization,
            image_dataset_directory, learning_rate, text_encoder_lr, train_batch_size, num_epochs,
            network_rank, network_alpha, train_text_encoder, optimizer_type,
            enable_dynamic_cropping, crop_resolution, enable_multi_stage,
            concept_a_keywords, concept_b_keywords
        ]

        # The run method of the button, linking the UI to the backend logic 
        start_training_button.click(
            fn=start_training, # The function to call from core/training_logic.py
            inputs=all_inputs,
            outputs=[status_output]
        )

        # Return the Gradio block, tab title, and a unique ID 
        return [(ui_component, "Advanced SD2 LoRA Trainer", "sd2_lora_trainer_tab")]

# Register the function to create the tab when the UI is built 
call_callbacks.add_extension_callbacks("on_ui_tabs", on_ui_tabs)