# sd-multitrain
Experimental SD WebUI multi trainer extension

## Advanced SD2 LoRA Trainer

An extension for the AUTOMATIC1111 Stable Diffusion WebUI, designed for robust and flexible LoRA (Low-Rank Adaptation) training, with a special focus on the architectural nuances of Stable Diffusion 2.0 models.

This project provides a dedicated UI tab with advanced features, including dynamic image cropping and a novel multi-stage training workflow, all built upon a modular and extensible backend.

## Key Features

* [cite_start]**Dedicated UI Tab**: A clean, organized "Advanced SD2 LoRA Trainer" tab keeps training separate from image generation workflows[cite: 65, 70].
* **Full SD2.0 Support**:
    * [cite_start]Correctly loads the **OpenCLIP** text encoder required by SD2 models, addressing the "data divide" from SD1.x[cite: 20, 25, 112].
    * [cite_start]Features an essential **`v_parameterization`** checkbox to dynamically switch the loss function, preventing `NaN` loss errors when training v-prediction models[cite: 42, 44, 90].
    * [cite_start]Strongly recommends training the text encoder, which is often vital for teaching SD2 models new concepts from a lower baseline[cite: 33, 90].
* **Advanced Training Features**:
    * **Dynamic Cropping**: An optional data augmentation step that randomly crops images each epoch to improve the model's compositional understanding.
    * **Multi-Stage Training**: Train a general base LoRA, then automatically divide the dataset by text prompt keywords to fine-tune specialized LoRA variations, capturing more nuanced concepts.
* **Memory Efficiency**:
    * [cite_start]Includes an `install.py` script to automatically set up dependencies like `bitsandbytes`[cite: 53, 136].
    * [cite_start]Defaults to the **AdamW8bit optimizer** to significantly reduce VRAM usage with minimal impact on quality[cite: 91, 157].
* **Extensible by Design**: The project is structured into modular Python files, making it easy for developers to modify or add new features.

## User Guide

### Installation

1.  Navigate to the `extensions` folder in your `stable-diffusion-webui` directory.
2.  Clone or download this project's folder (`advanced-sd2-lora-trainer`) into the `extensions` directory.
3.  Restart the AUTOMATIC1111 WebUI. [cite_start]The required Python packages will be installed automatically on the first launch[cite: 52, 95].

### Standard Training Workflow

1.  **Navigate to the Trainer**: Open the "Advanced SD2 LoRA Trainer" tab in the WebUI.
2.  **Prepare Your Dataset**:
    * [cite_start]Your training images must be in a directory where each concept has its own sub-folder[cite: 102].
    * [cite_start]Name sub-folders using the format `[repeats]_[classname]`, for example, `20_mycharacter`[cite: 102, 103]. This tells the trainer to repeat each image in that folder 20 times per epoch.
    * [cite_start]For each image (e.g., `image01.png`), create a corresponding text file (`image01.txt`) with a detailed description[cite: 104].
    * **Important**: Do *not* include your main trigger word in the caption files. [cite_start]The LoRA learns to associate the trigger word with the features you *omit* from the description[cite: 106, 107].
3.  **Configure Core Settings**:
    * **Base Model Checkpoint**: Select the SD2.0 model you want to train on.
    * [cite_start]**`v2 Model` Checkbox**: **This must be checked** for any SD2.0 model to ensure the correct OpenCLIP text encoder is loaded[cite: 89, 111].
    * **`v_parameterization` Checkbox**: Check this **only** if your model name ends in `-v-` (e.g., `768-v-ema.ckpt`). [cite_start]A mismatch here is the most common cause of `NaN` loss[cite: 43, 90, 169].
    * **Image/Dataset Directory**: Enter the path to the main directory containing your `repeats_classname` sub-folders.
    * **Output LoRA Name**: Give your LoRA a unique filename.
4.  **Set Training Parameters**:
    * [cite_start]**Learning Rate**: A good starting point for the U-Net is `1e-4`[cite: 150]. [cite_start]If you are training the text encoder, use a lower rate like `5e-5` or `2e-5`[cite: 151].
    * **Number of Epochs**: The total number of training steps is key. [cite_start]Aim for 100-200 steps per training image[cite: 160]. Calculate epochs based on this target. [cite_start](Example: 20 images with 10 repeats = 200 steps/epoch. For a 2000-step target, you need 10 epochs [cite: 161, 162, 163, 164]).
    * [cite_start]**Network Rank & Alpha**: Rank determines the LoRA's capacity[cite: 152]. [cite_start]A rank of 32-64 is common for complex subjects[cite: 154]. Alpha scales its strength. [cite_start]A common practice is setting alpha to half the rank or simply `1`[cite: 155].
    * **Train Text Encoder**: **This should be checked**. [cite_start]It is highly recommended for SD2 models to properly learn new concepts[cite: 33, 90].
5.  **Start Training**: Click the "Start Training" button and monitor the progress in the "Status/Log" box. [cite_start]Your final `.safetensors` file will be saved in the `models/Lora` directory[cite: 132, 146].

### Using Advanced Features

#### Dynamic Cropping

* **Purpose**: To improve the model's understanding of different compositions and prevent overfitting to the exact framing of your training images.
* **How to Use**:
    1.  Go to the "Advanced Features" tab and open the "Advanced Augmentations" accordion.
    2.  Check the **`Enable Dynamic Cropping`** box.
    3.  Set the desired `Crop Resolution`. The script will randomly crop each image to this size during every epoch.

#### Multi-Stage Training

* **Purpose**: To create a general LoRA for a subject and then automatically fine-tune specialized variations (e.g., "close-up portrait" vs. "full body action shot") without extra manual work.
* **How to Use**:
    1.  In the "Advanced Features" tab, open the "Multi-Stage Training" accordion.
    2.  Check the **`Enable Multi-Stage Training`** box.
    3.  In the **`Concept A Keywords`** and **`Concept B Keywords`** text boxes, enter comma-separated words that distinguish your image subsets. (e.g., `close-up, portrait, face` for A and `full body, wide shot, outdoors` for B).
    4.  When you start training, the process will:
        * **Stage 1**: Train a base LoRA on all images for a portion of the total epochs and save it.
        * **Dataset Division**: Split your dataset into two groups based on which keywords appear in their captions.
        * **Stage 2**: Load the base LoRA and continue training it separately on each data subset.
    5.  **Output**: You will get three files: `my-lora_base.safetensors`, `my-lora_concept_A.safetensors`, and `my-lora_concept_B.safetensors`.

## Developer Guide

### Development Philosophy

This extension is built with a modular structure to make it easy to understand, modify, and extend. The UI is decoupled from the backend logic, and core functionalities are separated into their own modules.

### Project Structure

[cite_start]The project follows the standard AUTOMATIC1111 WebUI extension layout[cite: 47, 49].

```
advanced-sd2-lora-trainer/
├── scripts/
[cite_start]│   ├── lora_trainer_ui.py       # Main script for UI (Gradio) and callbacks [cite: 67]
│   ├── core/
│   │   ├── training_logic.py    # Central dispatcher and main training loop
│   │   ├── data_handler.py      # Dataset loading and preparation
│   │   ├── model_management.py  # Model loading, saving, and PEFT injection
│   │   └── project_utils.py     # Helper functions
│   └── advanced/
│       ├── multi_stage_trainer.py # Logic for the multi-stage process
│       └── dynamic_cropping.py  # Image augmentation logic
[cite_start]├── install.py                     # Manages Python dependencies [cite: 52]
[cite_start]├── preload.py                     # Adds custom command-line arguments [cite: 55]
├── javascript/
[cite_start]│   └── custom.js                  # Custom frontend behaviors [cite: 57]
├── localizations/
[cite_start]│   └── en.json                    # UI localization file [cite: 59]
[cite_start]└── style.css                      # Custom CSS for UI styling [cite: 58]
```

### Core Modules Explained

* [cite_start]**`lora_trainer_ui.py`**: Defines the entire Gradio interface using `gr.Blocks` for maximum layout control[cite: 81]. [cite_start]It uses the `on_ui_tabs` callback to create a new top-level tab[cite: 67].
* **`training_logic.py`**: Acts as a central dispatcher. The main `start_training` function gathers all UI parameters and decides whether to call the standard `run_training_loop` or hand off control to the `run_multi_stage_training` function from the advanced module.
* **`model_management.py`**:
    * Handles loading the correct U-Net and Text Encoder based on the `v2 Model` flag.
    * [cite_start]Uses the `peft` library to inject LoRA adapters into the models[cite: 114]. [cite_start]The `LoraConfig` specifies the `rank`, `alpha`, and, most importantly, the `target_modules` (the specific layers to adapt, like attention projections)[cite: 117, 118].
* [cite_start]**`data_handler.py`**: Contains the `LoRATrainingDataset` class, which handles parsing the `[repeats]_[class]` directory structure[cite: 102], reading captions, and applying transformations.

### How to Extend the Project

* **Adding a New Optimizer**:
    1.  Add the optimizer's name to the `gr.Dropdown` choices in `lora_trainer_ui.py`.
    2.  In `training_logic.py`, modify the optimizer setup section within `run_training_loop` to instantiate your new optimizer class when its name is selected.
* **Adding a New Advanced Training Method**:
    1.  Create a new Python file in the `advanced/` directory (e.g., `my_new_trainer.py`).
    2.  Inside, create a main function (e.g., `run_my_new_training`) that accepts the `training_params` dictionary and `yield`s status updates.
    3.  In `training_logic.py`, import your new function.
    4.  Add a UI control (e.g., a checkbox) in `lora_trainer_ui.py` to enable your new method.
    5.  In the `start_training` dispatcher function in `training_logic.py`, add an `elif` condition to call your new training function when its checkbox is checked.

## Troubleshooting

* **`OutOfMemoryError`**: This is the most common issue. [cite_start]Try the following in order[cite: 165]:
    * [cite_start]Reduce the **Batch Size** to 1[cite: 165].
    * [cite_start]Use an 8-bit optimizer like **AdamW8bit**[cite: 166].
    * [cite_start]Reduce the **Network Rank**[cite: 166].
    * [cite_start]Use WebUI command-line arguments like `--lowvram` or `--medvram`[cite: 167].
* [cite_start]**`NaN Loss` (Loss becomes "Not a Number")**: This almost always indicates a configuration mismatch[cite: 168].
    * [cite_start]**Check `v_parameterization`**: Ensure it is checked *only* for v-prediction models[cite: 169].
    * [cite_start]**Learning Rate Too High**: An excessively high learning rate can cause numerical instability[cite: 170].
* **Overfitting**: The LoRA produces results that are too rigid or "burnt."
    * [cite_start]Reduce the `Number of Epochs` or total training steps[cite: 172].
    * [cite_start]Lower the `Learning Rate`[cite: 172].
    * [cite_start]Add more variety to your training data or use data augmentation like Dynamic Cropping[cite: 172].
* **LoRA Has No Effect**: The generated image is not influenced by the LoRA.
    * [cite_start]**Incorrect `target_modules`**: The LoRA weights were injected into the wrong layers, so nothing was effectively trained[cite: 173]. This is a developer-side issue.
    * [cite_start]**Text Encoder Not Trained**: The concept may have been too new or complex for the model to learn through the U-Net alone, especially on SD2[cite: 174]. Ensure `Train Text Encoder` is checked.

## Bibliography

1.  *A Technical Guide to Implementing a LORA Training Extension for Stable Diffusion 2.0 in the AUTOMATIC1111 WebUI* (Provided Document)
2.  Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
3.  *Hugging Face PEFT Library Documentation*. [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft).