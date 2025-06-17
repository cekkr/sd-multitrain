# /advanced-sd2-lora-trainer/scripts/core/data_handler.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from advanced.dynamic_cropping import perform_random_crop

class LoRATrainingDataset(Dataset):
    """A PyTorch Dataset for loading and preparing training data for LoRA."""
    def __init__(self, image_dir, tokenizer, enable_cropping=False, crop_resolution=768):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.enable_cropping = enable_cropping
        self.crop_resolution = crop_resolution
        
        self.image_paths = []
        self.captions = []
        
        self._load_data()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _load_data(self):
        """
        Loads image paths and captions from the structured directory.
        Expects subfolders like '10_myconcept'. 
        """
        for subdir in os.listdir(self.image_dir):
            path = os.path.join(self.image_dir, subdir)
            if os.path.isdir(path):
                try:
                    repeats = int(subdir.split('_')[0])
                except (ValueError, IndexError):
                    print(f"Warning: Skipping subdirectory '{subdir}' due to incorrect naming format.")
                    continue
                
                image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_file in image_files:
                    base_name, _ = os.path.splitext(img_file)
                    caption_file = os.path.join(path, f"{base_name}.txt")
                    
                    if os.path.exists(caption_file):
                        with open(caption_file, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        
                        # Add the image and caption for the number of specified repeats
                        for _ in range(repeats):
                            self.image_paths.append(os.path.join(path, img_file))
                            self.captions.append(caption)
                    else:
                        print(f"Warning: Caption file not found for {img_file}, skipping.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply dynamic cropping if enabled
        if self.enable_cropping:
            image = perform_random_crop(image, self.crop_resolution)
        else:
            image = image.resize((self.crop_resolution, self.crop_resolution), Image.Resampling.LANCZOS)

        # Apply standard transformations
        pixel_values = self.transform(image)
        
        # Tokenize caption
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids.squeeze(0)}