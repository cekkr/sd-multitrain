# /advanced-sd2-lora-trainer/install.py

import launch

# A list of required packages for LoRA training, including PEFT, diffusers, and 8-bit optimizers
# 
required_packages = [
    'peft==0.11.1',
    'diffusers==0.27.2',
    'accelerate==0.29.3',
    'bitsandbytes==0.43.1'
]

print("---")
print("Initializing Advanced SD2 LoRA Trainer")

for pkg in required_packages:
    # launch.is_installed is used to check if the package is already in the environment 
    package_name = pkg.split('==')[0]
    if not launch.is_installed(package_name):
        # launch.run_pip installs the package into the WebUI's Python virtual environment 
        print(f"Installing requirement: {pkg}")
        launch.run_pip(f"install {pkg}", f"Requirement for Advanced SD2 LoRA Trainer: {pkg}")
    else:
        print(f"Requirement already satisfied: {package_name}")

print("Advanced SD2 LoRA Trainer dependencies checked.")
print("---")