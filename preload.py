# /advanced-sd2-lora-trainer/preload.py

from modules.shared import cmd_opts
from modules.cmd_args import parser

def preload(parser):
    """
    This function is called by the WebUI launcher before parsing command-line arguments.
    Use it to add new arguments specific to your extension.
    """
    parser.add_argument(
        '--lora-trainer-default-rank', 
        type=int, 
        default=None, 
        help='Set a default network rank for the Advanced SD2 LoRA Trainer.'
    )
    
    # You can access this value later in your script via:
    # from modules.shared import cmd_opts
    # default_rank = cmd_opts.lora_trainer_default_rank
    print("Advanced LoRA Trainer: Preload script executed, custom arguments added.")