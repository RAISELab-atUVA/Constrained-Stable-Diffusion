import torch
from huggingface_hub import hf_hub_download, upload_file
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
from PIL import Image
from diffusers.models.autoencoders.vae import Encoder
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import json
from torchvision import models, transforms
from datetime import datetime
import pandas as pd

### FRAMEWORK
print("\n\nMetamaterial\n\n")
subject = "metamaterial"
main_dir = "/path/to/wd"
results_dir = "/path/to/Metamaterial/Results"

### PRETRAINED MODEL IMPORT
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")
lora_path = f'{main_dir}/models_training/{subject}_model'
pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
embedding_path = f"{main_dir}/models_training/{subject}_model/{subject}_model_emb.safetensors"
state_dict = load_file(embedding_path)
# load embeddings of text_encoder 1 (CLIP ViT-L/14)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# load embeddings of text_encoder 2 (CLIP ViT-G/14)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

### MODEL SETTINGS
num_images = 100
pipe.encoder_train = True
prompt = f"metamaterial"
method = "c" # "o":original model, "c":constraint model
csv_target_path = f"{main_dir}/metamaterial/TargetCurves/target_1.csv" # Target curve
pipe.starting_step = 24 # step at which the constraint starts
pipe.phase = 1
pipe.framework = "metamaterial"
pipe.abaqus_tmp_dir = f"{main_dir}/metamaterial/AbaqusTmp"

# Model preparation
pipe.current_date  = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
pipe.constraint = True if method == "c" else False
pipe.early_prj_stop_flag = False

# Original model initialization
if not pipe.constraint:
    print(f"Model: Original")
    pipe.folder_name = f"run_{pipe.current_date}_original"
    saving_mode = "L"

# Constraint model initialization
if pipe.constraint:
    print(f"Model: Constraint (our model)")
    pipe.folder_name = f"run_{pipe.current_date}_projected"
    saving_mode = "L"
    pipe.stress_strain_target = torch.tensor(pd.read_csv(csv_target_path, header=None).iloc[0].values, dtype=torch.float32).to("cuda:0")

### UTILS
def save_tensor_as_image(tensor, file_path, saving_mode='L'):
    # Convert the tensor to a NumPy array
    tensor = tensor.cpu().detach().numpy()
    
    # Normalize the tensor to the range [0, 255] if necessary
    tensor = np.clip(tensor, -1, 1)
    tensor += 1
    tensor *= 0.5
    
    # Convert to 8-bit (0-255)
    tensor = (tensor * 255).astype(np.uint8)
    
    # Create an image object from the NumPy array
    image = Image.fromarray(tensor, mode=saving_mode)

    # Save the image
    image.save(file_path)


### MAIN
dir_path = f"{results_dir}/{pipe.folder_name}/images"

for i in range(num_images):

    print(f"\nImage n {i+1}")
    pipe.image_count = i+1

    # constraint model
    if pipe.constraint:
        file_name = f"{i+1}_projected.png"
    
    # Original model
    else:
        file_name = f"{i+1}_original.png"
    
    # Set up the path and check if already exists
    if not pipe.phase == 1:
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, file_name)
        if os.path.exists(save_path): 
            print(f"    Already exists")
            continue
    
    # Diffusion process
    pipe.first_cycle = True
    image = pipe(prompt=prompt, height = 128, width = 128, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}) #.images[0]

    # Save image
    if pipe.phase == 2:
        image = torch.mean(image, dim=1, keepdim=True)
        save_tensor_as_image(image.squeeze(), save_path, saving_mode) 
