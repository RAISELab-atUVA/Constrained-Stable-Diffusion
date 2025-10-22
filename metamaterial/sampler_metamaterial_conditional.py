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
import csv

### FRAMEWORK
print("\n\nMetamaterial conditional\n\n")
subject = "metamaterial_conditional"
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
num_images = 1
pipe.encoder_train = True
method = "cond" # "o":original model
csv_target_path = f"{main_dir}/metamaterial/TargetCurves/target_1.csv" # Target curve
pipe.framework = "metamaterial"

# Model preparation
pipe.current_date  = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
pipe.early_prj_stop_flag = False

# Original model initialization
print(f"Model: Conditional")
pipe.folder_name = f"run_{pipe.current_date}_conditional"
saving_mode = "L"

# Prompt - target
with open(csv_target_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    row = list(reader)[0]
    prompt = '-'.join([str(int(float(row[i]))) for i in range(0, 51, 5)])

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

    file_name = f"{i+1}_original.png"
    
    # Set up the path and check if already exists
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    save_path = os.path.join(dir_path, file_name)
    if os.path.exists(save_path): 
        print(f"    Already exists")
        continue
    
    # Diffusion process
    pipe.first_cycle = True
    image = pipe(prompt=prompt, height = 128, width = 128, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}) #.images[0]

    # Save image
    image = torch.mean(image, dim=1, keepdim=True)
    save_tensor_as_image(image.squeeze(), save_path, saving_mode) 
