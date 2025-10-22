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
import sys

### FRAMEWORK
print("\n\nCopyright\n\n")
subject = "copyright"
main_dir = "/path/to/wd"
results_dir = "/path/to/Copyright/Results"
scorer_dir = "/path/to/Copyright/Scorer"
PCA_dir = "/path/to/Copyright/PCA"

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
prompt = f"cartoon mouse"
method = "c" # "o":original model, "c": constrained model
pipe.starting_step = 1 # step at which the constraint starts
pipe.score_target = 0 # "0" to project toward Jerry, "1" toward Mickey
pipe.discrete_label = True  # "True" = {0,1}, "False" = [0,1]
pipe.early_prj_stop_threshold = 20 # If the score is < of the thresold, avoid constraint in the following diffusion steps. "None" if not used
pipe.framework = "copyright"

# Model preparation
pipe.current_date  = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
pipe.constraint = True if method == "c" else False
pipe.early_prj_stop_flag = False

# Original model initialization
if not pipe.constraint:
    print(f"Model: Original")
    pipe.folder_name = f"run_{pipe.current_date}_original"
    saving_mode = "RGB"
    
# Constraint model initialization
if pipe.constraint:
    print(f"Model: Constraint (our model)")
    pipe.folder_name = f"run_{pipe.current_date}_projected"
    saving_mode = "RGB"
    pipe.scorer_dir = scorer_dir 
    pipe.PCA_dir = PCA_dir 

### UTILS
def convert_to_grayscale(image_tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=torch.device('cuda:0')).view(3, 1, 1)
    grayscale_image = (weights * image_tensor).sum(dim=0, keepdim=True)
    return grayscale_image


def save_tensor_as_image(tensor, file_path, saving_mode='RGB'):
    # Convert the tensor to a NumPy array
    tensor = tensor.cpu().detach().numpy()
    
    # Normalize the tensor to the range [0, 255] if necessary
    tensor = np.clip(tensor, -1, 1)
    tensor += 1
    tensor *= 0.5
    
    # Convert to 8-bit (0-255)
    tensor = (tensor * 255).astype(np.uint8)
    
    # Create an image object from the NumPy array
    tensor = np.transpose(tensor, (1, 2, 0))
    image = Image.fromarray(tensor, mode=saving_mode)

    # Save the image
    image.save(file_path)


### MAIN
dir_path = f"{results_dir}/{pipe.folder_name}/images"

for i in range(num_images):

    print(f"\nImage n {i+1}")
    pipe.image_count = i+1
    pipe.early_prj_stop_flag = False

    # Constraint model
    if pipe.constraint:
        file_name = f"{i+1}_projected.png"
        from copyright_resnet_binary import ResNet50Binary
        pipe.scorer_model = ResNet50Binary()
        pipe.scorer_model = pipe.scorer_model.to("cuda")
    
    # Original model
    else:
        file_name = f"{i+1}_original.png"
    
    # Set up the path and check if already exists
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    save_path = os.path.join(dir_path, file_name)
    if os.path.exists(save_path): 
        print(f"    Already exists")
        continue
    
    # Diffusion process
    pipe.first_cycle = True
    image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}) #.images[0]            

    # Save image
    save_tensor_as_image(image.squeeze(), save_path, saving_mode) 
