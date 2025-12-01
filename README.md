<div align="center"><h1>&nbsp;Training-Free Constrained Generation With Stable Diffusion Models</h1></div>

<p align="center">
  <a href="https://arxiv.org/abs/2502.05625" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-2402.03559-red.svg" alt="arXiv">
</a>
<a href="https://neurips.cc/virtual/2025/poster/117807" target="_blank">
  <img src="https://img.shields.io/badge/NeurIPS%20Poster-2025-blue.svg" alt="NeurIPS Poster">
</a>
<a href="">
    <img src="https://img.shields.io/badge/Version-v1.0.0-orange.svg" alt="Version">
  </a>

</p>


**Training-Free Constrained Generation With Stable Diffusion Models** proposes a novel integration of stable diffusion models with constrained optimization frameworks, enabling the generation of outputs satisfying stringent physical and functional requirements. The effectiveness of this approach is demonstrated through material design experiments requiring adherence to precise morphometric properties, challenging inverse design tasks involving the generation of materials inducing specific stress-strain responses, and copyright-constrained content generation tasks.


## Overview

This repository includes three frameworks for different applications, each addressing a specific constraint:

1. **Porosity**: Generates microstructures with a desired level of porosity.
2. **Metamaterials**: Inverse design of mechanical metamaterials with specific stress-strain curves.
3. **Copyright**: Avoids generating images that resemble copyrighted content.

Additionally, for the **Metamaterials** and **Porosity** applications, conditional models are provided as baselines, but they do not ensure constraint satisfaction.

In total, the repository contains **five frameworks**:

- **Three constrained frameworks** (Copyright, Metamaterials, Porosity)
- **Two conditional models** (Metamaterials Conditional, Porosity Conditional)

## Repository Structure

The repository is structured into the following directories:


### 1. `diffusers/`

Contains the core **Stable Diffusion** library, mainly adapted from Hugging Face's `diffusers`. The primary modification is in:

```
/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
```


### 2. `models_training/`

Includes scripts necessary for training **Stable Diffusion** models:

- `metadata_generation_xxx.py` : Generate a JSON file containing metadata for training samples (set path imgs\_directory).
- Training scripts for each framework:
  - `run_xxx.sh`: Launch training (set MAIN\_PATH: Working directory and DATASET\_PATH: Dataset directory).
  - `train_dreambooth_lora_sdxl_advanced.py`: The main training script.


### 3. `porosity/`

Implements the **Porosity Constraint** framework:

- Generates microstructures with a **desired porosity** level.
- Uses a **Top-K algorithm** for correction.

#### Steps:

1. After training, use:
   ```
   sampler_porosity.py
   ```
   - Unconstrained mode: `method="o"`
   - Constrained mode: `method="c"` (uses `projection.py`).

2. Conditional baseline:
   ```
   sampler_porosity_conditional.py
   ```
   - Does not ensure constraint satisfaction.



### 4. `metamaterials/`

Implements the **Metamaterials Constraint** framework:

- Used for **inverse design** of mechanical metamaterials with predefined stress-strain curves.
- Uses a **Differentiable Perturbed Optimizer (DPO)** combined with **Abaqus** for projection. Currently the process is manual.

#### Steps:

1. After training, use:
   ```
   sampler_metamaterial.py
   ```
   - Unconstrained mode: `method="o"`
   - Constrained mode: `method="c"` (requires Abaqus collaboration).
   1. **Manual Abaqus Process**:
      - Set `pipe.phase=1` to save an image in `AbaqusTmp/Latest_run`.
      - Run `do.py` to generate the corrected image in Abaqus.
      - The corrected image is saved with the suffix `_aligned`.
      - Set `pipe.phase=2` to complete denoising.
2. Conditional baseline:
   ```
   sampler_metamaterial_conditional.py
   ```
   - Does not ensure constraint satisfaction.


### 5. `copyright/`

Implements the **Copyright Constraint** framework:

- The model detects whether a generated image resembles copyrighted content and redirects the generation toward non-copyrighted subjects.
- The classifier (`scorer`) is trained to distinguish copyrighted from non-copyrighted samples.
- Principal Component Analysis (PCA-2) is applied on the last layer of the scorer (2048 features) to define clusters for copyrighted and non-copyrighted samples.
- If a generated sample falls within the copyrighted cluster, it is projected towards the non-copyrighted cluster centroid while staying within the training distribution.
- Projection occurs at high noise levels, requiring a noisy image classifier.

#### Steps:

1. After training, use:
   ```
   sampler_copyright.py
   ```
   - Unconstrained mode: `method="o"`
   - Constrained mode: `method="c"` (requires pre-trained Scorer and PCA metadata)



## Requirements

- Python 3.8+
- Hugging Face `diffusers`
- PyTorch, CUDA
- Abaqus (for **Metamaterials** framework)
- Standard ML libraries (`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `opencv`, etc.)





## Reference
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2502.05625).
```
@inproceedings{zampini2025training,
  title={Training-free constrained generation with stable diffusion models},
  author={Zampini, Stefano and Christopher, Jacob K and Oneto, Luca and Anguita, Davide and Fioretto, Ferdinando},
	booktitle = {Neural Information Processing Systems},
	year = {2025}
}
```

## Acknowledgements

This research is partially supported by NSF grants 2334936, 2334448, and NSF CAREER Award 2401285. The authors acknowledge Research Computing at the University of Virginia for providing computational resources that have contributed to the results reported within this paper. The views and conclusions of this work are those of the authors only.
