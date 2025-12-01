module purge

MAIN_PATH="/path/to/wd"
DATASET_PATH="/path/to/Metamaterial/Dataset"
SUBJECT="metamaterial"
PROMPT="metamaterial"

pip install peft
pip install appdirs
pip install sentry_sdk
pip install prodigyopt
pip install datasets
pip install  wandb

cd "$MAIN_PATH/diffusers"
pip install -e .

cd "$MAIN_PATH/models_training"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="$DATASET_PATH"
export OUTPUT_DIR="${SUBJECT}_model"
export PROMPT="$PROMPT"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

python train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --pretrained_vae_model_name_or_path="$VAE_PATH" \
  --dataset_name="$DATASET_NAME" \
  --instance_prompt="$PROMPT" \
  --validation_prompt="$PROMPT" \
  --output_dir="$OUTPUT_DIR" \
  --caption_column="prompt" \
  --mixed_precision="fp16" \
  --resolution=96 \
  --train_batch_size=3 \
  --repeats=1 \
  --report_to="wandb"\
  --gradient_accumulation_steps=5 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --train_text_encoder_ti_frac=0.5\
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=1000 \
  --checkpointing_steps=2000 \
  --seed="0"

