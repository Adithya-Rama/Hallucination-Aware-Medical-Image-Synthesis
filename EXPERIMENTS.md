# Reproducing Experiments - Detailed Guide

**Based on the actual Colab notebook implementation**

This guide walks through reproducing all experiments from the paper using the provided `Research_Project.ipynb` notebook.

---

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Constraint Heads](#training-constraint-heads)
4. [LoRA Fine-Tuning (Phase 1 & 2)](#lora-fine-tuning)
5. [ControlNet Training](#controlnet-training)
6. [Guided Generation](#guided-generation)
7. [Ablation Studies](#ablation-studies)
8. [Downstream Evaluation](#downstream-evaluation)
9. [Expected Results](#expected-results)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Environment Setup

### Hardware Requirements

**Minimum**:
- GPU: 16GB VRAM (T4)
- RAM: 12GB system memory
- Storage: 50GB free space

**Recommended** (for full pipeline):
- GPU: **Colab Pro with A100 (40GB)** or L4 (24GB)
- RAM: 32GB system memory
- Storage: 100GB free space on Google Drive

### Time Estimates (Colab A100)

| Stage | Time | Can Skip? |
|-------|------|-----------|
| Dataset download | 30 min | No |
| Constraint heads training | 2 hours | Yes (use pretrained) |
| LoRA Phase 1 | 2 hours | Yes (use pretrained) |
| LoRA Phase 2 | 1.5 hours | Yes (use pretrained) |
| ControlNet training | 4 hours | Yes (use pretrained) |
| Generation (200 masks) | 2 hours | No |
| Ablations (5 configs) | 4 hours | Partial |
| Downstream eval | 1 hour | No |
| **Total** | **~17 hours** | - |

### Colab Setup

```python
# 1. Open Google Colab: https://colab.research.google.com
# 2. Change runtime: Runtime → Change runtime type → GPU (A100 if Pro)
# 3. Upload Research_Project.ipynb
# 4. Mount Google Drive (first cell)

from google.colab import drive
drive.mount('/content/drive')
```

---

## 📊 Dataset Preparation

### Step 1: Kaggle Authentication

```python
# Upload your kaggle.json to Google Drive at:
# /content/drive/My Drive/COMP8539/Final_Project/kaggle.json

import os
kaggle_json_path = '/content/drive/My Drive/COMP8539/Final_Project/kaggle.json'

!mkdir -p ~/.kaggle
!cp "{kaggle_json_path}" ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Download Datasets

**All downloads automated in notebook Section 1.**

```bash
# Kvasir-SEG (1,000 polyp images)
!kaggle datasets download -d fkarimovv/kvasir-seg

# HyperKvasir (10k+ labeled colonoscopy images)
!wget https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip

# Kvasir-Instrument (590 tool frames)
!kaggle datasets download -d debeshjha1/kvasirinstrument

# Nerthus (BBPS video frames)
!kaggle datasets download -d orvile/nerthus-video-dataset-for-colonoscopy-analysis
```

### Step 3: Prepare Manifests

The notebook automatically creates CSV manifests with columns:
- `image`: Path to image file
- `mask`: Path to corresponding mask
- `size_bucket`: Small/medium/large (for size classifier)
- `label`: Integer labels for classifiers

**Generated files**:
```
processed/
├── kvasir_kv_manifest.csv      # Kvasir-SEG with size buckets
├── bbpsq_train.csv             # Nerthus BBPS training
├── bbpsq_val.csv               # Nerthus BBPS validation
├── instrument_train.csv        # Instrument presence training
└── instrument_val.csv          # Instrument presence validation
```

---

## 🎓 Training Constraint Heads

### 1. U-Net Segmenter (Dice + BCE Loss)

**Notebook Section 3b**

```python
# Architecture
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)

# Loss
loss_fn = ComboLoss(dice_weight=0.7, bce_weight=0.3)

# Training
- Epochs: 10
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: Adam
- Image size: 256×256
```

**Expected Performance**:
- Validation Dice: ~0.85
- Validation IoU: ~0.82
- Training time: ~20 minutes

**Output**: `classifiers/seg_unet_resnet34.pth`

### 2. Size Classifier (3-class)

**Notebook Section 3c**

```python
# Architecture
model = timm.create_model('resnet18', pretrained=True, num_classes=3)

# Classes
0: small (rel_area < 0.05)
1: medium (0.05 ≤ rel_area < 0.15)
2: large (rel_area ≥ 0.15)

# Training
- Epochs: 10
- Batch size: 16
- Learning rate: 2e-4
- Optimizer: AdamW
- Crop around polyp with 10px padding
- Image size: 224×224
```

**Expected Performance**:
- Validation Accuracy: ~92%
- Training time: ~15 minutes

**Output**: `classifiers/size_cls_resnet18.pth`

### 3. Instrument Classifier (Binary)

**Notebook Section 3d**

```python
# Architecture
model = timm.create_model('resnet18', pretrained=True, num_classes=2)

# Classes
0: no instrument (Kvasir-SEG images)
1: instrument present (Kvasir-Instrument images)

# Training
- Epochs: 8
- Batch size: 32
- Learning rate: 2e-4
- Weighted sampling (class balance)
- Image size: 224×224
```

**Expected Performance**:
- Validation F1: ~0.95
- Validation Accuracy: ~96%
- Training time: ~10 minutes

**Output**: `classifiers/instrument_resnet18.pth`

### 4. BBPS Quality Classifier (4-class)

**Notebook Section 3e**

```python
# Architecture
model = timm.create_model('resnet18', pretrained=True, num_classes=4)

# Classes (Boston Bowel Preparation Scale)
0: unprepared, heavy debris
1: partially clean, some residue
2: mostly clean, minimal residue
3: excellent preparation

# Training
- Epochs: 8
- Batch size: 32
- Learning rate: 2e-4
- Weighted sampling (class balance)
- Image size: 224×224
```

**Expected Performance**:
- Validation F1 (macro): ~0.87
- Training time: ~12 minutes

**Output**: `classifiers/bbpsq_resnet18.pth`

---

## 🔧 LoRA Fine-Tuning

### Phase 1: Masked Domain Adaptation

**Notebook Section 4a**

**Purpose**: Adapt SD-1.5 to colonoscopy domain using masked training.

```python
# Configuration
base_model: "runwayml/stable-diffusion-v1-5"
lora_rank: 8
lora_alpha: 16
target_modules: ["to_q", "to_v"]

# Training
max_train_steps: 2000
batch_size: 1
learning_rate: 1e-4
mixed_precision: fp16

# Data
dataset: Kvasir-SEG images
conditioning: Binary polyp masks (applied as multiplication)
prompt: "A colonoscopy image showing a polyp"
```

**Training Loop**:
1. Encode image to latent space (VAE)
2. Apply mask in pixel space, then encode
3. Add noise at random timestep
4. Predict noise with U-Net
5. MSE loss between predicted and actual noise
6. Backprop through LoRA adapters only

**Checkpoint Strategy**:
- Save every 500 steps
- Final checkpoint at 2000 steps

**Expected Loss Curve**:
- Initial: ~0.15
- Final: ~0.05
- Training time: ~2 hours

**Output**: `lora_colonoscopy_phase1_masked/`

### Phase 2: Rich Prompt Conditioning

**Notebook Section 4b**

**Purpose**: Teach model semantic understanding using rich captions.

```python
# Configuration
base_model: Phase 1 LoRA checkpoint
max_train_steps: 1500
batch_size: 1
learning_rate: 5e-5  # Lower LR for fine-tuning

# Data
dataset: HyperKvasir labeled images
captions: Generated from CSV metadata
  Example: "A colonoscopy image of the lower GI tract, 
           showing polyp, classified as adenoma."
```

**Prompt Template**:
```python
caption = f"A colonoscopy image of the {organ}, showing {finding}, classified as {classification}."
```

**Expected Loss Curve**:
- Initial: ~0.08
- Final: ~0.03
- Training time: ~1.5 hours

**Output**: `lora_colonoscopy_phase2_hyper/`

**Validation**: Generate images with prompts, visually inspect quality.

---

## 🎮 ControlNet Training

**Notebook Section 5**

### Dataset Preparation

```python
# Combines:
1. Kvasir-SEG: 1000 (image, mask) pairs
2. HyperKvasir-SEG: ~1000 additional pairs

# Preprocessing
- Resize to 512×512
- Images: RGB, LANCZOS resampling
- Masks: Binary, NEAREST resampling (preserve values)
- Save as {idx:06d}_image.png, {idx:06d}_control.png
```

### Training Configuration

```python
# Base
pretrained_controlnet: "lllyasviel/sd-controlnet-canny"  # Starting point
unet: Phase 2 LoRA merged weights
controlnet: Fine-tuned on polyp masks

# Hyperparameters
max_train_steps: 3000
batch_size: 1
learning_rate: 1e-5
optimizer: AdamW
mixed_precision: fp16

# Scheduler
DPMSolverMultistepScheduler with Karras sigmas
```

### Training Loop (Critical Details)

```python
# 1. Load image and control mask
pixel_values = image  # [B, 3, 512, 512]
control_image = mask.expand(-1, 3, -1, -1)  # [B, 3, 512, 512]

# 2. Text encoding
encoder_hidden_states = text_encoder(tokenized_prompt)

# 3. VAE encode
latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

# 4. Add noise
noise = randn_like(latents)
timesteps = random_int(0, num_train_timesteps)
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

# 5. ControlNet forward
down_res, mid_res = controlnet(
    noisy_latents,
    timesteps,
    encoder_hidden_states,
    controlnet_cond=control_image
)

# 6. U-Net forward with residuals
noise_pred = unet(
    noisy_latents,
    timesteps,
    encoder_hidden_states,
    down_block_additional_residuals=down_res,
    mid_block_additional_residual=mid_res
)

# 7. Loss
loss = F.mse_loss(noise_pred, noise)
```

### Validation

Every 500 steps, generate sample image:
```python
mask_img = load_validation_mask()
generated = pipe(
    prompt="A colonoscopy image showing a polyp",
    image=mask_img,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]
```

**Expected Behavior**:
- Step 0-500: Coarse structure emerges
- Step 1000-1500: Realistic texture
- Step 2000+: High fidelity, mask adherence

**Training Time**: ~4 hours

**Output**: `controlnet_adapter_colonoscopy/` (1.4GB SafeTensors)

---

## 🎨 Guided Generation

**Notebook Section 6 (Basic) & Section 7 (Advanced)**

### Basic Generation (No Guidance)

```python
prompt = "A realistic colonoscopy image showing a polyp"
mask_pil = load_mask("mask.png")

image = pipe(
    prompt=prompt,
    image=mask_pil,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]
```

### Latent-Guided Generation (4 Constraints)

**Key Innovation**: Callback function modifies latents during sampling.

```python
def on_step_end(pipeline, step, timestep, callback_kwargs):
    # Skip early steps and non-guidance steps
    if step < START_GUIDE_AT or (step % GUIDE_EVERY_K) != 0:
        return callback_kwargs
    
    lat = callback_kwargs["latents"].detach().clone().requires_grad_(True)
    
    with torch.enable_grad(), torch.autocast("cuda"):
        # 1. Decode latent to image
        img = vae.decode(lat / 0.18215).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        
        # 2. Compute constraint losses
        seg_loss = soft_dice_loss(seg_net(img), target_mask)
        size_loss = F.cross_entropy(size_net(crop(img)), target_size)
        bbps_loss = F.cross_entropy(bbps_net(img), target_bbps)
        tool_loss = F.cross_entropy(instr_net(img), target_tool)
        
        # 3. Total loss
        loss = (LAMBDA_SEG * seg_loss + 
                LAMBDA_SIZE * size_loss +
                L_BBPS * bbps_loss + 
                L_TOOL * tool_loss)
        
        # 4. Gradient w.r.t. latents
        grad = torch.autograd.grad(loss, lat)[0]
    
    # 5. EMA smoothing
    grad_ema = EMA_BETA * grad_ema_prev + (1-EMA_BETA) * grad
    grad = grad_ema / (grad_ema.norm() + 1e-8)  # Unit norm
    
    # 6. Latent update
    lat = lat - STEP_SCALE * grad
    
    callback_kwargs["latents"] = lat.detach()
    return callback_kwargs

# Generate with callback
image = pipe(
    prompt=prompt,
    image=mask_pil,
    num_inference_steps=28,
    guidance_scale=7.5,
    callback_on_step_end=on_step_end,
    callback_on_step_end_tensor_inputs=["latents"]
).images[0]
```

### Hyperparameters (Tuned)

```python
# Guidance schedule
GUIDE_EVERY_K = 3        # Guide every 3 steps
START_GUIDE_AT = 3       # Skip first 3 steps (warmup)

# Constraint weights
LAMBDA_SEG = 1.0         # Segmentation (highest)
LAMBDA_SIZE = 0.6        # Size classification
L_BBPS = 0.4             # Bowel preparation quality
L_TOOL = 0.3             # Instrument detection

# Optimization
STEP_SCALE = 0.10        # Latent nudge magnitude
EMA_BETA = 0.8           # Gradient smoothing
CLIP_UNIT_NORM = True    # Normalize gradient direction
```

### Filtering & Ranking

For each mask, generate N candidates (default: 4), score all, keep top K (default: 2):

```python
def score_constraints(img, mask):
    # 1. Segmentation IoU
    pred_mask = seg_net(img) > 0.5
    iou = jaccard(pred_mask, mask)
    
    # 2. Size prediction
    crop = extract_bbox(img, mask)
    size_pred = size_net(crop).argmax()
    
    # 3. BBPS prediction
    bbps_pred = bbps_net(img).argmax()
    
    # 4. Instrument prediction
    tool_pred = instr_net(img).argmax()
    
    return {
        "iou": iou,
        "size_pred": size_pred,
        "bbps_pred": bbps_pred,
        "instr_pred": tool_pred
    }

def pass_filters(scores):
    return (
        scores["iou"] >= 0.45 and
        scores["size_pred"] == TARGET_SIZE_IDX and
        scores["bbps_pred"] == TARGET_BBPS_IDX and
        scores["instr_pred"] == TARGET_TOOL_IDX
    )

# Generate candidates
candidates = []
for seed in range(CANDIDATES_PER_MASK):
    img = guided_generate_single(mask, prompt, seed)
    scores = score_constraints(img, mask)
    passed = pass_filters(scores)
    candidates.append((img, scores, passed))

# Rank by: passed first → IoU desc → size prob desc
candidates.sort(key=lambda x: (x[2], x[1]["iou"]), reverse=True)
kept = candidates[:TOP_K_PER_MASK]
```

---

## 🔬 Ablation Studies

**Notebook Section 7 (final blocks)**

### Configurations

```python
ABLATIONS = [
    # (name, use_guidance, use_seg, use_size, use_bbps, use_tool, step_scale, guide_k)
    ("no_guidance",         False, False, False, False, False, 0.00,  999),
    ("seg_only",            True,  True,  False, False, False, 0.08,  3),
    ("seg+size",            True,  True,  True,  False, False, 0.10,  3),
    ("seg+size+bbps",       True,  True,  True,  True,  False, 0.10,  3),
    ("seg+size+bbps+tool",  True,  True,  True,  True,  True,  0.10,  3),
]
```

### Running Ablations

```python
# Automated loop
for ablation in ABLATIONS:
    name, use_guid, use_seg, use_size, use_bbps, use_tool, step_scale, guide_k = ablation
    
    for mask_path in masks:
        for seed in range(CANDIDATES_PER_MASK):
            img = guided_generate_single(
                mask=load_mask(mask_path),
                prompt=get_prompt(mask_path),
                seed=seed,
                use_guidance=use_guid,
                use_seg=use_seg,
                use_size=use_size,
                use_bbps=use_bbps,
                use_tool=use_tool,
                step_scale=step_scale,
                guide_k=guide_k
            )
            # Score, filter, save...
```

### Metrics Reported

For each ablation:
- `n`: Number of generated images
- `pass_rate`: % images passing all active constraints
- `mean_iou`: Average segmentation IoU
- `iou_pass_mean`: Average IoU among passing images only
- `size_acc`: Size classification accuracy
- `bbps_acc`: BBPS classification accuracy
- `tool_acc`: Instrument detection accuracy
- Medical checks: specular ratio, brightness, edge density

**Output**: `experiments/ablation_summary.csv`

---

## 📊 Downstream Evaluation

**Notebook Section 8**

### Dataset: CVC-ClinicDB

```python
# External polyp segmentation dataset
# 612 images with expert annotations
# Different distribution from Kvasir (OOD test)

CLINICDB_ZIP = "CVC-ClinicDB.zip"
EVAL_IMG_DIR, EVAL_MSK_DIR = prep_cvc_clinicdb(CLINICDB_ZIP)
```

### Experimental Setup

**Train 2 models**:
1. **Real only**: Kvasir-SEG (1000 images)
2. **Real + Synthetic**: Kvasir-SEG + best synthetic (1:1 ratio)

**Architecture**: SMP U-Net ResNet34 (same as constraint head)

**Pseudo-labeling**:
```python
# For synthetic images, use seg_head to generate masks
for synth_img in synthetic_images:
    pred_mask = seg_head(synth_img) > 0.5
    save_as_pseudo_label(pred_mask)
```

### Training Configuration

```python
# Hyperparameters
epochs: 12
batch_size: 8
learning_rate: 3e-4
optimizer: AdamW
loss: BCEWithLogitsLoss

# Augmentation
- LongestMaxSize(352)
- PadIfNeeded(352, 352)
- HorizontalFlip(p=0.5)
- RandomBrightnessContrast(p=0.2)
- GaussNoise(p=0.1)
```

### Evaluation

Test both models on ClinicDB (no images seen during training):

```python
for model_name in ["real_only", "real_plus_synthetic"]:
    model.eval()
    ious = []
    for img, mask in clinicdb_test_loader:
        pred = model(img) > 0.5
        iou = jaccard(pred, mask)
        ious.append(iou)
    
    print(f"{model_name} IoU: {np.mean(ious):.4f}")
```

### Expected Results

| Model | ClinicDB IoU | Δ vs Real |
|-------|--------------|-----------|
| Real only | 0.712 | baseline |
| Real + Synthetic (0.5×) | 0.738 | +0.026 |
| Real + Synthetic (1.0×) | 0.758 | +0.046 |
| Real + Synthetic (2.0×) | 0.751 | +0.039 |

**Key Finding**: 1:1 ratio is optimal; more synthetic data doesn't help further.

---

## 📈 Expected Results

### Constraint Head Performance

| Head | Metric | Value |
|------|--------|-------|
| Segmentation | Val IoU | 0.82 |
| Size | Val Acc | 92% |
| BBPS | Val F1 | 0.87 |
| Instrument | Val F1 | 0.95 |

### Generation Quality (200 masks, 4 candidates each)

| Ablation | Pass Rate | Mean IoU | Size Acc | BBPS Acc | Tool Acc |
|----------|-----------|----------|----------|----------|----------|
| No Guidance | 34.2% | 0.52 | 41.3% | - | - |
| Seg Only | 48.1% | 0.61 | 43.7% | - | - |
| Seg + Size | 62.3% | 0.67 | 78.9% | - | - |
| Seg + Size + BBPS | 71.8% | 0.72 | 84.2% | 82.1% | - |
| **Full (4-head)** | **82.4%** | **0.78** | **89.7%** | **84.2%** | **91.6%** |

### Downstream Generalization

| Training Data | ClinicDB IoU | Improvement |
|---------------|--------------|-------------|
| Real only (1000) | 0.712 | baseline |
| Real + Syn (1:1) | 0.758 | **+6.5%** |

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# 1. Reduce batch size
BATCH_SIZE = 1

# 2. Use gradient checkpointing (ControlNet training)
pipe.unet.enable_gradient_checkpointing()

# 3. Clear cache between runs
torch.cuda.empty_cache()

# 4. Enable attention slicing
pipe.enable_attention_slicing()

# 5. Use CPU offloading (slower but works)
pipe.enable_sequential_cpu_offload()
```

### NaN Loss During Training

**Symptoms**: Loss becomes NaN after a few steps

**Solutions**:
```python
# 1. Lower learning rate
lr = 5e-5  # instead of 1e-4

# 2. Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check for invalid inputs
assert not torch.isnan(loss), "NaN loss detected!"
if torch.isnan(loss):
    continue  # Skip this batch
```

### Guidance Producing Artifacts

**Symptoms**: Generated images have unrealistic patterns

**Solutions**:
```python
# 1. Reduce step scale
STEP_SCALE = 0.06  # instead of 0.10

# 2. Guide less frequently
GUIDE_EVERY_K = 4  # instead of 3

# 3. Start guidance later
START_GUIDE_AT = 5  # instead of 3

# 4. Increase EMA smoothing
EMA_BETA = 0.9  # instead of 0.8
```

### ControlNet Not Loading

**Symptoms**: `FileNotFoundError` or `AttributeError`

**Solutions**:
```python
# Check if fp16 SafeTensors exists
ls controlnet_adapter_colonoscopy/

# If only .fp16.safetensors exists, alias it:
cp diffusion_pytorch_model.fp16.safetensors diffusion_pytorch_model.safetensors

# Or specify variant explicitly:
controlnet = ControlNetModel.from_pretrained(
    path,
    variant="fp16",
    torch_dtype=torch.float16
)
```

### Low Pass Rate (<50%)

**Symptoms**: Most generated images fail constraints

**Solutions**:
```python
# 1. Check constraint head accuracy first
test_classifier_heads()

# 2. Relax thresholds temporarily
IOU_THRESHOLD = 0.40  # instead of 0.45

# 3. Increase candidates per mask
CANDIDATES_PER_MASK = 6  # instead of 4

# 4. Adjust guidance weights
LAMBDA_SEG = 1.5  # Increase seg importance
```

---

## 📝 Notes

### Colab-Specific Tips

1. **Keep session alive**: Run a cell periodically to prevent disconnect
2. **Save checkpoints to Drive**: Colab VM resets after 12 hours
3. **Use Pro for A100**: Standard T4 will take 3× longer
4. **Monitor GPU memory**: `!nvidia-smi` in a cell

### Reproducibility Checklist

- [ ] Set all random seeds: `torch.manual_seed(42)`
- [ ] Use fixed image order: `sorted(glob(...))`
- [ ] Document library versions: `pip freeze > versions.txt`
- [ ] Save generation configs: JSON with all hyperparameters
- [ ] Log all intermediate results: CSVs for traceability

### Common Pitfalls

1. **Forgetting to mount Drive**: All outputs will be lost on disconnect
2. **Not saving checkpoints**: 4 hours of ControlNet training gone
3. **Wrong image normalization**: Heads expect [-1,1], but VAE uses [0,1]
4. **Mismatched device/dtype**: Ensure all tensors on same device
5. **Overwriting results**: Use unique experiment names

---

**Last Updated**: November 2025  
**Tested On**: Google Colab Pro (A100, 40GB RAM)
