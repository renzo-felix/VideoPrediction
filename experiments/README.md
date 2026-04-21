# Video Steering Vectors

Extracting and testing steering vectors for concepts in video foundation models (V-JEPA, VideoMAE).

## Setup

```bash
conda activate video
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate training videos (135 videos: 3 sizes × 3 colors × 5 speeds × 3 reps)
python src/generate_videos.py --mode training

# 2. Generate test videos (compositionally novel objects)
python src/generate_videos.py --mode test --category size
python src/generate_videos.py --mode test --category color
python src/generate_videos.py --mode test --category shape
python src/generate_videos.py --mode test --category material
python src/generate_videos.py --mode test --category texture

# 3. Extract activations (layer 9 by default)
python src/extract_activations.py --mode training --model vjepa
python src/extract_activations.py --mode training --model videomae

# 4. Train steering vectors (stratified method, n=4 per stratum)
python src/train_steering.py --model vjepa --multi --method stratified --n-per-stratum 4
python src/train_steering.py --model videomae --multi --method stratified --n-per-stratum 4

# 5. Test transfer across categories
python src/test_transfer.py --model vjepa
python src/test_transfer.py --model videomae
```

## Results

### V-JEPA (Stratified n=4)
- Speed correlation: r=0.673
- Size confound: r=0.506
- Best PC3: r=0.556 (cleaner, r_size=0.040)

### VideoMAE (Stratified n=4)
- Speed correlation: r=0.573
- Size confound: r=0.732
- Best PC4: r=0.865 (much cleaner, r_size=0.213)

## Key Findings

### 1. Perceptual Confounding
Models entangle speed and size representations even though training data is orthogonal (r=0.003 in metadata). This is a property of how models perceive motion.

**Implication**: Difference-of-means captures mixed signals. For clean speed vectors.

### 2. Zero Train/Test Overlap Design
**Training**: sphere + {small,medium,large} + {red,blue,green} + matte + plain

**Test baselines** use compositionally novel objects:
- SIZE: cube + yellow + glossy
- COLOR: cylinder + xlarge + metallic
- SHAPE: xlarge + yellow + glossy
- MATERIAL: capsule + xlarge + yellow
- TEXTURE: cube + xlarge + yellow

This ensures **no overlap** between train and test, enabling strong claims about object-agnostic speed representations.

## Experimental Design

### Training Set (135 videos)
- **Configuration**: 3 sizes × 3 colors × 5 speeds (5-45 px/frame) × 3 repetitions
- **Objects**: sphere + {small,medium,large} + {red,blue,green} + matte + plain
- **Purpose**: Learn speed concept with stratified sampling to control confounds

### Test Sets (Zero Overlap)
Each category tests transfer to compositionally novel objects:
- **SIZE** (80 videos): cube + yellow + glossy + varying sizes
- **COLOR** (120 videos): cylinder + xlarge + metallic + varying colors
- **SHAPE** (80 videos): varying shapes + xlarge + yellow + glossy
- **MATERIAL** (60 videos): capsule + xlarge + yellow + varying materials
- **TEXTURE** (60 videos): cube + xlarge + yellow + varying textures

See `TRAIN_TEST_SPLIT.md` for complete overlap verification.

### Model Configuration
- **V-JEPA**: ViT-Giant, layer 9, hidden_dim=1408
- **VideoMAE v2**: ViT-Giant, layer 9, hidden_dim=1408
- **Input**: 224×224 resolution, 16 frames per video
- **Checkpoints**: Place in `models/` directory (see config.py)

## Methods

### Stratified Difference-of-Means (Default)
```bash
--method stratified --n-per-stratum 4
```
Balances size/color distributions when selecting high/low speed groups.
- **Pro**: Causal interpretation, controls confounds
- **Con**: Moderate size confounds remain (r_size=0.51-0.73)
- **When to use**: For interpretable steering with confound control

### Percentile-based
```bash
--method percentile
```
Selects top/bottom 33% by speed without stratification.
- **Pro**: Simple, uses more samples
- **Con**: Higher confounds, less control
- **When to use**: Quick baseline comparison

### PCA-based (Post-hoc Analysis)
Not a training method - compare steering vectors to principal components.
- **Pro**: Cleanest speed representations (r_size=0.04-0.21)
- **Con**: No causal interpretation, unsupervised
- **Result**: PC4 (VideoMAE) and PC3 (V-JEPA) outperform difference-of-means