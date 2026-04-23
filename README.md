# Multitask Learning & Sensor Fusion using HydraNet

A multi-head deep learning architecture for simultaneous **object detection**, **semantic segmentation**, **localization**, and **depth estimation**, extended with **attention-based sensor fusion** for multimodal RGB + depth/LiDAR inputs.

## Pipeline

```
RGB Image ──────────────────────────────────────────────────────────────┐
                                                                         │
Secondary (Depth / LiDAR) ──► ModalityProjector ──► AttentionFusion ◄──┤
                                 (Conv projection)   (CrossAttn + CBAM)  │
                                                                         │
              ResNet-50 Backbone ──► FPN (P2–P5) ──► Fused Features ────┘
                                                            │
              ┌─────────────────────────────────────────────┤
              ▼            ▼               ▼               ▼
        DetectionHead  SegmentHead   LocalizHead      DepthHead
        (FCOS-style)  (multi-scale   (global pool    (encoder-decoder
       cls+reg+ctr     FPN merge)     + MLP)          + FPN skips)
```

---

## Architecture

### Shared Backbone + FPN
ResNet-50 encodes the RGB input into multi-scale feature maps C2–C5 (strides 4, 8, 16, 32). A Feature Pyramid Network (FPN) produces four pyramid levels P2–P5 all with 256 channels, merging top-down semantic information with fine-grained spatial detail.

### Attention-based Sensor Fusion
When a secondary modality is present (depth map, LiDAR projection):

1. **ModalityProjector** — two-layer conv block maps raw secondary input into 256-channel FPN feature space.
2. **CrossModalAttention** — multi-head attention where RGB feature queries attend to secondary-modality key/value pairs at each FPN level. This lets the network learn *where* the depth signal is relevant for each task.
3. **CBAM** — Convolutional Block Attention Module (channel + spatial) refines the fused feature, suppressing noise and emphasising task-relevant regions.

This fusion is applied independently at all four FPN levels, so coarse semantics and fine spatial detail both benefit from the secondary modality.

### Task Heads

| Head | Design | Output |
|------|--------|--------|
| **Detection** | FCOS anchor-free, 4-conv tower, GroupNorm | Per-level cls logits, exp-activated bbox (l,r,t,b), centerness |
| **Segmentation** | Multi-scale FPN merge → 2× ConvTranspose | Per-pixel class scores at full resolution |
| **Localization** | AdaptiveAvgPool → MLP (512 → 256) | Per-class (x, y, w, h) + orientation (sin θ, cos θ) |
| **Depth** | Skip-connection decoder P5 → P2, Sigmoid × max_depth | Per-pixel depth map |

### Multi-task Loss — Uncertainty Weighting
Each task's loss is weighted by a learnable precision term `1/σᵢ²`, with regularisation `log σᵢ` (Kendall et al., CVPR 2018):

```
L_total = Σᵢ [ (1/2σᵢ²) · Lᵢ + log σᵢ ]
```

`log σᵢ²` is a free parameter per task, automatically balancing all task losses during training without hand-tuning weights.

### Depth Head — Three-Phase Fine-tuning
Adds the depth head to a pretrained HydraNet checkpoint **without retraining from scratch** by progressively unfreezing layers:

| Phase | What trains | LR multiplier |
|-------|-------------|---------------|
| 1 (epochs 1 – 5) | `depth_head` only | 1× |
| 2 (epochs 6 – 15) | `depth_head` + FPN | 0.5× |
| 3 (epochs 16+) | Full network | 0.1× |

---

## Installation

```bash
git clone https://github.com/sri-godugu/multitask-learning-and-sensor-fusion-using-hydranets.git
cd multitask-learning-and-sensor-fusion-using-hydranets

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU note**: A CUDA-capable GPU is strongly recommended for training. Inference runs on CPU but is significantly slower.

---

## Usage

### Train all tasks end-to-end
```bash
python scripts/train.py --config configs/hydranet.yaml --data /path/to/data

# With backbone warm-up (backbone frozen for first 5 epochs)
python scripts/train.py --config configs/hydranet.yaml --data /path/to/data \
    --freeze-backbone --epochs 50 --batch-size 8
```

### Fine-tune the depth head on a pretrained checkpoint
```bash
python scripts/finetune_depth.py \
    --checkpoint checkpoints/hydranet_epoch050.pt \
    --data /path/to/depth_data \
    --phase1-epochs 5 --phase2-epochs 10 --epochs 20
```

### Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/hydranet_epoch050.pt \
    --data /path/to/data --split val
```

### Single-image inference
```bash
# RGB only
python scripts/inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/hydranet_epoch050.pt \
    --output output/

# With depth/LiDAR fusion
python scripts/inference.py \
    --image path/to/rgb.jpg \
    --depth-input path/to/depth.png \
    --checkpoint checkpoints/hydranet_epoch050.pt
```

---

## Data Format

```
data/
  images/          *.jpg or *.png
  depth/           *.png  (16-bit, millimetres — loaded as metres)
  seg_labels/      *.png  (uint8 per-pixel class IDs)
  det_labels/      *.txt  (YOLO format: class cx cy w h, normalised)
  train.txt        (optional: list of sample stems)
  val.txt
```

---

## Project Structure

```
├── configs/
│   ├── hydranet.yaml          # Model architecture + active tasks
│   ├── training.yaml          # Learning rate, scheduler, batch size
│   └── fusion.yaml            # Sensor fusion settings
├── src/
│   ├── backbone/
│   │   ├── resnet.py          # ResNet feature extractor (C2–C5)
│   │   └── fpn.py             # Feature Pyramid Network
│   ├── heads/
│   │   ├── detection.py       # FCOS-style anchor-free detection
│   │   ├── segmentation.py    # Multi-scale FPN segmentation
│   │   ├── localization.py    # Location + orientation regression
│   │   └── depth.py           # Skip-connection depth decoder
│   ├── fusion/
│   │   ├── attention.py       # ChannelAttention, SpatialAttention, CBAM, CrossModalAttention
│   │   └── fusion_module.py   # Per-level AttentionFusionModule
│   ├── models/
│   │   └── hydranet.py        # Main HydraNet (backbone + FPN + fusion + heads)
│   ├── losses/
│   │   ├── multitask_loss.py  # Uncertainty-weighted total loss
│   │   ├── detection_loss.py  # Focal loss + smooth-L1
│   │   ├── segmentation_loss.py
│   │   ├── localization_loss.py
│   │   └── depth_loss.py      # Scale-invariant log loss + gradient term
│   ├── data/
│   │   ├── dataset.py         # MultiTaskDataset
│   │   └── transforms.py      # Augmentations + normalisation
│   └── utils/
│       ├── freeze_utils.py    # freeze/unfreeze helpers + summary printer
│       ├── metrics.py         # mIoU, AbsRel, RMSE, delta<1.25
│       └── visualization.py   # Colour-mapped segmentation + depth outputs
├── scripts/
│   ├── train.py               # End-to-end multi-task training
│   ├── finetune_depth.py      # 3-phase selective depth-head fine-tuning
│   ├── evaluate.py            # Quantitative evaluation
│   └── inference.py           # Single-image inference + visualisation
├── tests/
│   ├── test_backbone.py
│   ├── test_heads.py
│   ├── test_fusion.py
│   ├── test_hydranet.py
│   └── test_losses.py
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Results

*(To be added after GPU evaluation)*

---

## References

- Kendall, A., Gal, Y., & Cipolla, R. (2018). [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115). *CVPR 2018*.
- Lin, T.-Y. et al. (2017). [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144). *CVPR 2017*.
- Woo, S. et al. (2018). [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521). *ECCV 2018*.
- Eigen, D., Puhrsch, C., & Fergus, R. (2014). [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283). *NeurIPS 2014*.
- Tian, Z. et al. (2019). [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355). *ICCV 2019*.
