# GAMMA-CNN
*A Genetic Algorithm–Driven Convolutional Neural Network with Multi-Modal Fusion for Pulsar Candidate Identification*

This mini-package provides a **Genetic Algorithm (GA)** to automatically search CNN architectures for pulsar candidate identification on HTRU, FAST-like datasets, with optional **multi‑modal fusion** (flexible modality combinations, early/mid/late fusion, add/concat/attention).

---

# 1) Dataset Format & Preprocessing

GAMMA-CNN expects a compact pickle (`.pkl`) file with aligned modalities and consistent dimensions:

```
{
  "train": {
    "profile": (N, F),
    "dm":      (N, F),
    "subband": (N, F, F),
    "subint":  (N, F, F),
    "pdm":     (N, F, F),
    "diag":    (N, F, F),
    "label":   (N,)
  },
  "test": { ... }
}
```

Note: The input dimensions of the dataset are not fixed. GAMMA-CNN supports 1D, 2D, and 3D inputs (e.g., RGB-like multi-channel images). Examples include:

- "subband": (N, 3, F, F)       # 3-channel 2D image
- "profile": (N, F, F)          # 2D profile representation
- "dm":      (N, F)             # 1D curve

As long as the array shapes are consistent within each modality, these formats are fully supported.

We strongly recommend keeping the spatial resolution F consistent across all modalities. When all inputs share the same F, the model trains directly on the original resolution with no additional resizing.

If different modalities have different resolutions (e.g., F vs. F'), GAMMA-CNN applies **Automatic shape alignment** during multi-modal training:

- 1D arrays → resized to **64**
- 2D arrays → resized to **64×64**
- 3D arrays → resized to **3×64×64**

This alignment unifies modality shapes for fusion, but may introduce information loss for high-resolution inputs. Users should be aware of this when preparing datasets.


### Modality availability

GAMMA-CNN supports **any number of modalities**, from **single-modality** to **multi-modality**, depending on the dataset provided by the user.

#### Single-modality example (subband only)

If the dataset contains only one modality (e.g., subband), users provide:

```
dataset_subband = {
"x_train": train_sb,
"y_train": train_y,
"x_test": test_sb,
"y_test": test_y,
}
```
The system automatically builds a **single-stream CNN** (fusion disabled).

---

#### Multi-modal example (4 modalities)

If four modalities are available (profile, DM curve, subband, subintegration), users provide:

```
dataset_multi = {
"x_train_multi": [train_profile, train_dm, train_sb, train_si],
"y_train": train_y,
"x_test_multi": [test_profile, test_dm, test_sb, test_si],
"y_test": test_y,
}
```

GAMMA-CNN automatically constructs a **four-stream network** and applies fusion when `fusion=True`.

---

#### Flexible subsets of modalities

GAMMA-CNN also supports **arbitrary subsets** of the available modalities. For example:

Two-modal fusion:
```
x_train_multi = [train_sb, train_si]
```
→ system builds a **two-stream network**

Three-modal fusion:

```
x_train_multi = [train_profile, train_sb, train_si]
```
→ system builds a **three-stream network**

The framework automatically detects the number of modalities based on the list length, aligns their dimensions, and constructs the corresponding multi-stream architecture — **no placeholder or mask is required**.





---

# 2) Quick Start

```
python HTRUtest.py
```

This loads the dataset, builds multi-modal inputs, initializes AutoCNN, runs GA search, and produces Excel logs.

---

# 3) GA / Model Hyperparameters (`AutoCNN`)

## Full Parameter Table (with Required / Recommended / Default)

| Parameter | Required? | Recommended Range | Default | Meaning |
|----------|-----------|-------------------|---------|---------|
| population_size | **Yes** | 5–50 | — | Individuals per generation |
| maximal_generation_number | **Yes** | 1–50 | — | Number of GA iterations |
| dataset | **Yes** | — | — | Multi-modal dataset dict |
| output_head_builder | No | — | None | Custom head builder |
| epoch_number | No | 5–30 | 1 | Epochs per individual |
| optimizer_fn | No | Adam/Sgd | None→Adam | Optimizer factory |
| crossover_probability | No | 0.5–1.0 | 0.9 | Crossover chance |
| mutation_probability | No | 0.1–0.4 | 0.2 | Mutation chance |
| mutation_operation_distribution | No | — | None | Mutation weights |
| fitness_cache | No | — | "fitness.json" | Cache file |
| logs_dir | No | — | "./logs/train_data" | Log directory |
| checkpoint_dir | No | — | "./checkpoints" | Checkpoints |
| device | No | cuda/mps/cpu | "cpu" | Compute device |
| loss_type | No | bce/ce/weighted/focal | "ce" | Loss function |
| class_weight | No | [1.0, w] | None | For weighted loss |
| batch_size | No | 4–256 | 32 | Batch size |
| fusion | No | True/False | False | Enable fusion |
| fusion_pos | No | early/mid/late/all | "early" | Fusion position |
| fusion_type | No | add/concat/attention/mix | "add" | Fusion method |
| num_streams | No | auto | None | # of streams |
| seed | No | — | 42 | Random seed |
| logger_path | No | — | "ga_stats.json" | Stats output |

---

# 3.1 Fusion Details (per reviewer request)

### Supported fusion positions (`fusion_pos`)
- early
- mid
- late
- all → random choice at build time

### Supported fusion types (`fusion_type`)
- add  
- concat → **with automatic 1×1 convolution for channel compression**  
- attention → normalized over modality axis  
- mix → random type at build time

### Attention fusion specifics
- **Normalization axis:** modality dimension (`dim=0`)  
- **Temperature:** τ = **1** (standard softmax)

---

# 3.2 Example pseudocode for flexible modality subsets

```
inputs = load_available_modalities()      # e.g., 3 or 4 modalities
aligned = [resize_and_norm(x) for x in inputs]
features = [stem_block(x) for x in aligned]

if fusion_pos == "early":
    fused = fuse(features)
elif fusion_pos == "mid":
    fused = mid_level_fusion(features)
elif fusion_pos == "late":
    fused = late_fusion(features)

output = classifier(fused)
```

---

# 4) Minimal Usage Example

### **Single-Modality Example (Subband Only)**

```python
from gan_mm_12d_cuda import AutoCNN
import torch

dataset_subband = {
    "x_train": train_sb,   # shape: (N, 64, 64)
    "y_train": train_y,
    "x_test":  test_sb,
    "y_test":  test_y,
}

device = "cuda" if torch.cuda.is_available() else \
         ("mps" if torch.backends.mps.is_available() else "cpu")

ga = AutoCNN(
    population_size=10,
    maximal_generation_number=10,
    dataset=dataset_subband,
    epoch_number=10,
    batch_size=64,
    fusion=False,      # single-modality → fusion disabled
    device=device,
)

best = ga.run("single_subband_log.xlsx")
print("Best:", best)
```



### **Multi-Modal Example (Profile + DM + Subband + Subintegration)**

```python
from gan_mm_12d_cuda import AutoCNN
import torch

dataset_multi = {
    "x_train_multi": [train_profile, train_dm, train_sb, train_si],
    "y_train": train_y,
    "x_test_multi":  [test_profile,  test_dm,  test_sb,  test_si],
    "y_test": test_y,
}

device = "cuda" if torch.cuda.is_available() else \
         ("mps" if torch.backends.mps.is_available() else "cpu")

ga = AutoCNN(
    population_size=20,
    maximal_generation_number=10,
    dataset=dataset_multi,
    epoch_number=10,
    batch_size=128,
    device=device,
    loss_type="bce",
    fusion=True,
    fusion_pos="late",
    fusion_type="attention",
)

best = ga.run("multi_modal_log.xlsx")
print("Best:", best, getattr(best, "_fusion_choice", None))
```

---

# 5) Outputs
- Checkpoints under `./checkpoints/model_<hash>/`
- Cached metrics in `fitness_cache`

---

# 6) Notes
- GA search is compute-heavy; use few epochs during search.
- BCE produces single-logit output with threshold 0.5.
- Inconsistent modality shapes are auto-aligned.

---

This README incorporates all reviewer-requested clarifications.
