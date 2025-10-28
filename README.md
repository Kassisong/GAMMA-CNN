
# GAMMA-CNN (AutoCNN for Pulsar Candidate Identification)

This mini-package provides a **Genetic Algorithm (GA)** to search CNN structures for pulsar candidate identification on HTRU, FAST-like datasets, with optional **multi‑modal fusion** 

> **Non‑breaking cleanup**: The code in `cnn_structure.py`, `gamma_cnn.py`, and `HTRUtest.py` was annotated with concise English docstrings and a few unused imports were removed. **No functionality was changed.**

---

## 1) Dataset format & preprocessing

The preprocessing can be adapted to different datasets, as long as all input modalities are aligned and have consistent dimensions.
Below is the expected structure of a compact pickle (`.pkl`) file used by `HTRUtest.py`:

```python
{
  "train": {  # each key -> ndarray
    "profile": (N_train, feature_size),         # 1D pulse profile
    "dm":      (N_train, feature_size),         # 1D DM curve
    "subband": (N_train, feature_size, feature_size),     # 2D sub-band image
    "subint":  (N_train, feature_size, feature_size),     # 2D sub-integration image
    "pdm":     (N_train, feature_size, feature_size),     # 2D period-DM (optional here)
    "diag":    (N_train, feature_size, feature_size),     # 2D diagnostic (optional here)
    "label":   (N_train,)             # int labels {0,1} (RFI=0, Pulsar=1)
  },
  "test": { ... same keys/shapes for test split ... }
}
```

**Normalization**: arrays should be `float32` and normalized to a comparable range (e.g., `[0,1]`).

If you feed **multi‑modal inputs** directly through `AutoCNN` (`fusion=True`), inconsistent shapes are supported: 1D streams are resized to length **64** and 2D streams to **64×64** using linear interpolation (pandas).

---

## 2) Quick start (HTRUtest)

1. Place your dataset pickle in the working directory (or adjust the path in `HTRUtest.py`).
2. Run:

```bash
python HTRUtest.py
```

What it does:
- loads the pickle,
- builds a **4‑stream** dataset (`profile`, `dm`, `subband`, `subint`),
- initializes `AutoCNN` with **fusion enabled** (`late + attention` by default),
- runs GA search and writes an Excel log with **two sheets**: `population` and `children`.

> Tip: On Apple Silicon, the script uses `mps` if CUDA is unavailable.

---

## 3) GA / Model hyperparameters (`gan_mm_12d_cuda.AutoCNN`)

| Parameter | Type | Typical Range(recommend) | Meaning |
|---|---:|---:|---|
| `population_size` | int | 5–50 | Individuals per generation. Uniqueness check auto‑enabled when ≤20. |
| `maximal_generation_number` | int | 1–50 | Number of GA iterations. |
| `epoch_number` | int | 5–30 | Training epochs per individual. |
| `batch_size` | int | 4–256 | Mini‑batch size. |
| `optimizer_fn` | callable | — | Optimizer factory (default: Adam). |
| `loss_type` | str | `bce` / `ce` / `weighted` / `focal` | Loss selection. `bce` → 1 logit; `ce` → C logits. |
| `class_weight` | list[float] | e.g. `[1.0, 20.0]` | For `weighted` CE or `bce` (`pos_weight = w1/w0`). |
| **`crossover_probability`** | float | **[0.0, 1.0]** (default **0.9**) | Probability to perform crossover when creating offsprings. |
| **`mutation_probability`** | float | **[0.0, 1.0]** (default **0.2**) | Probability to mutate an offspring. |
| `mutation_operation_distribution` | tuple[float] | sums to 1 | Weights for mutation ops: `('add_skip','add_pooling','remove','change')`. |
| `fitness_cache` | str | path | JSON file storing evaluated metrics. |
| `logs_dir` / `checkpoint_dir` | str | paths | Where logs and checkpoints are stored. |
| `device` | str | `"cuda"` / `"mps"` / `"cpu"` | Compute device string. |

### Fusion controls
| Parameter | Type | Values | Effect |
|---|---:|---|---|
| `fusion` | bool | `True/False` | Enable multi‑modal fusion (needs ≥2 streams). |
| `fusion_pos` | str | `early` / `mid` / `late` / `all` | Where to fuse (`all` randomly picks one at build time). |
| `fusion_type` | str | `add` / `concat` / `attention` / `mix` | How to fuse (`mix` randomly picks one at build time). |
| `num_streams` | int or `None` | — | If `None`, inferred from `x_train_multi` length or channel count. |

**Inheritance rule**: in fusion mode, each child **inherits** one parent's **(`fusion_pos`, `fusion_type`)** to keep a consistent modality design.

---

## 4) Minimal usage example

```python
from gan_mm_12d_cuda import AutoCNN
import torch

dataset_multi = {
    "x_train_multi": [train_profile, train_dm, train_sb, train_si],
    "y_train": train_y,
    "x_test_multi":  [test_profile,  test_dm,  test_sb,  test_si],
    "y_test": test_y,
}

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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
    crossover_probability=0.9,
    mutation_probability=0.2,
)

best = ga.run("evolution_log.xlsx")
print("Best:", best, getattr(best, "_fusion_choice", None))
```

---

## 5) Outputs

- **Checkpoints**: best weights per individual under `./checkpoints/model_<hash>/model_<hash>.pt`.
- **Fitness cache**: the JSON file configured by `fitness_cache` accumulates metrics across runs.

---

## 6) Notes

- GA search is compute‑heavy; use small epochs during search and re‑train your **best** structure longer afterwards.
- With `loss_type="bce"`, a single logit is emitted; metrics use a default 0.5 threshold.
- On inconsistent modality sizes, the code automatically resizes 1D to `(64)` and 2D to `64×64` using linear interpolation.
