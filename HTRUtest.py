# NOTE: This revision contains clean English comments and docstrings only. No functional changes.

import pickle
import numpy as np
import torch
import warnings
from gamma_cnn import AutoCNN

warnings.filterwarnings("ignore", category=FutureWarning)


def load_compact_dataset(pkl_path):
    """Load pre-split train/test dataset from pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    train = data["train"]
    test  = data["test"]

    return (
        train["profile"], train["dm"], train["subband"], train["subint"], train["pdm"], train["diag"], train["label"],
        test["profile"],  test["dm"],  test["subband"],  test["subint"],  test["pdm"],  test["diag"],  test["label"]
    )


# -----------------------------
# 1. Load dataset
# -----------------------------
pkl_path = "251024_HTRU_limited.pkl"

(train_profile, train_dm, train_sb, train_si, train_pdm, train_diag, train_y,
 test_profile,  test_dm,  test_sb,  test_si,  test_pdm,  test_diag,  test_y) = load_compact_dataset(pkl_path)

print("✅ Data loaded successfully!")
print(f"Train set: {train_dm.shape[0]} | Test set: {test_dm.shape[0]}")


# -----------------------------
# 2. Convert 1D data to 2D (expand to [N, 64, 64])
# -----------------------------
train_profile2d = np.tile(train_profile[:, None, :], (1, 64, 1))
test_profile2d  = np.tile(test_profile[:,  None, :], (1, 64, 1))
train_dm2d      = np.tile(train_dm[:, None, :], (1, 64, 1))
test_dm2d       = np.tile(test_dm[:,  None, :], (1, 64, 1))


# -----------------------------
# 3. Build multi-branch dataset (profile + DM + subbands + subints)
# -----------------------------
dataset_multi = {
    "x_train_multi": [train_profile2d, train_dm2d, train_sb, train_si],
    "y_train": train_y,
    "x_test_multi":  [test_profile2d,  test_dm2d,  test_sb,  test_si],
    "y_test": test_y
}

print("Input shape of first modality:", dataset_multi["x_train_multi"][0].shape)


# -----------------------------
# 4. Initialize Genetic Algorithm (GA)
# -----------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

ga = AutoCNN(
    population_size=20,
    maximal_generation_number=10,
    dataset=dataset_multi,
    epoch_number=10,
    batch_size=128,
    device=device,
    loss_type="bce",           # "ce" / "weighted" / "focal"
    fusion=True,
    fusion_pos="late",         # "early" / "mid" / "late" / "all"
    fusion_type="attention",   # "add" / "concat" / "attention" / "mix"
)

# -----------------------------
# 5. Run Evolution Process
# -----------------------------
best_cnn = ga.run("single_diag_t10_t10.xlsx")

print("✅ Evolution completed!")
print("Best structure:", "-".join(map(str, best_cnn.layers)))
print("Fusion configuration:", getattr(best_cnn, "_fusion_choice", None))
